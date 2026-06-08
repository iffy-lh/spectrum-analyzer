/**
 ****************************************************************************************************
 * @file        main.c
 * @brief       频谱分析仪 — OLED风格优化版
 *   特性: 汉宁窗+幅度补偿 / 对数频率轴 / 快升慢降波浪感 / 亮蓝单色零缝隙
 *   技术栈: CMSIS-DSP + ADC/DMA + 正点原子LCD
 ****************************************************************************************************
 */

#include "./SYSTEM/sys/sys.h"
#include "./SYSTEM/usart/usart.h"
#include "./SYSTEM/delay/delay.h"
#include "./BSP/LED/led.h"
#include "./BSP/LCD/lcd.h"
#include "./BSP/ADC/adc.h"
#include "./BSP/DMA/dma.h"
#include "./CMSIS/DSP/Include/arm_math.h"
#include "math.h"

/*==========================================================================
 * 参数
 *==========================================================================*/
#define DEMO_MODE       0
#define FFT_SIZE        256
#define SAMPLE_RATE     47619    /* ADC连续模式: 12MHz/252cycle≈47.6kHz */

/* --- MAX9814 麦克风模块 ---
 * VCC=3.3V时输出偏置≈1.65V→ADC值2048
 * VCC=5V时偏置≈2.5V→ADC值3103
 * 用万用表测模块OUT引脚对GND电压, 改下面的值 */
#define MAX9814_DC      2048    /* ADC直流偏置值 */

#define DISP_BINS       128
#define NBARS           128      /* 显示柱数 */
#define FALL_RATE       2.0f     /* 慢降速度(像素/帧), 越大下落越快 */
#define BAR_COLOR       0x07FF   /* 亮蓝/青色 */
#define TOPBAR_H        18       /* 顶栏高度 */
#define DB_RANGE        60.0f    /* 动态范围: 0 ~ -60 dB */

/*==========================================================================
 * 全局变量
 *==========================================================================*/
uint16_t g_adc_buf[FFT_SIZE];
float    g_fft_in[FFT_SIZE * 2];
float    g_fft_out[FFT_SIZE];
float    g_smooth[NBARS];            /* 每根柱子的包络高度(快升慢降) */
volatile uint8_t g_ready = 0;

/*==========================================================================
 * 汉宁窗系数 (预计算)
 *==========================================================================*/
static float hanning_window[FFT_SIZE];

static void hanning_init(void)
{
    uint16_t i;
    for (i = 0; i < FFT_SIZE; i++)
        hanning_window[i] = 0.5f * (1.0f - arm_cos_f32(2.0f * PI * i / (FFT_SIZE - 1)));
}

/*==========================================================================
 * 对数频率分箱映射 (预计算: FFT bin → 显示柱)
 *==========================================================================*/
static uint8_t g_bin_to_bar[FFT_SIZE];   /* FFT bin i → 显示柱 col */

static void log_binmap_init(void)
{
    uint16_t i;
    float f_min = (float)SAMPLE_RATE / FFT_SIZE;   /* bin 1 = 156.25 Hz */
    float f_max = (float)SAMPLE_RATE / 2.0f;        /* Nyquist = 20 kHz */
    float log_ratio = log10f(f_max / f_min);

    g_bin_to_bar[0] = 0;  /* DC bin → column 0 */
    for (i = 1; i < FFT_SIZE; i++)
    {
        float freq = (float)i * SAMPLE_RATE / FFT_SIZE;
        float col;
        if (freq <= f_min)
            col = 0.0f;
        else
            col = (log10f(freq / f_min) / log_ratio) * (float)(NBARS - 1);

        if (col < 0.0f)        col = 0.0f;
        if (col >= (float)NBARS) col = (float)(NBARS - 1);
        g_bin_to_bar[i] = (uint8_t)(col + 0.5f);
    }
}

/*==========================================================================
 * TIM2 → ADC触发 (40kHz)
 *==========================================================================*/
void tim2_trigger_init(uint16_t arr, uint16_t psc)
{
    RCC->APB1ENR |= 1 << 0;
    TIM2->PSC = psc;
    TIM2->ARR = arr;
    TIM2->CR2 &= ~(7 << 4);
    TIM2->CR2 |= 2 << 4;
    TIM2->EGR |= 1 << 0;
    TIM2->CR1 |= 1 << 0;
}

/*==========================================================================
 * ADC + DMA 初始化
 *==========================================================================*/
void adc_dma_spectrum_init(void)
{
    /* 使用正点原子BSP原生API: adc_dma_init(校准+DMA基础配置) + adc_dma_enable(启动) */
    adc_dma_init((uint32_t)g_adc_buf);
    adc_dma_enable(FFT_SIZE);
}

void DMA1_Channel1_IRQHandler(void)
{
    if (DMA1->ISR & (1 << 1))
    {
        g_ready = 1;
        DMA1->IFCR |= 1 << 1;
    }
}

/*==========================================================================
 * 模拟音乐测试信号 — 鼓点 + 贝斯 + 旋律
 *==========================================================================*/
void gen_test_signal(void)
{
    uint16_t i;
    static uint32_t tick = 0;    /* 节拍计数器, 约60fps */
    float val;
    tick++;

    /* 音乐参数: 用tick控制节奏 */
    uint32_t beat = tick / 50;           /* 约每0.8秒一个节拍 */

    /* 贝斯音符序列 (bin值: 低频) */
    float bass_notes[] = {4, 5, 6, 5, 4, 5, 7, 5};  /* 8个音符循环 */
    float bass_bin = bass_notes[beat % 8];

    /* 旋律音符序列 (bin值: 中频) */
    float mel_notes[] = {20, 24, 28, 24, 30, 28, 24, 20,
                         22, 26, 30, 26, 32, 30, 26, 22};
    float mel_bin = mel_notes[beat % 16];

    /* 和弦音符 (pad/氛围音) */
    float pad_notes[] = {12, 16, 20};

    for (i = 0; i < FFT_SIZE; i++)
    {
        val = (float)MAX9814_DC;

        /* 1. 底鼓: 每拍一下, 低频冲击 */
        float kick_env = (tick % 50 < 8) ? (1.0f - (float)(tick % 50) / 8.0f) : 0.0f;
        val += 1200.0f * kick_env * arm_sin_f32(2.0f * PI * 3.0f * i / FFT_SIZE);

        /* 2. 贝斯线: 持续低频, 跟着音符变 */
        float bass_amp = 500.0f + 300.0f * arm_sin_f32(2.0f * PI * (float)(tick % 100) / 100.0f);
        val += bass_amp * arm_sin_f32(2.0f * PI * bass_bin * i / FFT_SIZE);
        val += bass_amp * 0.5f * arm_sin_f32(2.0f * PI * bass_bin * 2.0f * i / FFT_SIZE);

        /* 3. 军鼓/拍手: 每2拍一下 */
        float snare_env = ((tick % 100) > 45 && (tick % 100) < 52) ? 0.8f : 0.0f;
        val += 600.0f * snare_env * arm_sin_f32(2.0f * PI * 18.0f * i / FFT_SIZE);
        /* 军鼓高频噪声 */
        val += 200.0f * snare_env * arm_sin_f32(2.0f * PI * 55.0f * i / FFT_SIZE);

        /* 4. 主旋律: 中频, 跟音符走 */
        float mel_env = 0.7f + 0.3f * arm_sin_f32(2.0f * PI * (float)(tick % 30) / 30.0f);
        val += 400.0f * mel_env * arm_sin_f32(2.0f * PI * mel_bin * i / FFT_SIZE);

        /* 5. 和弦铺垫: 持续中低频, 营造氛围 */
        for (int p = 0; p < 3; p++)
        {
            float pad_amp = 150.0f + 50.0f * arm_sin_f32(2.0f * PI * (float)((tick + p*20) % 80) / 80.0f);
            val += pad_amp * arm_sin_f32(2.0f * PI * pad_notes[p] * i / FFT_SIZE);
        }

        /* 6. 镲片/高频: 每4拍一下 */
        float hat_env = ((beat % 4 == 0) && (tick % 50 < 5)) ? 0.6f : 0.0f;
        if (beat % 2 == 0 && tick % 25 < 3) hat_env = 0.3f;  /* 半拍弱镲 */
        val += 250.0f * hat_env * arm_sin_f32(2.0f * PI * 80.0f * i / FFT_SIZE);
        val += 150.0f * hat_env * arm_sin_f32(2.0f * PI * 95.0f * i / FFT_SIZE);

        /* 7. 微量白噪声(模拟真实感) */
        val += (float)((i * 7919 + tick * 6271) % 128 - 64) * 0.2f;

        if (val < 0)   val = 0;
        if (val > 4095) val = 4095;
        g_adc_buf[i] = (uint16_t)val;
    }
}

/*==========================================================================
 * FFT (汉宁窗 + CMSIS-DSP)
 *==========================================================================*/
void do_fft(void)
{
    uint16_t i;
    arm_cfft_radix4_instance_f32 scfft;

    for (i = 0; i < FFT_SIZE; i++)
    {
        g_fft_in[2*i]   = ((float)g_adc_buf[i] - (float)MAX9814_DC) * hanning_window[i];
        g_fft_in[2*i+1] = 0.0f;
    }

    arm_cfft_radix4_init_f32(&scfft, FFT_SIZE, 0, 1);
    arm_cfft_radix4_f32(&scfft, g_fft_in);
    arm_cmplx_mag_f32(g_fft_in, g_fft_out, FFT_SIZE);

    /* 汉宁窗幅度补偿: coherent gain = 0.5, 乘2恢复 */
    for (i = 0; i < FFT_SIZE; i++)
        g_fft_out[i] *= 2.0f;
}

/*==========================================================================
 * 抛物线插值 — 亚bin精度峰值频率
 * 输入: bin = 峰值所在bin索引
 * 返回: 精确频率(Hz)
 *==========================================================================*/
static float interpolate_peak(uint16_t bin)
{
    float y0, y1, y2, delta;

    if (bin < 1 || bin >= DISP_BINS - 1) return (float)bin * SAMPLE_RATE / FFT_SIZE;

    y0 = g_fft_out[bin - 1];
    y1 = g_fft_out[bin];
    y2 = g_fft_out[bin + 1];

    /* 抛物线拟合: delta = (y0 - y2) / (2*(y0 - 2*y1 + y2)) */
    float denom = 2.0f * (y0 - 2.0f * y1 + y2);
    if (fabs(denom) < 0.0001f) delta = 0.0f;
    else delta = (y0 - y2) / denom;

    if (delta < -0.5f) delta = -0.5f;
    if (delta >  0.5f) delta =  0.5f;

    return ((float)bin + delta) * SAMPLE_RATE / FFT_SIZE;
}

/*==========================================================================
 * LCD频谱显示 — OLED风格: 亮蓝单色 + 零缝隙 + 快升慢降 + 对数频率轴
 *==========================================================================*/
void draw_spectrum(void)
{
    uint16_t i, x, bar_h;
    uint16_t w = lcddev.width;
    uint16_t h = lcddev.height;
    char buf[32];
    float col_max[NBARS];
    float max_mag = 0;
    uint16_t peak_bin = 0;

    lcd_clear(BLACK);

    /* --- 第1步: 对数分箱 (FFT bins → 显示柱) --- */
    for (i = 0; i < NBARS; i++) col_max[i] = 0.0f;
    for (i = 1; i < FFT_SIZE / 2; i++)
    {
        uint8_t col = g_bin_to_bar[i];
        if (g_fft_out[i] > col_max[col])
            col_max[col] = g_fft_out[i];
    }

    /* --- 第2步: 找全局峰值 --- */
    for (i = 0; i < NBARS; i++)
        if (col_max[i] > max_mag) max_mag = col_max[i];
    for (i = 1; i < FFT_SIZE / 2; i++)
        if (g_fft_out[i] >= max_mag) { peak_bin = i; break; }
    if (max_mag < 1.0f) max_mag = 1.0f;

    /* --- 第3步: 快升慢降包络 --- */
    #define BAR_MX_H   184
    #define BAR_BASE_Y 212
    #define DB_REF     262144.0f

    for (i = 0; i < NBARS; i++)
    {
        float dbfs = 20.0f * log10f(col_max[i] / DB_REF + 0.00001f);
        float db_norm = (dbfs + DB_RANGE) / DB_RANGE;
        if (db_norm < 0.0f) db_norm = 0.0f;
        if (db_norm > 1.0f) db_norm = 1.0f;

        float target = db_norm * (float)BAR_MX_H;

        if (target >= g_smooth[i])
            g_smooth[i] = target;               /* 瞬升 */
        else
        {
            g_smooth[i] -= FALL_RATE;           /* 慢降 */
            if (g_smooth[i] < target) g_smooth[i] = target;
            if (g_smooth[i] < 0.0f)  g_smooth[i] = 0.0f;
        }
    }

    /* === 顶栏 === */
    lcd_fill(0, 0, w, 24, 0x0841);
    float exact_hz = interpolate_peak(peak_bin);
    sprintf(buf, "%d Hz", (int)exact_hz);
    lcd_show_string(5, 2, 100, 16, 16, buf, YELLOW);

    float dbfs = 20.0f * log10f(max_mag / DB_REF + 0.00001f);
    sprintf(buf, "%.0f dBFS", dbfs);
    lcd_show_string(115, 2, 80, 16, 16, buf, 0x7E8C);

    /* --- 底部分隔线 + 频率标尺 --- */
    lcd_draw_line(0, BAR_BASE_Y + 1, w, BAR_BASE_Y + 1, 0x2104);
    lcd_show_string(0,         BAR_BASE_Y + 3, 30, 12, 12, "0.1k", GRAY);
    lcd_show_string(w*1/6-10,  BAR_BASE_Y + 3, 30, 12, 12, "0.5k", GRAY);
    lcd_show_string(w*2/6-10,  BAR_BASE_Y + 3, 30, 12, 12, "1k",   GRAY);
    lcd_show_string(w*3/6-10,  BAR_BASE_Y + 3, 40, 12, 12, "5k",   GRAY);
    lcd_show_string(w*4/6-10,  BAR_BASE_Y + 3, 40, 12, 12, "10k",  GRAY);
    lcd_show_string(w*5/6-10,  BAR_BASE_Y + 3, 40, 12, 12, "20k",  GRAY);

    /* --- 亮蓝密柱, 零缝隙 --- */
    uint16_t bar_w = w / NBARS;

    for (i = 0; i < NBARS; i++)
    {
        bar_h = (uint16_t)g_smooth[i];
        if (bar_h > BAR_MX_H) bar_h = BAR_MX_H;
        if (bar_h < 1) continue;

        x = i * bar_w;
        lcd_fill(x, BAR_BASE_Y - bar_h, x + bar_w + 1, BAR_BASE_Y, BAR_COLOR);
    }
}

/*==========================================================================
 * main
 *==========================================================================*/
int main(void)
{
    uint16_t i;

    sys_stm32_clock_init(9);
    delay_init(72);
    usart_init(72, 115200);
    led_init();

    /* 初始化汉宁窗 + 对数分箱映射 + 包络数组 */
    hanning_init();
    log_binmap_init();
    for (i = 0; i < NBARS; i++) g_smooth[i] = 0.0f;

    LED0(0); delay_ms(200);
    LED0(1); delay_ms(200);
    LED0(0); delay_ms(200);
    LED0(1);

    lcd_init();
    lcd_display_dir(1);     /* 横屏: 320×240, 频率轴沿长边展开 */

#if DEMO_MODE
    lcd_show_string(30, 50, 200, 16, 16, "Spectrum Analyzer", RED);
    lcd_show_string(30, 70, 200, 16, 16, "F1 Elite Board", RED);
    lcd_show_string(30, 100, 200, 12, 12, "DEMO MODE", GREEN);
    delay_ms(1500);

    while (1)
    {
        gen_test_signal();
        do_fft();
        draw_spectrum();
        LED0_TOGGLE();
        delay_ms(60);
    }
#else
    lcd_show_string(30, 50, 200, 16, 16, "Spectrum Analyzer", RED);
    lcd_show_string(30, 70, 200, 16, 16, "F1 Elite + MAX9814", RED);
    lcd_show_string(30, 100, 200, 12, 12, "MAX9814 -> PA1", GREEN);
    delay_ms(1500);

    adc_dma_spectrum_init();

    while (1)
    {
        if (g_ready)
        {
            g_ready = 0;
            do_fft();
            draw_spectrum();
            adc_dma_enable(FFT_SIZE);
            LED0_TOGGLE();
        }
        delay_ms(1);
    }
#endif
}
