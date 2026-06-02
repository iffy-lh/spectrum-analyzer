/**
 ****************************************************************************************************
 * @file        main.c
 * @brief       频谱分析仪 — 优化版
 *   特性: 汉宁窗 / 峰值保持+衰减 / 幅度渐变颜色 / 满屏跳动
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
#define DEMO_MODE       1
#define FFT_SIZE        256
#define SAMPLE_RATE     40000
#define DISP_BINS       128
#define BAR_MAX_H       180      /* 柱子最大像素高度 */
#define PEAK_DECAY      3        /* 峰值下落速度(像素/帧) */

/*==========================================================================
 * 全局变量
 *==========================================================================*/
uint16_t g_adc_buf[FFT_SIZE];
float    g_fft_in[FFT_SIZE * 2];
float    g_fft_out[FFT_SIZE];
float    g_peak[128];                /* 每根柱子的峰值保持 (NBARS=128) */
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
 * 颜色生成 — 幅度→渐变热力图 (低→绿, 中→黄, 高→红)
 *==========================================================================*/
static uint16_t heatmap_color(float ratio)
{
    uint8_t r, g;

    if (ratio < 0.0f) ratio = 0.0f;
    if (ratio > 1.0f) ratio = 1.0f;

    if (ratio < 0.5f)
    {
        /* 绿→黄: G满, R递增 */
        g = 31;
        r = (uint8_t)(ratio * 2.0f * 31.0f);
    }
    else
    {
        /* 黄→红: R满, G递减 */
        r = 31;
        g = (uint8_t)((1.0f - ratio) * 2.0f * 31.0f);
    }

    return (uint16_t)((r << 11) | (g << 5));
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
    uint32_t i;

    RCC->APB2ENR |= 1 << 2;
    GPIOA->CRL &= ~(0xF << 4);

    RCC->APB2ENR |= 1 << 9;
    RCC->CFGR &= ~(3 << 14);
    RCC->CFGR |= 2 << 14;

    ADC1->CR1 = 0;
    ADC1->CR2 = 0;
    ADC1->CR2 |= 7 << 17;
    ADC1->CR2 |= 1 << 20;
    ADC1->CR2 |= 1 << 8;
    ADC1->SQR1 = 0 << 20;
    ADC1->SQR3 = 1 << 0;
    ADC1->SMPR2 |= 7 << 3;

    ADC1->CR2 |= 1 << 0;
    for(i=0;i<100;i++) __NOP();
    ADC1->CR2 |= 1 << 3;
    while(ADC1->CR2 & (1<<3));
    ADC1->CR2 |= 1 << 2;
    while(ADC1->CR2 & (1<<2));

    RCC->AHBENR |= 1 << 0;
    delay_ms(5);

    DMA1_Channel1->CCR = 0;
    DMA1_Channel1->CPAR = (uint32_t)&ADC1->DR;
    DMA1_Channel1->CMAR = (uint32_t)g_adc_buf;
    DMA1_Channel1->CNDTR = FFT_SIZE;
    DMA1_Channel1->CCR |= 1 << 1;
    DMA1_Channel1->CCR |= 1 << 5;
    DMA1_Channel1->CCR |= 1 << 7;
    DMA1_Channel1->CCR |= 1 << 8;
    DMA1_Channel1->CCR |= 1 << 10;
    DMA1_Channel1->CCR |= 1 << 12;
    DMA1_Channel1->CCR |= 1 << 0;

    sys_nvic_init(3, 3, DMA1_Channel1_IRQn, 2);
    ADC1->CR2 |= 1 << 0;
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
        val = 2048.0f;

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
        g_fft_in[2*i]   = ((float)g_adc_buf[i] - 2048.0f) * hanning_window[i];
        g_fft_in[2*i+1] = 0.0f;
    }

    arm_cfft_radix4_init_f32(&scfft, FFT_SIZE, 0, 1);
    arm_cfft_radix4_f32(&scfft, g_fft_in);
    arm_cmplx_mag_f32(g_fft_in, g_fft_out, FFT_SIZE);
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
 * LCD频谱显示 — 单色密柱 + dB网格 + 频率/幅度轴
 *==========================================================================*/
void draw_spectrum(void)
{
    uint16_t i, x, bar_h, peak_h;
    uint16_t w = lcddev.width;
    uint16_t h = lcddev.height;
    char buf[32];
    float max_mag = 0;
    uint16_t peak_bin = 0;

    lcd_clear(BLACK);

    for (i = 1; i < DISP_BINS; i++)
        if (g_fft_out[i] > max_mag) { max_mag = g_fft_out[i]; peak_bin = i; }
    if (max_mag < 1.0f) max_mag = 1.0f;

    /* === 顶栏 === */
    lcd_fill(0, 0, w, 24, 0x0841);
    float exact_hz = interpolate_peak(peak_bin);
    sprintf(buf, "%d Hz", (int)exact_hz);
    lcd_show_string(5, 2, 100, 16, 16, buf, YELLOW);

    /* 绝对 dBFS: 参考 ADC满量程 (2048*128=262144) */
    float dbfs = 20.0f * log10f(max_mag / 262144.0f + 0.00001f);
    sprintf(buf, "%.0f dBFS", dbfs);
    lcd_show_string(115, 2, 80, 16, 16, buf, 0x7E8C);

    /* === 参数 === */
    #define NBARS      128
    #define BAR_MX_H   184
    #define BAR_BASE_Y 212
    #define BAR_COLOR   0x07E0       /* 纯绿 */
    #define DB_RANGE    72.0f        /* 显示范围: 0 ~ -72 dBFS */
    #define DB_REF      262144.0f    /* 0 dBFS 参考值 (2048 amplitude * 128 FFT gain) */

    /* --- dB网格 (0, -24, -48, -72 dBFS) --- */
    for (i = 0; i < 4; i++)
    {
        float db_val = (float)i * (-DB_RANGE / 3.0f);
        uint16_t gy = BAR_BASE_Y - (uint16_t)((1.0f - (float)i / 3.0f) * BAR_MX_H);
        for (x = 0; x < w; x += 8)
            lcd_draw_line(x, gy, x + 3, gy, 0x2104);
        sprintf(buf, "%.0f", db_val);
        lcd_show_string(2, gy - 5, 28, 12, 12, buf, GRAY);
    }

    /* --- 频率标尺 --- */
    lcd_draw_line(0, BAR_BASE_Y+1, w, BAR_BASE_Y+1, 0x2104);
    lcd_show_string(0,         BAR_BASE_Y+3, 30, 12, 12, "0.1k", GRAY);
    lcd_show_string(w*1/6-10,  BAR_BASE_Y+3, 30, 12, 12, "0.5k", GRAY);
    lcd_show_string(w*2/6-10,  BAR_BASE_Y+3, 30, 12, 12, "1k", GRAY);
    lcd_show_string(w*3/6-10,  BAR_BASE_Y+3, 40, 12, 12, "5k", GRAY);
    lcd_show_string(w*4/6-10,  BAR_BASE_Y+3, 40, 12, 12, "10k", GRAY);
    lcd_show_string(w*5/6-10,  BAR_BASE_Y+3, 40, 12, 12, "20k", GRAY);

    /* --- 128根密柱(0缝隙) --- */
    uint16_t bar_w = w / NBARS;    /* ≈2.5px */

    for (i = 0; i < NBARS; i++)
    {
        uint16_t bin = i + 1;
        float dbfs = 20.0f * log10f(g_fft_out[bin] / DB_REF + 0.00001f);
        float db_norm = (dbfs + DB_RANGE) / DB_RANGE;   /* -72dB→0, 0dB→1 */
        if (db_norm < 0.0f) db_norm = 0.0f;
        if (db_norm > 1.0f) db_norm = 1.0f;

        bar_h = (uint16_t)(db_norm * (float)BAR_MX_H);

        /* 峰值保持 */
        if (bar_h > g_peak[i])
            g_peak[i] = (float)bar_h;
        else if (g_peak[i] > 4.0f)
            g_peak[i] -= 4.0f;
        else
            g_peak[i] = 0;
        peak_h = (uint16_t)g_peak[i];
        if (peak_h > BAR_MX_H) peak_h = BAR_MX_H;

        x = i * bar_w;

        /* 画柱(无缝) */
        if (bar_h > 0)
            lcd_fill(x, BAR_BASE_Y - bar_h, x + bar_w, BAR_BASE_Y, BAR_COLOR);

        /* 峰值白点 */
        if (peak_h > bar_h)
            lcd_fill(x, BAR_BASE_Y - peak_h, x + bar_w, BAR_BASE_Y - peak_h, WHITE);
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

    /* 初始化汉宁窗 */
    hanning_init();

    /* 初始化峰值数组 */
    for (i = 0; i < 128; i++) g_peak[i] = 0;

    LED0(0); delay_ms(200);
    LED0(1); delay_ms(200);
    LED0(0); delay_ms(200);
    LED0(1);

    lcd_init();

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
    lcd_show_string(30, 70, 200, 16, 16, "F1 Elite - ADC Mode", RED);
    lcd_show_string(30, 100, 200, 12, 12, "Init ADC+DMA...", GRAY);

    tim2_trigger_init(99, 17);
    adc_dma_spectrum_init();

    lcd_show_string(30, 100, 200, 12, 12, "Ready! Mic->PA1", GREEN);
    delay_ms(1000);

    while (1)
    {
        if (g_ready)
        {
            g_ready = 0;
            do_fft();
            draw_spectrum();
            LED0_TOGGLE();
        }
        delay_ms(10);
    }
#endif
}
