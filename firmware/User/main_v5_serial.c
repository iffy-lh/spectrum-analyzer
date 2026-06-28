/**
 ****************************************************************************************************
 * @file        main_v5_serial.c
 * @brief       频谱分析仪 v5 — 串口数据输出版
 *   在 v4 基础上增加: 通过USART1将ADC数据发送到PC上位机
 *
 *   数据协议:
 *     帧头:  0xAA 0x55
 *     长度:  2字节 (小端, 数据区字节数 = SEND_SIZE * 2)
 *     数据:  SEND_SIZE × uint16_t (小端, 从g_adc_buf每2个取1个)
 *     校验:  1字节 (所有数据字节的XOR)
 *
 *   接线: F1精英版 USB_232口 → 电脑USB (CH340G USB转串口)
 *   波特率: 500000 (BRR=9, 精确分频)
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
#define DEMO_MODE       0        /* 0=真实ADC, 1=测试信号 */
#define FFT_SIZE        256      /* FFT点数(保持256) */
#define SEND_SIZE       128      /* 每次发送的采样点数(从256中抽取) */
#define SAMPLE_RATE     47619    /* ADC连续模式: 12MHz/252cycle≈47.6kHz */
#define MAX9814_DC      2048     /* ADC直流偏置值 */

#define DISP_BINS       128
#define NBARS           128
#define FALL_RATE       2.0f
#define BAR_COLOR       0x07FF
#define TOPBAR_H        18
#define DB_RANGE        60.0f

/*==========================================================================
 * 全局变量
 *==========================================================================*/
uint16_t g_adc_buf[FFT_SIZE];
float    g_fft_in[FFT_SIZE * 2];
float    g_fft_out[FFT_SIZE];
float    g_smooth[NBARS];
volatile uint8_t g_ready = 0;

/*==========================================================================
 * 汉宁窗 / 对数分箱 (同v4, 省略重复代码, 需从原有main.c合并)
 *==========================================================================*/
static float hanning_window[FFT_SIZE];

static void hanning_init(void)
{
    uint16_t i;
    for (i = 0; i < FFT_SIZE; i++)
        hanning_window[i] = 0.5f * (1.0f - arm_cos_f32(2.0f * PI * i / (FFT_SIZE - 1)));
}

static uint8_t g_bin_to_bar[FFT_SIZE];

static void log_binmap_init(void)
{
    uint16_t i;
    float f_min = (float)SAMPLE_RATE / FFT_SIZE;
    float f_max = (float)SAMPLE_RATE / 2.0f;
    float log_ratio = log10f(f_max / f_min);

    g_bin_to_bar[0] = 0;
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
 * ADC + DMA 初始化 (使用正点原子BSP)
 *==========================================================================*/
void adc_dma_spectrum_init(void)
{
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
 * ★ v5新增: 串口发送ADC数据到PC ★
 *
 * 协议格式 (每帧):
 *   [0xAA] [0x55] [len_lo] [len_hi] [data ...] [checksum]
 *
 * 数据区: SEND_SIZE(128) × uint16_t, 小端
 * checksum: 数据区所有字节的XOR
 *
 * 耗时估算: 260 bytes × 10 bits/byte / 500000 bps ≈ 5.2ms
 *          DMA周期: 256 / 47619 ≈ 5.4ms
 *          刚好来得及, 不会丢帧
 *==========================================================================*/
void send_adc_over_uart(void)
{
    uint16_t i, j;
    uint8_t checksum = 0;
    uint16_t data_len = SEND_SIZE * 2;          /* 128 × 2 = 256 字节 */
    uint8_t *p;

    /* --- 帧头 --- */
    while (!(USART1->SR & (1 << 7)));   /* 等待TXE */
    USART1->DR = 0xAA;
    while (!(USART1->SR & (1 << 7)));
    USART1->DR = 0x55;

    /* --- 数据长度 (小端, 2字节) --- */
    while (!(USART1->SR & (1 << 7)));
    USART1->DR = (uint8_t)(data_len & 0xFF);
    while (!(USART1->SR & (1 << 7)));
    USART1->DR = (uint8_t)(data_len >> 8);

    /* --- 数据区: 从256个采样点中取偶数索引(128个) --- */
    /* 每2个采样抽1个, 有效采样率 ≈ 23.8kHz, 对噪音分析足够 */
    p = (uint8_t *)g_adc_buf;
    for (i = 0; i < SEND_SIZE; i++)
    {
        j = i * 2;                    /* 每隔1个取1个: 索引 0,2,4,...,254 */
        uint8_t lo = p[j * 2];        /* g_adc_buf[i*2] 的低字节 */
        uint8_t hi = p[j * 2 + 1];    /* g_adc_buf[i*2] 的高字节 */

        checksum ^= lo;
        checksum ^= hi;

        while (!(USART1->SR & (1 << 7)));
        USART1->DR = lo;
        while (!(USART1->SR & (1 << 7)));
        USART1->DR = hi;
    }

    /* --- 校验和 --- */
    while (!(USART1->SR & (1 << 7)));
    USART1->DR = checksum;
}

/*==========================================================================
 * FFT + 显示 (同v4)
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

    for (i = 0; i < FFT_SIZE; i++)
        g_fft_out[i] *= 2.0f;
}

static float interpolate_peak(uint16_t bin)
{
    float y0, y1, y2, delta;
    if (bin < 1 || bin >= DISP_BINS - 1) return (float)bin * SAMPLE_RATE / FFT_SIZE;
    y0 = g_fft_out[bin - 1];
    y1 = g_fft_out[bin];
    y2 = g_fft_out[bin + 1];
    float denom = 2.0f * (y0 - 2.0f * y1 + y2);
    if (fabs(denom) < 0.0001f) delta = 0.0f;
    else delta = (y0 - y2) / denom;
    if (delta < -0.5f) delta = -0.5f;
    if (delta >  0.5f) delta =  0.5f;
    return ((float)bin + delta) * SAMPLE_RATE / FFT_SIZE;
}

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

    for (i = 0; i < NBARS; i++) col_max[i] = 0.0f;
    for (i = 1; i < FFT_SIZE / 2; i++)
    {
        uint8_t col = g_bin_to_bar[i];
        if (g_fft_out[i] > col_max[col])
            col_max[col] = g_fft_out[i];
    }

    for (i = 0; i < NBARS; i++)
        if (col_max[i] > max_mag) max_mag = col_max[i];
    for (i = 1; i < FFT_SIZE / 2; i++)
        if (g_fft_out[i] >= max_mag) { peak_bin = i; break; }
    if (max_mag < 1.0f) max_mag = 1.0f;

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
            g_smooth[i] = target;
        else
        {
            g_smooth[i] -= FALL_RATE;
            if (g_smooth[i] < target) g_smooth[i] = target;
            if (g_smooth[i] < 0.0f)  g_smooth[i] = 0.0f;
        }
    }

    /* 顶栏 */
    lcd_fill(0, 0, w, 24, 0x0841);
    float exact_hz = interpolate_peak(peak_bin);
    sprintf(buf, "%d Hz", (int)exact_hz);
    lcd_show_string(5, 2, 100, 16, 16, buf, YELLOW);

    float dbfs = 20.0f * log10f(max_mag / DB_REF + 0.00001f);
    sprintf(buf, "%.0f dBFS", dbfs);
    lcd_show_string(115, 2, 80, 16, 16, buf, 0x7E8C);

    /* 底部分隔线 + 频率标尺 */
    lcd_draw_line(0, BAR_BASE_Y + 1, w, BAR_BASE_Y + 1, 0x2104);
    lcd_show_string(0,         BAR_BASE_Y + 3, 30, 12, 12, "0.1k", GRAY);
    lcd_show_string(w*1/6-10,  BAR_BASE_Y + 3, 30, 12, 12, "0.5k", GRAY);
    lcd_show_string(w*2/6-10,  BAR_BASE_Y + 3, 30, 12, 12, "1k",   GRAY);
    lcd_show_string(w*3/6-10,  BAR_BASE_Y + 3, 40, 12, 12, "5k",   GRAY);
    lcd_show_string(w*4/6-10,  BAR_BASE_Y + 3, 40, 12, 12, "10k",  GRAY);
    lcd_show_string(w*5/6-10,  BAR_BASE_Y + 3, 40, 12, 12, "20k",  GRAY);

    /* 亮蓝密柱 */
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
 * ★★★ main — v5串口输出版 ★★★
 *==========================================================================*/
int main(void)
{
    uint16_t i;

    sys_stm32_clock_init(9);
    delay_init(72);

    /* --- UART初始化: 先用115200跑通printf, 然后切到高速 --- */
    usart_init(72, 115200);

    /* 重新配置BRR → 500000 bps (72MHz / (16*9) = 500000, 精确) */
    USART1->BRR = 9;   /* BRR = PCLK2/(16×Baud) = 72M/(16×500k) = 9 */
                       /* 注意: 这里直接写9 (DIV_Mantissa=9, DIV_Fraction=0) */

    led_init();

    hanning_init();
    log_binmap_init();
    for (i = 0; i < NBARS; i++) g_smooth[i] = 0.0f;

    LED0(0); delay_ms(200);
    LED0(1); delay_ms(200);
    LED0(0); delay_ms(200);
    LED0(1);

    lcd_init();
    lcd_display_dir(1);

#if DEMO_MODE
    lcd_show_string(30, 50, 200, 16, 16, "Spectrum Analyzer", RED);
    lcd_show_string(30, 70, 200, 16, 16, "F1 Elite v5 SERIAL", RED);
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
    lcd_show_string(30, 30, 260, 16, 16, "Spectrum Analyzer", RED);
    lcd_show_string(30, 50, 260, 16, 16, "v5 SERIAL OUTPUT", YELLOW);
    lcd_show_string(30, 75, 260, 12, 12, "MAX9814 -> PA1", GREEN);
    lcd_show_string(30, 95, 260, 12, 12, "UART: 500000bps -> PC", GREEN);
    delay_ms(1500);

    adc_dma_spectrum_init();

    while (1)
    {
        if (g_ready)
        {
            g_ready = 0;

            /* ★ v5关键: 先把ADC数据发给电脑 ★ */
            send_adc_over_uart();

            /* 然后继续做FFT和显示 (和v4一样) */
            do_fft();
            draw_spectrum();
            adc_dma_enable(FFT_SIZE);
            LED0_TOGGLE();
        }
        delay_ms(1);
    }
#endif
}
