/**
 ****************************************************************************************************
 * @file        main.c  (v7 — USB虚拟串口)
 * @brief       频谱分析仪 v7
 *   基于 v4 完整功能 + USB CDC 虚拟串口输出
 *   用 USB_SLAVE 口连接电脑, 绕开有问题的 CH340G
 *
 *   接线:
 *     MAX9814: VCC→3.3V, GND→GND, OUT→PA1
 *     USB_SLAVE 口 → 电脑 (STM32内置USB, 需装ST官方VCP驱动)
 *     ST-Link 口 → 烧录
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
#include "string.h"

/* USB CDC */
#include "usbd_core.h"
#include "usbd_desc.h"
#include "usbd_cdc.h"
#include "usbd_cdc_interface.h"

/*==========================================================================
 * 参数
 *==========================================================================*/
#define DEMO_MODE       0       /* 0=MAX9814, 1=音乐演示 */
#define FFT_SIZE        256
#define SAMPLE_RATE     47619

#define MAX9814_DC      2048
#define DISP_BINS       128
#define NBARS           128
#define FALL_RATE       2.0f
#define BAR_COLOR       0x07FF
#define DB_RANGE        60.0f

/*==========================================================================
 * 全局变量
 *==========================================================================*/
uint16_t g_adc_buf[FFT_SIZE];
float    g_fft_in[FFT_SIZE * 2];
float    g_fft_out[FFT_SIZE];
float    g_smooth[NBARS];
volatile uint8_t g_ready = 0;

/* USB */
USBD_HandleTypeDef USBD_Device;
extern volatile uint8_t g_device_state;   /* USB连接状态: 0=未连接, 1=已连接 */

/*==========================================================================
 * 汉宁窗 + 对数分箱
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
        float col = (freq <= f_min) ? 0.0f : (log10f(freq / f_min) / log_ratio) * (float)(NBARS - 1);
        if (col < 0.0f) col = 0.0f;
        if (col >= (float)NBARS) col = (float)(NBARS - 1);
        g_bin_to_bar[i] = (uint8_t)(col + 0.5f);
    }
}

/*==========================================================================
 * ADC + DMA (v4 BSP方案)
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
 * FFT
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
    for (i = 0; i < FFT_SIZE; i++) g_fft_out[i] *= 2.0f;
}

/*==========================================================================
 * LCD 频谱显示 (同v4)
 *==========================================================================*/
static float interpolate_peak(uint16_t bin)
{
    float y0, y1, y2, delta;
    if (bin < 1 || bin >= DISP_BINS - 1) return (float)bin * SAMPLE_RATE / FFT_SIZE;
    y0 = g_fft_out[bin - 1]; y1 = g_fft_out[bin]; y2 = g_fft_out[bin + 1];
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
    char buf[32];
    float col_max[NBARS], max_mag = 0;
    uint16_t peak_bin = 0;

    lcd_clear(BLACK);
    for (i = 0; i < NBARS; i++) col_max[i] = 0.0f;
    for (i = 1; i < FFT_SIZE / 2; i++)
    { uint8_t col = g_bin_to_bar[i]; if (g_fft_out[i] > col_max[col]) col_max[col] = g_fft_out[i]; }
    for (i = 0; i < NBARS; i++) if (col_max[i] > max_mag) max_mag = col_max[i];
    for (i = 1; i < FFT_SIZE / 2; i++) if (g_fft_out[i] >= max_mag) { peak_bin = i; break; }
    if (max_mag < 1.0f) max_mag = 1.0f;

#define BAR_MX_H   184
#define BAR_BASE_Y 212
#define DB_REF     262144.0f
    for (i = 0; i < NBARS; i++)
    {
        float dbfs = 20.0f * log10f(col_max[i] / DB_REF + 0.00001f);
        float db_norm = (dbfs + DB_RANGE) / DB_RANGE;
        if (db_norm < 0.0f) db_norm = 0.0f; if (db_norm > 1.0f) db_norm = 1.0f;
        float target = db_norm * (float)BAR_MX_H;
        if (target >= g_smooth[i]) g_smooth[i] = target;
        else { g_smooth[i] -= FALL_RATE; if (g_smooth[i] < target) g_smooth[i] = target; if (g_smooth[i] < 0.0f) g_smooth[i] = 0.0f; }
    }
    lcd_fill(0, 0, w, 24, 0x0841);
    sprintf(buf, "%d Hz", (int)interpolate_peak(peak_bin));
    lcd_show_string(5, 2, 100, 16, 16, buf, YELLOW);
    sprintf(buf, "%.0f dBFS", 20.0f * log10f(max_mag / DB_REF + 0.00001f));
    lcd_show_string(115, 2, 80, 16, 16, buf, 0x7E8C);
    lcd_draw_line(0, BAR_BASE_Y + 1, w, BAR_BASE_Y + 1, 0x2104);
    lcd_show_string(0, BAR_BASE_Y+3, 30, 12, 12, "0.1k", GRAY);
    lcd_show_string(w*1/6-10, BAR_BASE_Y+3, 30, 12, 12, "0.5k", GRAY);
    lcd_show_string(w*2/6-10, BAR_BASE_Y+3, 30, 12, 12, "1k", GRAY);
    lcd_show_string(w*3/6-10, BAR_BASE_Y+3, 40, 12, 12, "5k", GRAY);
    lcd_show_string(w*4/6-10, BAR_BASE_Y+3, 40, 12, 12, "10k", GRAY);
    lcd_show_string(w*5/6-10, BAR_BASE_Y+3, 40, 12, 12, "20k", GRAY);
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
 * ★ v7: USB CDC 发送 FFT 频谱数据
 *
 * 协议: [0xAA 0x55] [len_lo len_hi] [128×uint16 FFT幅度] [XOR]
 * 每帧261字节, USB CDC全速12Mbps, 几毫秒发完
 *==========================================================================*/
void send_fft_over_usb(void)
{
    uint16_t i;
    uint8_t checksum = 0;
    uint16_t data_len = DISP_BINS * sizeof(uint16_t);  /* 256 */
    static uint16_t out_buf[DISP_BINS];
    uint8_t *p;

    /* FFT幅度 → uint16 */
    for (i = 0; i < DISP_BINS; i++) {
        float v = g_fft_out[i];
        if (v < 0) v = 0;
        if (v > 65535) v = 65535;
        out_buf[i] = (uint16_t)v;
    }
    p = (uint8_t *)out_buf;

    /* 组帧 */
    static uint8_t frame[261];
    frame[0] = 0xAA;
    frame[1] = 0x55;
    frame[2] = (uint8_t)(data_len & 0xFF);
    frame[3] = (uint8_t)(data_len >> 8);
    for (i = 0; i < data_len; i++) {
        frame[4 + i] = p[i];
        checksum ^= p[i];
    }
    frame[4 + data_len] = checksum;

    /* USB CDC 发送 (一次发整帧, 不拆包) */
    cdc_vcp_data_tx(frame, 261);
}

/*==========================================================================
 * ★★★ main v7 ★★★
 *==========================================================================*/
int main(void)
{
    uint16_t i;
    uint8_t  usb_status = 0;

    sys_stm32_clock_init(9);
    delay_init(72);
    usart_init(72, 115200);     /* 保留串口(虽然CH340G有问题, 不影响USB) */
    led_init();

    hanning_init();
    log_binmap_init();
    for (i = 0; i < NBARS; i++) g_smooth[i] = 0.0f;

    lcd_init();
    lcd_display_dir(1);

    /* --- USB CDC 初始化 --- */
    lcd_show_string(30, 20, 280, 16, 16, "Spectrum Analyzer v7", RED);
    lcd_show_string(30, 40, 280, 14, 14, "USB Virtual COM Port", YELLOW);
    lcd_show_string(30, 60, 280, 14, 14, "MAX9814 -> PA1", GREEN);
    lcd_show_string(30, 90, 280, 14, 14, "USB: Connecting...", GREEN);

    usbd_port_config(0);        /* USB 先断开 */
    delay_ms(500);
    usbd_port_config(1);        /* USB 再连接 */
    delay_ms(500);

    USBD_Init(&USBD_Device, &VCP_Desc, 0);
    USBD_RegisterClass(&USBD_Device, USBD_CDC_CLASS);
    USBD_CDC_RegisterInterface(&USBD_Device, &USBD_CDC_fops);
    USBD_Start(&USBD_Device);

    /* 等待 USB 连接 (最多等5秒) */
    for (i = 0; i < 500; i++) {
        if (g_device_state == 1) break;
        delay_ms(10);
    }

    if (g_device_state == 1) {
        lcd_show_string(30, 90, 280, 14, 14, "USB: CONNECTED!       ", GREEN);
        LED1(0);
    } else {
        lcd_show_string(30, 90, 280, 14, 14, "USB: no cable?        ", RED);
    }
    delay_ms(1000);

    /* --- ADC+DMA 启动 --- */
    adc_dma_spectrum_init();

    /* --- 主循环 --- */
    while (1)
    {
        /* USB 状态变化 */
        if (usb_status != g_device_state) {
            usb_status = g_device_state;
            if (usb_status == 1) {
                lcd_show_string(30, 90, 280, 14, 14, "USB: CONNECTED!       ", GREEN);
                LED1(0);
            } else {
                lcd_show_string(30, 90, 280, 14, 14, "USB: disconnected     ", RED);
                LED1(1);
            }
        }

        if (g_ready)
        {
            g_ready = 0;
            do_fft();

            /* USB已连接 → 发数据, 否则跳过 */
            if (g_device_state == 1) {
                send_fft_over_usb();
            }

            draw_spectrum();
            adc_dma_enable(FFT_SIZE);
            LED0_TOGGLE();
        }
        delay_ms(1);
    }
}
