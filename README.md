# 正点原子F1精英版 频谱分析仪
# STM32F103ZET6 Spectrum Analyzer with CMSIS-DSP

## 功能
- CMSIS-DSP FFT (arm_cfft_radix4_f32) 频谱分析
- 定时器2触发ADC1 + DMA1 循环采样 (40kHz)
- 2.8寸 TFT LCD 显示 (128柱 + dB刻度 + 频率标尺)
- 汉宁窗 + 抛物线插值 + 峰值保持
- 演示模式：内部合成音乐信号
- 真实模式：PA1 接麦克风采集音频

## 硬件
- 正点原子 F1 精英版 (STM32F103ZET6)
- 2.8寸 TFT LCD (ILI9341)
- 信号输入: PA1 (ADC1_CH1)

## 编译
- Keil MDK5 + ARM Compiler 6
- 工程文件: Projects/MDK-ARM/atk_f103.uvprojx

## 模式切换
main.c 顶部:
```c
#define DEMO_MODE  1  // 演示模式
#define DEMO_MODE  0  // 真实ADC模式
```
