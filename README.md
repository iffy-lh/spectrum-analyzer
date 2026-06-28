# 频谱分析仪 / 噪音识别分析系统

基于 STM32F103ZET6 精英版 + MAX9814 麦克风模块的实时音频频谱分析项目。

> ⚠️ 本仓库包含**两个独立部分**，请根据需要查看对应目录。

---

## 项目结构

```
spectrum-analyzer/
├── firmware/          ← 🔌 STM32 嵌入式固件 (C语言, Keil MDK)
├── pc_analyzer/       ← 💻 PC 端 Python 分析系统
└── README.md
```

---

## 🔌 第一部分：STM32 嵌入式固件 (`firmware/`)

运行在正点原子 F1 精英版开发板上的实时频谱采集程序。

### 硬件配置

| 组件 | 型号/连接 |
|------|----------|
| 主控 | STM32F103ZET6 (精英版) |
| 麦克风 | MAX9814 (GAIN=VDD=50dB, AR悬空, OUT→PA1) |
| 通信 | USB CDC 虚拟串口 (USB_SLAVE口, COM12, 115200bps) |
| 显示 | 3.2寸 TFTLCD 横屏 320×240 |

### 技术特性

- **ADC+DMA 采集**：TIM2 触发，采样率 ~47.6kHz，DMA1_CH1 搬运
- **FFT 频谱分析**：CMSIS-DSP 库 arm_cfft_f32，256 点 FFT，频率分辨率 186Hz/bin
- **显示效果**：对数频率轴 + 快升慢降包络，亮蓝单色密柱
- **模式切换**：`DEMO_MODE` 宏可在测试信号/真实 ADC 采集间切换
- **版本演进**：v1 基础框架 → v4 MAX9814 真实 ADC + USB CDC

### 固件文件

```
firmware/
├── User/main.c          # 主程序 (频谱采集 + FFT + 显示)
├── Drivers/             # BSP 驱动 (LCD, LED, ADC, DMA, USB CDC)
├── Middlewares/         # CMSIS-DSP 库
└── Projects/MDK-ARM/    # Keil MDK 工程文件
```

### 编译烧录

1. 用 Keil MDK (C:\D\Keil5\) 打开 `Projects/MDK-ARM/atk_f103.uvprojx`
2. 编译器选 AC6，ST-Link SWD 烧录
3. 或用命令行：`powershell.exe -Command "Start-Process 'C:\D\Keil5\UV4\UV4.exe' -ArgumentList '-f path\to\atk_f103.uvprojx -o log.txt'"`

---

## 💻 第二部分：PC 端 Python 分析系统 (`pc_analyzer/`)

接收 STM32 串口数据，进行实时频谱显示、噪音分类和烦恼度评估。

### 功能模块

| 模块 | 文件 | 功能 |
|------|------|------|
| 主入口 | `main.py` | 支持 WAV离线 / 串口实时 / Demo模拟 三种模式 |
| 实时频谱 | `terminal_analyzer.py` | 终端 ASCII 频谱显示（无需 matplotlib，可直接在 VS Code 终端运行） |
| 可视化 | `live_analyzer.py` | matplotlib 实时频谱图 + 噪音分类 |
| 串口接收 | `serial_receiver.py` | 从 STM32 接收 ADC 数据（协议：0xAA 0x55 帧头 + XOR 校验） |
| 特征提取 | `audio_features.py` | FFT / MFCC / 谐波检测 / 时域特征 |
| 噪音分类 | `noise_classifier.py` | 两层分类：规则匹配 Layer1 + ML 分类器 Layer2 (RandomForest) |
| 烦恼度 | `annoyance_evaluator.py` | A计权声压级 + 类型权重 + 时间/时长情境调整 |
| 音库 | `sound_library.py` | 噪音类型查询 + 频段能量匹配 |

### 安装运行

```bash
cd pc_analyzer/
pip install -r requirements.txt

# 终端版（推荐，VS Code / PowerShell 直接跑）
python terminal_analyzer.py COM12

# 实时频谱图版
python live_analyzer.py COM12

# 离线 WAV 分析
python main.py --input test.wav
```

### 依赖

```
numpy>=1.21.0
pyserial>=3.5
scikit-learn>=1.0.0   # ML分类器（可选）
```

---

## 📋 版本历史

| 版本 | 日期 | 内容 |
|------|------|------|
| v1-v3 | 2025 | 基础框架：TFTLCD 显示、DMA+ADC、FFT |
| v4 | 2025-06 | MAX9814 真实 ADC 模式、USB CDC 虚拟串口 |
| v5 | 2025-06 | ADC+DMA 初始化修复（硬件复位 + 校准顺序） |
| v6 | 2025-06 | 串口协议完善、PC 端 ASCII 频谱显示 |
| v7 | 2025-06 | PC 端完整分析系统：噪音分类 + 烦恼度评估 + 多模式支持 |

---

## 🔗 相关链接

- GitHub: https://github.com/iffy-lh/spectrum-analyzer
- 开发板：正点原子 F1 精英版 (STM32F103ZET6)
- 工具链：Keil MDK AC6 + ST-Link SWD
