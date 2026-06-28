"""
噪音频谱实时分析 — STM32上位机

接收 STM32 通过串口发来的 FFT 频谱数据(128个幅度值),
实时显示频谱图并做噪音分类。

用法 (在 VS Code 终端或 PowerShell 中):
    python live_analyzer.py COM11

需要安装:
    pip install pyserial numpy matplotlib
"""

import sys
import struct
import time
import threading
from collections import deque

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # VS Code 兼容, 也可以改成 Qt5Agg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

try:
    import serial
except ImportError:
    print("需要 pyserial: pip install pyserial")
    sys.exit(1)

# ============================================================
# 频谱参数 (与 STM32 保持一致)
# ============================================================
FFT_SIZE   = 256
NBARS      = 128
SAMPLE_RATE = 47619  # STM32 采样率
FREQ_MIN   = SAMPLE_RATE / FFT_SIZE   # 156 Hz (bin 1)
FREQ_MAX   = SAMPLE_RATE / 2          # 23.8 kHz (Nyquist)

# 对数频率轴 (和 STM32 的 log_binmap 对应)
def log_binmap():
    log_ratio = np.log10(FREQ_MAX / FREQ_MIN)
    freqs = []
    for i in range(NBARS):
        if i == 0:
            f = FREQ_MIN
        else:
            f = FREQ_MIN * 10 ** (log_ratio * i / (NBARS - 1))
        freqs.append(f)
    return np.array(freqs)

FREQ_AXIS = log_binmap()  # 128个柱对应的实际频率(Hz)

# ============================================================
# 串口接收
# ============================================================
class SpectrumReceiver:
    """接收 STM32 的 FFT 频谱数据"""

    def __init__(self, port, baudrate=500000):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.buf = bytearray()
        self.latest_spectrum = np.zeros(NBARS)
        self.lock = threading.Lock()
        self.running = False
        self.frame_count = 0
        self.error_count = 0

    def connect(self):
        self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
        print(f"[*] 串口已连接: {self.port} @ {self.baudrate} bps")
        return True

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print("[*] 接收线程已启动")

    def _run(self):
        while self.running:
            try:
                if self.ser.in_waiting:
                    self.buf.extend(self.ser.read(self.ser.in_waiting))
                    self._parse()
                else:
                    time.sleep(0.001)
            except Exception as e:
                print(f"[!] 串口错误: {e}")
                break

    def _parse(self):
        while True:
            idx = self.buf.find(b'\xaa\x55')
            if idx < 0:
                if len(self.buf) > 1:
                    self.buf = self.buf[-1:]  # keep last byte (might be 0xAA)
                return

            if idx > 0:
                self.buf = self.buf[idx:]

            if len(self.buf) < 5:
                return

            data_len = self.buf[2] | (self.buf[3] << 8)
            frame_total = 5 + data_len

            if len(self.buf) < frame_total:
                return

            # 校验
            csum = 0
            for b in self.buf[4:4 + data_len]:
                csum ^= b

            if csum == self.buf[4 + data_len]:
                # 解析 128 个 uint16
                spectrum = np.zeros(NBARS, dtype=np.float32)
                for i in range(NBARS):
                    lo = self.buf[4 + i * 2]
                    hi = self.buf[4 + i * 2 + 1]
                    spectrum[i] = float(lo | (hi << 8))

                with self.lock:
                    self.latest_spectrum = spectrum
                    self.frame_count += 1
            else:
                self.error_count += 1

            self.buf = self.buf[frame_total:]

    def get_spectrum(self):
        with self.lock:
            return self.latest_spectrum.copy()

    def stop(self):
        self.running = False
        if self.ser:
            self.ser.close()


# ============================================================
# 噪音分类 (简化版, 基于频谱特征规则)
# ============================================================
NOISE_TYPES = {
    "quiet":    "安静",
    "speech":   "说话声",
    "traffic":  "交通噪音",
    "hvac":     "空调/风扇",
    "shrill":   "尖锐啸叫",
    "music":    "音乐",
}

def classify_spectrum(spectrum):
    """
    基于频谱特征的简单规则分类

    spectrum: 128个频率bin的幅度值
    """
    total = np.sum(spectrum) + 1e-6

    # 频段能量
    # FREQ_AXIS: 156Hz ~ 23810Hz 对数分布
    # 低频(0~40柱): ~156~1kHz
    # 中频(40~80): ~1k~5kHz
    # 高频(80~127): ~5k~23.8kHz

    low_band  = np.sum(spectrum[:25])  / total   # ~156-500Hz
    mid_band  = np.sum(spectrum[25:60]) / total   # ~500-3kHz
    high_band = np.sum(spectrum[60:])  / total    # ~3k-23.8kHz

    # 能量水平
    rms = np.sqrt(np.mean(spectrum ** 2))
    peak_val = np.max(spectrum)
    peak_bin = np.argmax(spectrum)

    # 分类逻辑
    if rms < 50:
        return "quiet", 0.90

    if high_band > 0.5 and peak_bin > 80:
        return "shrill", min(0.95, high_band)

    if low_band > 0.55 and high_band < 0.15:
        return "traffic", min(0.85, low_band)

    if low_band > 0.6 and rms < 300 and high_band < 0.10:
        return "hvac", min(0.80, low_band)

    if mid_band > 0.35 and 0.15 < high_band < 0.35:
        return "speech", min(0.75, mid_band * 1.5)

    if 0.25 < mid_band < 0.50 and low_band > 0.2:
        return "music", 0.60

    # 最佳匹配
    if low_band > mid_band and low_band > high_band:
        return "traffic", 0.50
    elif mid_band > high_band:
        return "speech", 0.40
    else:
        return "shrill", 0.40


# ============================================================
# 实时可视化
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("用法: python live_analyzer.py COM端口")
        print("例如: python live_analyzer.py COM11")
        # 列出可用串口
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        print(f"\n可用串口: {[p.device for p in ports]}")
        sys.exit(1)

    port = sys.argv[1]
    rx = SpectrumReceiver(port)
    rx.connect()
    rx.start()

    print("\n等待数据... (关闭图表窗口停止)")
    print(f"如果无数据显示, 检查:")
    print(f"  1. STM32 USB_232 插好了吗?")
    print(f"  2. LCD 上频谱柱子在跳动吗?")
    print(f"  3. 串口号是 {port} 吗?\n")

    # 历史数据 (用于平滑)
    history = np.zeros(NBARS)

    # 创建图表
    plt.ion()
    fig, (ax_spec, ax_info) = plt.subplots(2, 1, figsize=(12, 7),
        gridspec_kw={'height_ratios': [3, 1]})
    fig.canvas.manager.set_window_title('噪音频谱实时分析')

    # 频谱图
    bars = ax_spec.bar(range(NBARS), np.zeros(NBARS), width=1.0,
                       color='#00d4ff', edgecolor='none')
    ax_spec.set_xlim(0, NBARS)
    ax_spec.set_ylim(0, 5000)
    ax_spec.set_ylabel('幅度', fontsize=10)
    ax_spec.set_title('实时频谱 (STM32 FFT)', fontsize=12, fontweight='bold')
    ax_spec.grid(axis='y', alpha=0.2)

    # 频率刻度
    tick_positions = [0, 25, 50, 75, 100, 127]
    tick_labels = ['156Hz', '500Hz', '2kHz', '5kHz', '12kHz', '23.8kHz']
    ax_spec.set_xticks(tick_positions)
    ax_spec.set_xticklabels(tick_labels, fontsize=8)

    # 信息区
    ax_info.axis('off')
    info_text = ax_info.text(0.02, 0.5, '', fontsize=13, fontfamily='monospace',
                              va='center', transform=ax_info.transAxes)

    frame_count = 0
    last_update = time.time()

    while plt.fignum_exists(fig.number):
        spectrum = rx.get_spectrum()

        if np.sum(spectrum) > 0:
            # 平滑
            history = history * 0.7 + spectrum * 0.3

            # 更新柱状图
            for bar, h in zip(bars, history):
                bar.set_height(h)

            # 自动调整 Y 轴
            max_h = np.max(history)
            if max_h > 0:
                ax_spec.set_ylim(0, max(100, max_h * 1.2))

            # 分类
            noise_id, confidence = classify_spectrum(history)
            noise_name = NOISE_TYPES.get(noise_id, noise_id)

            # 峰值频率
            peak_bin = np.argmax(history)
            peak_freq = FREQ_AXIS[peak_bin]

            # 信息文本
            info = (
                f"  [?25h 噪音类型]: {noise_name}  (置信度 {confidence:.0%})\n"
                f"  [?25h 峰值频率]: {peak_freq:.0f} Hz\n"
                f"  [?25h 频谱能量]: 低 {np.sum(history[:25])/np.sum(history)*100:.0f}%  "
                f"中 {np.sum(history[25:60])/np.sum(history)*100:.0f}%  "
                f"高 {np.sum(history[60:])/np.sum(history)*100:.0f}%\n"
                f"  [?25h 接收帧数]: {rx.frame_count}  |  错误: {rx.error_count}"
            )
            info_text.set_text(info)

            # 根据噪音类型变色
            colors = {"quiet": "#00ff88", "speech": "#ffaa00", "traffic": "#ff4444",
                      "hvac": "#44aaff", "shrill": "#ff00ff", "music": "#aa88ff"}
            color = colors.get(noise_id, "#ffffff")
            ax_spec.set_title(f'噪音类型: {noise_name}', fontsize=12,
                              fontweight='bold', color=color)

            frame_count += 1

        # 刷新
        fig.canvas.flush_events()
        time.sleep(0.03)  # ~30fps

    rx.stop()
    print("\n[*] 已停止")


if __name__ == "__main__":
    main()
