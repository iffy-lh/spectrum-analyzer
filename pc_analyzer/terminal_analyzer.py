"""
噪音频谱实时分析 — 终端版 (无需 matplotlib)

在 VS Code 终端或 PowerShell 中运行:
    python terminal_analyzer.py COM11
"""

import sys
import time
import threading
from collections import deque
import os
import numpy as np

try:
    import serial
except ImportError:
    print("需要: pip install pyserial")
    sys.exit(1)

# ============================================================
# 参数
# ============================================================
NBARS = 128
SAMPLE_RATE = 47619
FREQ_MIN = SAMPLE_RATE / 256
FREQ_MAX = SAMPLE_RATE / 2

def log_axis():
    log_ratio = np.log10(FREQ_MAX / FREQ_MIN)
    freqs = []
    for i in range(NBARS):
        f = FREQ_MIN * 10 ** (log_ratio * i / (NBARS - 1)) if i > 0 else FREQ_MIN
        freqs.append(f)
    return np.array(freqs)

FREQ_AXIS = log_axis()

# ============================================================
# 串口
# ============================================================
class Receiver:
    def __init__(self, port):
        self.ser = serial.Serial(port, 500000, timeout=0.1)
        self.buf = bytearray()
        self.spectrum = np.zeros(NBARS)
        self.lock = threading.Lock()
        self.frames = 0
        self.errors = 0
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            try:
                if self.ser.in_waiting:
                    self.buf.extend(self.ser.read(self.ser.in_waiting))
                    self._parse()
                else:
                    time.sleep(0.001)
            except:
                break

    def _parse(self):
        while True:
            idx = self.buf.find(b'\xaa\x55')
            if idx < 0:
                self.buf = self.buf[-1:] if len(self.buf) > 1 else self.buf
                return
            if idx > 0:
                self.buf = self.buf[idx:]
            if len(self.buf) < 5:
                return
            dlen = self.buf[2] | (self.buf[3] << 8)
            total = 5 + dlen
            if len(self.buf) < total:
                return
            csum = 0
            for b in self.buf[4:4+dlen]:
                csum ^= b
            if csum == self.buf[4+dlen]:
                spec = np.zeros(NBARS, dtype=np.float32)
                for i in range(NBARS):
                    spec[i] = float(self.buf[4+i*2] | (self.buf[4+i*2+1] << 8))
                with self.lock:
                    self.spectrum = spec
                    self.frames += 1
            else:
                self.errors += 1
            self.buf = self.buf[total:]

    def get(self):
        with self.lock:
            return self.spectrum.copy(), self.frames, self.errors

    def stop(self):
        self.running = False
        self.ser.close()


def classify(spec, history):
    """简单规则分类"""
    total = np.sum(spec) + 1e-6
    if total < 1e-3:
        return "quiet"

    low  = np.sum(spec[:25])  / total
    mid  = np.sum(spec[25:60]) / total
    high = np.sum(spec[60:])  / total

    if np.sqrt(np.mean(spec**2)) < 50:
        return "quiet"
    if high > 0.45:
        return "shrill"
    if low > 0.55 and high < 0.15:
        return "traffic"
    if low > 0.65 and high < 0.08:
        return "hvac"
    if mid > 0.35:
        return "speech"
    return "mixed"


LABELS = {
    "quiet":   "安静环境    ",
    "speech":  "说话声      ",
    "traffic": "交通噪音    ",
    "hvac":    "空调/风扇   ",
    "shrill":  "尖锐啸叫    ",
    "mixed":   "混合噪音    ",
}

COLORS = {
    "quiet":   "\033[92m",  # green
    "speech":  "\033[93m",  # yellow
    "traffic": "\033[91m",  # red
    "hvac":    "\033[94m",  # blue
    "shrill":  "\033[95m",  # magenta
    "mixed":   "\033[97m",  # white
}
RESET = "\033[0m"


def draw_ascii_spectrum(spec):
    """ASCII 频谱图"""
    max_val = np.max(spec)
    if max_val < 1:
        max_val = 1

    WIDTH = 100
    # 只取前 64 个 bin (0-~5kHz, 最重要)
    display = spec[:64]
    step = max(1, len(display) // WIDTH)

    bars = ""
    for i in range(0, len(display), step):
        val = display[i:i+step].max()
        h = int(val / max_val * 15)
        if h == 0 and val > 0:
            h = 1
        chars = " ▁▂▃▄▅▆▇█"
        bars += chars[min(h, 8)]

    return bars


def main():
    if len(sys.argv) < 2:
        print("用法: python terminal_analyzer.py COM端口")
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        print(f"可用串口: {[p.device for p in ports]}")
        sys.exit(1)

    print("=" * 70)
    print("  [?25h 噪音频谱实时分析 — 终端版")
    print("=" * 70)
    print(f"  串口: {sys.argv[1]} @ 500000 bps")
    print(f"  等待数据... (Ctrl+C 停止)")
    print()

    rx = Receiver(sys.argv[1])
    history = np.zeros(NBARS)
    noise_history = deque(maxlen=10)

    try:
        time.sleep(1)  # 等第一帧

        while True:
            spec, frames, errors = rx.get()

            if frames > 0:
                history = history * 0.75 + spec * 0.25

                # 分类
                noise_type = classify(spec, history)
                noise_history.append(noise_type)

                # 稳定分类 (最近10帧众数)
                from collections import Counter
                stable = Counter(noise_history).most_common(1)[0][0]

                # 峰值频率
                peak_bin = np.argmax(history)
                peak_freq = FREQ_AXIS[peak_bin]

                # ASCII 频谱
                bar = draw_ascii_spectrum(history)

                # 频段能量
                total = np.sum(history) + 1e-6
                low_pct  = int(np.sum(history[:25])  / total * 100)
                mid_pct  = int(np.sum(history[25:60]) / total * 100)
                high_pct = int(np.sum(history[60:])  / total * 100)

                color = COLORS.get(stable, "")

                # 清屏 + 输出
                os.system('cls' if sys.platform == 'win32' else 'clear')
                print("=" * 70)
                print(f"  [?25h 噪音类型]: {color}{LABELS.get(stable, stable)}{RESET}   峰值: {peak_freq:.0f} Hz")
                print(f"  [?25h 频段能量]: 低频 {low_pct:3d}% | 中频 {mid_pct:3d}% | 高频 {high_pct:3d}%")
                print(f"  [?25h 帧数]: {frames}  |  错误: {errors}")
                print("=" * 70)
                print(f"  [?25h 频谱]: {bar}")
                print(f"  [?25h 频率轴]: 156Hz{'-'*30}500Hz{'-'*20}2kHz{'-'*15}5kHz{'-'*10}12kHz  23.8kHz")

                if stable == "quiet":
                    print("\n  [?25h 听]: 当前环境安静")
                elif stable == "speech":
                    print("\n  [?25h 听]: 检测到说话声 -- 可能干扰学习")
                elif stable == "traffic":
                    print("\n  [?25h 听]: 交通噪音 -- 低频隆隆声")
                elif stable == "shrill":
                    print("\n  [?25h 听]: 尖锐噪音 -- 可能损伤听力!")

                print("\n  (Ctrl+C 停止)")

            time.sleep(0.15)

    except KeyboardInterrupt:
        print("\n\n[*] 已停止")
        print(f"[*] 共接收 {frames} 帧, {errors} 错误")
    finally:
        rx.stop()


if __name__ == "__main__":
    main()
