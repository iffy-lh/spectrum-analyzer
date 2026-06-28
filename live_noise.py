"""实时噪音分析 — USB CDC 版"""
import serial, time, struct, os, sys
from collections import deque, Counter
import numpy as np

if len(sys.argv) < 2:
    print("用法: python live_noise.py COM端口")
    sys.exit(1)

port = sys.argv[1]
ser = serial.Serial(port, 115200, timeout=0.1)
buf = bytearray()
spec = np.zeros(128)
frames = errors = 0
history = deque(maxlen=5)   # 快速响应, 2秒内切换

SAMPLE_RATE = 47619
F_MIN = SAMPLE_RATE / 256
F_MAX = SAMPLE_RATE / 2
log_r = np.log10(F_MAX / F_MIN)
FREQS = np.array([F_MIN * 10**(log_r * i / 127) for i in range(128)])

LABELS = {"quiet":"安静","speech":"说话声","traffic":"交通","hvac":"空调","shrill":"啸叫","construction":"施工","crowd":"嘈杂","mixed":"混合"}
COLORS = {"quiet":"\033[92m","speech":"\033[93m","traffic":"\033[91m","hvac":"\033[94m","shrill":"\033[95m","mixed":"\033[97m"}

def classify(s):
    t = np.sum(s) + 1e-6
    if t < 100: return "quiet"
    # 修正频段: 186Hz/bin, 256pt FFT
    # bass: bins 0-2  (0-558Hz)   - 交通/空调
    # mid:  bins 3-22 (558-4100Hz) - 说话声共振峰
    # high: bins 23+  (4100Hz+)    - 尖锐啸叫
    bass = np.sum(s[:3]) / t
    mid  = np.sum(s[3:23]) / t
    high = np.sum(s[23:]) / t
    if high > 0.35: return "shrill"
    if bass > 0.65 and mid < 0.2 and high < 0.1: return "hvac"
    if bass > 0.45 and mid < 0.25: return "traffic"
    if mid > 0.25: return "speech"
    if bass > 0.2 and mid > 0.2 and high > 0.15: return "construction"
    return "mixed"

def bar(s):
    m = np.max(s) or 1
    chars = " ▁▂▃▄▅▆▇█"
    line = ""
    for i in range(0, 128, 2):
        v = np.mean(s[i:i+2])
        h = min(8, int(v/m*8))
        if h==0 and v>0: h=1
        line += chars[h]
    return line

print("=" * 72)
print(f"  实时噪音分析 — {port} (STM32 USB CDC)")
print("  对着麦克风说话/拍手/敲桌子试试")
print("=" * 72)

try:
    while True:
        if ser.in_waiting:
            buf.extend(ser.read(ser.in_waiting))
            while True:
                idx = buf.find(b'\xaa\x55')
                if idx < 0: break
                if idx > 0: buf = buf[idx:]
                if len(buf) < 5: break
                dlen = buf[2] | (buf[3] << 8)
                total = 5 + dlen
                if len(buf) < total: break
                csum = 0
                for b in buf[4:4+dlen]: csum ^= b
                if csum == buf[4+dlen]:
                    for i in range(128):
                        lo = buf[4+i*2]; hi = buf[4+i*2+1]
                        spec[i] = float(lo | (hi << 8))
                    frames += 1
                else:
                    errors += 1
                buf = buf[total:]

        if frames > 0:
            history.append(classify(spec))
            noise_type = Counter(history).most_common(1)[0][0]
            peak_bin = np.argmax(spec)
            peak_freq = FREQS[peak_bin]
            t = np.sum(spec) + 1e-6
            bass = int(np.sum(spec[:3])/t*100)
            mid  = int(np.sum(spec[3:23])/t*100)
            high = int(np.sum(spec[23:])/t*100)
            color = COLORS.get(noise_type, "")
            b = bar(spec)

            os.system('cls')
            print("=" * 72)
            print(f"  [?25h 噪音]: {color}{LABELS.get(noise_type, noise_type)}\033[0m    峰值: {peak_freq:.0f} Hz")
            print(f"  [?25h 频段]: 低{bass}% | 中{mid}% | 高{high}%    帧:{frames} 错:{errors}")
            print("=" * 72)
            print(f"  {b}")
            print(f"  156Hz{' '*17}500Hz{' '*13}2kHz{' '*11}5kHz{' '*9}12kHz  23.8kHz")
            if noise_type == "quiet": print("\n  环境安静")
            elif noise_type == "traffic": print("\n  检测到低频交通噪音")
            elif noise_type == "shrill": print("\n  ⚠ 尖锐高频噪音!")
        time.sleep(0.05)

except KeyboardInterrupt:
    print(f"\n\n停止. 接收 {frames} 帧, {errors} 错误")
finally:
    ser.close()
