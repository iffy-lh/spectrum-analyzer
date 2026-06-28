"""
噪音识别演示 — 纯软件版 (无需 STM32 硬件)

在 VS Code 终端运行:  python demo_analyzer.py

模拟不同噪音类型的频谱 → 分类引擎识别 → 终端显示结果
"""

import sys, os, time, math
from collections import deque
import numpy as np

# ============================================================
# 简易音库 & 分类逻辑 (内嵌, 不依赖外部文件)
# ============================================================

FREQ_BANDS = {
    "sub_bass":  (0, 60),
    "bass":      (60, 250),
    "low_mid":   (250, 500),
    "mid":       (500, 2000),
    "high_mid":  (2000, 4000),
    "high":      (4000, 20000),
}

# 模拟噪音频谱生成器 — 返回 128 个 bin 的幅度值
def gen_spectrum(noise_type):
    """模拟不同类型噪音的 FFT 频谱 (128 bins)"""
    bins = 128
    spec = np.zeros(bins)

    # 频率轴 (对数分布, 和 STM32 FFT 一致)
    f_min = 156.0
    f_max = 20000.0
    log_r = math.log10(f_max / f_min)
    freqs = np.array([f_min * 10 ** (log_r * i / (bins - 1)) for i in range(bins)])

    if noise_type == "traffic":
        # 低频轰鸣: 50-500Hz 强, 噪声基底
        for i, f in enumerate(freqs):
            if f < 500:
                spec[i] = np.random.uniform(300, 800) * (1 - f/600)
            elif f < 3000:
                spec[i] = np.random.uniform(20, 80)
            else:
                spec[i] = np.random.uniform(0, 15)

    elif noise_type == "hvac":
        # 空调/风扇: 极低频, 60Hz 基频 + 谐波, 很平坦
        for i, f in enumerate(freqs):
            if f < 200:
                spec[i] = np.random.uniform(200, 500)
            elif f < 120:
                spec[i] = 400 + 200 * math.sin(f/60 * 2*math.pi) * 0.3
            else:
                spec[i] = np.random.uniform(0, 10)

    elif noise_type == "shrill":
        # 尖锐啸叫: 2-8kHz 强峰
        for i, f in enumerate(freqs):
            if 2000 < f < 8000:
                # 高斯峰在 4kHz 附近
                peak = 800 * math.exp(-((f - 4000) / 1000) ** 2)
                spec[i] = peak + np.random.uniform(0, 30)
            elif f < 2000:
                spec[i] = np.random.uniform(0, 30)
            else:
                spec[i] = np.random.uniform(0, 20)

    elif noise_type == "speech":
        # 说话声: 85-255Hz 基频 + 谐波 + 共振峰
        for i, f in enumerate(freqs):
            base = 0
            # 基频 150Hz + 谐波
            for h in range(1, 6):
                base += 200 / h * math.exp(-((f - 150*h) / 30) ** 2)
            # 第一共振峰 ~500Hz
            base += 300 * math.exp(-((f - 500) / 150) ** 2)
            # 第二共振峰 ~1500Hz
            base += 200 * math.exp(-((f - 1500) / 200) ** 2)
            spec[i] = base + np.random.uniform(0, 15)

    elif noise_type == "construction":
        # 施工: 宽带冲击
        for i, f in enumerate(freqs):
            if f < 8000:
                spec[i] = np.random.uniform(100, 600) * (1 - f/9000)
            else:
                spec[i] = np.random.uniform(0, 20)
            # 随机冲击峰
            if np.random.random() < 0.05:
                spec[i] += np.random.uniform(400, 1000)

    elif noise_type == "electrical":
        # 工频噪声: 50Hz 基频 + 奇次谐波 (50, 150, 250, 350...)
        for i, f in enumerate(freqs):
            base = 0
            for h in [50, 150, 250, 350, 450]:
                base += 500 / ((h/50)**0.7) * math.exp(-((f - h) / 15) ** 2)
            spec[i] = base + np.random.uniform(0, 8)

    elif noise_type == "quiet":
        # 安静: 很低的白噪声本底
        spec = np.random.uniform(0, 5, bins)

    elif noise_type == "crowd":
        # 人群嘈杂: 中频能量, 无谐波结构
        for i, f in enumerate(freqs):
            if 200 < f < 3000:
                spec[i] = np.random.uniform(150, 500)
            else:
                spec[i] = np.random.uniform(10, 80)

    elif noise_type == "impact":
        # 撞击: 全频带瞬间能量, 高频也强
        for i, f in enumerate(freqs):
            spec[i] = np.random.uniform(200, 800) * math.exp(-(f / 4000))

    return spec, freqs


def classify(spec):
    """基于频段能量的规则分类, 返回 (类型名, 中文名)"""
    total = np.sum(spec) + 1e-6
    low  = np.sum(spec[:25])  / total   # ~156-500Hz
    mid  = np.sum(spec[25:60]) / total   # ~500-3kHz
    high = np.sum(spec[60:])  / total   # ~3k-20kHz
    rms  = np.sqrt(np.mean(spec**2))

    # 能量太低 → 安静
    if rms < 20:
        return "quiet", "安静环境", 0.95

    # 高频主导 → 尖锐啸叫
    if high > 0.4:
        return "shrill", "尖锐啸叫", min(0.95, high * 1.2)

    # 低频极端主导 → 空调/风扇
    if low > 0.6 and high < 0.08 and rms < 300:
        return "hvac", "空调/风扇嗡鸣", min(0.9, low * 1.2)

    # 低频主导 + 没有太高频 → 交通噪音
    if low > 0.45 and high < 0.12:
        return "traffic", "交通噪音", min(0.85, low)

    # 低频 + 中频都有 → 施工
    if low > 0.2 and mid > 0.25 and high > 0.15:
        return "construction", "施工/建筑噪音", 0.70

    # 中频主导 → 说话声
    if mid > 0.35 and 0.1 < high < 0.3:
        return "speech", "说话声", min(0.80, mid * 1.4)

    # 中频宽 → 人群嘈杂
    if mid > 0.3:
        return "crowd", "人群嘈杂", 0.65

    # 低频 + 奇次谐波 → 工频噪声
    if low > 0.4 and rms > 50:
        return "electrical", "工频噪声", 0.60

    # 默认
    best = max([("low", low), ("mid", mid), ("high", high)], key=lambda x: x[1])
    if best[0] == "low":
        return "traffic", "交通噪音", 0.45
    elif best[0] == "high":
        return "shrill", "尖锐啸叫", 0.45
    else:
        return "speech", "说话声", 0.40


TYPE_LABELS = {
    "quiet":        "安静环境        ",
    "speech":       "说话声          ",
    "traffic":      "交通噪音        ",
    "hvac":         "空调/风扇嗡鸣   ",
    "shrill":       "尖锐啸叫/警报   ",
    "construction": "施工/建筑噪音    ",
    "electrical":   "工频噪声        ",
    "crowd":        "人群嘈杂        ",
    "impact":       "撞击/关门       ",
}


def draw_bar(spec):
    """画 ASCII 频谱柱"""
    max_v = np.max(spec)
    if max_v < 1: max_v = 1
    bars = " ▁▂▃▄▅▆▇█"
    line = ""
    step = max(1, len(spec) // 64)
    for i in range(0, min(64, len(spec)), 1):
        # 取 128 bins 里的前 64 个, 每组 2 个取平均
        idx = i * 2
        v = np.mean(spec[idx:idx+2]) if idx+2 <= len(spec) else spec[idx]
        h = int(v / max_v * 8)
        if h == 0 and v > 0: h = 1
        line += bars[min(h, 8)]
    return line


# ============================================================
# 主演示
# ============================================================

def main():
    noise_types = [
        ("traffic",      "交通噪音 — 低频轰鸣"),
        ("hvac",         "空调嗡鸣 — 60Hz基频"),
        ("shrill",       "尖锐啸叫 — 4kHz峰值"),
        ("speech",       "说话声 — 谐波+共振峰"),
        ("construction", "施工噪音 — 宽带冲击"),
        ("electrical",   "工频噪声 — 50Hz谐波"),
        ("quiet",        "安静环境 — 低本底"),
        ("crowd",        "人群嘈杂 — 中频混乱"),
        ("impact",       "撞击瞬态 — 全频爆发"),
    ]

    print("=" * 72)
    print("  [?25h 噪音识别分析系统 — 纯软件演示版")
    print("  [?25h 模拟频谱 → 特征提取 → 规则分类 → 结果展示")
    print("=" * 72)
    print()

    for noise_id, desc in noise_types:
        spec, freqs = gen_spectrum(noise_id)
        pred_id, pred_name, conf = classify(spec)

        match = "✓" if pred_id == noise_id else "≈"
        bar = draw_bar(spec)

        # 频段能量
        total = np.sum(spec) + 1e-6
        low_pct  = int(np.sum(spec[:25])  / total * 100)
        mid_pct  = int(np.sum(spec[25:60]) / total * 100)
        high_pct = int(np.sum(spec[60:])  / total * 100)

        # 峰值频率
        peak_bin = np.argmax(spec)
        peak_freq = freqs[peak_bin]

        print(f"  [{match}] 演示: {desc}")
        print(f"      检测: {pred_name}  置信度: {conf:.0%}  峰值: {peak_freq:.0f} Hz")
        print(f"      频段: 低频 {low_pct}% | 中频 {mid_pct}% | 高频 {high_pct}%")
        print(f"      频谱: {bar}")
        print()

    print("=" * 72)
    print("  运行成功! 以上展示了 9 种噪音的识别结果")
    print()
    print("  下一步: 修复 STM32 串口通信后, 把真实麦克风数据接入即可")
    print("=" * 72)


if __name__ == "__main__":
    main()
