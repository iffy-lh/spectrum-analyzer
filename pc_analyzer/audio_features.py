"""
音频特征提取模块

功能：
  - FFT频谱分析（幅度谱、功率谱）
  - 频段能量分解（sub_bass ~ high 六段）
  - 频谱特征：谱质心、谱带宽、谱滚降、谱平坦度
  - MFCC特征提取（梅尔倒谱系数）
  - 谐波检测（判断是否为人声）
  - 时域特征：RMS能量、过零率、峰值因子
"""

import numpy as np
from typing import Dict, Tuple, Optional


# ============================================================
# 纯numpy的简单峰值检测（替代scipy.signal.find_peaks）
# ============================================================
def find_peaks_simple(y: np.ndarray, height: float = 0, distance: int = 3) -> np.ndarray:
    """简单的峰值检测 — 纯numpy实现，替代scipy.signal.find_peaks"""
    if len(y) < 3:
        return np.array([], dtype=int)
    # 找局部极大值
    peaks = np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1
    # 高度过滤
    peaks = peaks[y[peaks] >= height]
    # 距离过滤（保留较高的峰）
    if len(peaks) > 1 and distance > 1:
        keep = [peaks[0]]
        for i in range(1, len(peaks)):
            if peaks[i] - keep[-1] >= distance:
                keep.append(peaks[i])
            elif y[peaks[i]] > y[keep[-1]]:
                keep[-1] = peaks[i]
        peaks = np.array(keep)
    return peaks


# ============================================================
# 频段定义（与音库保持一致）
# ============================================================
FREQ_BANDS: Dict[str, Tuple[float, float]] = {
    "sub_bass":  (0, 60),
    "bass":      (60, 250),
    "low_mid":   (250, 500),
    "mid":       (500, 2000),
    "high_mid":  (2000, 4000),
    "high":      (4000, 20000),
}


def compute_fft(samples: np.ndarray, sample_rate: float,
                window: str = "hann") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算实数FFT

    Args:
        samples: 时域采样点
        sample_rate: 采样率 (Hz)
        window: 窗函数类型 ("hann", "hamming", "blackman", "none")

    Returns:
        (freqs, magnitude, phase)
        - freqs: 频率轴 (Hz)
        - magnitude: 幅度谱
        - phase: 相位谱 (弧度)
    """
    n = len(samples)

    # 加窗
    if window == "hann":
        w = np.hanning(n)
    elif window == "hamming":
        w = np.hamming(n)
    elif window == "blackman":
        w = np.blackman(n)
    else:
        w = 1.0

    windowed = samples * w

    # 窗函数幅度补偿
    coherent_gain = np.mean(w)
    if coherent_gain > 1e-10:
        windowed = windowed / coherent_gain

    # FFT
    spectrum = np.fft.rfft(windowed)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)

    # 幅度归一化（除以N）
    magnitude = magnitude / n * 2  # ×2 补偿单边谱
    magnitude[0] = magnitude[0] / 2  # DC不×2

    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)

    return freqs, magnitude, phase


def compute_band_energies(freqs: np.ndarray, magnitude: np.ndarray) -> Dict[str, float]:
    """
    将频谱能量分配到各频段

    Returns:
        {band_name: energy_sum}, 未归一化
    """
    band_energies = {}
    for band_name, (lo, hi) in FREQ_BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        energy = np.sum(magnitude[mask] ** 2)
        band_energies[band_name] = float(energy)

    return band_energies


def compute_band_ratios(band_energies: Dict[str, float]) -> Dict[str, float]:
    """
    将频段能量归一化为占比

    Returns:
        {band_name: ratio (0~1)}, 所有频段之和=1
    """
    total = sum(band_energies.values())
    if total < 1e-12:
        return {k: 0.0 for k in band_energies}
    return {k: v / total for k, v in band_energies.items()}


def compute_spectral_features(freqs: np.ndarray, magnitude: np.ndarray) -> Dict[str, float]:
    """
    计算频谱特征

    Returns:
        {
            "centroid": 谱质心 (Hz) — 频谱"重心"，反映音色亮度
            "bandwidth": 谱带宽 (Hz) — 频谱扩散程度
            "rolloff": 谱滚降 (Hz) — 85%能量所在频率
            "flatness": 谱平坦度 (0~1) — 1=白噪声，0=纯音
            "crest": 谱峰因子 — 峰值/均值
            "skewness": 谱偏度 — 正=偏高频，负=偏低频
        }
    """
    eps = 1e-12

    # 功率谱
    power = magnitude ** 2
    total_power = np.sum(power)

    if total_power < eps:
        return {"centroid": 0, "bandwidth": 0, "rolloff": 0,
                "flatness": 1, "crest": 0, "skewness": 0}

    # 谱质心
    centroid = np.sum(freqs * power) / total_power

    # 谱带宽
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total_power)

    # 谱滚降 (85%)
    cumulative = np.cumsum(power)
    rolloff_idx = np.searchsorted(cumulative, 0.85 * cumulative[-1])
    rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    # 谱平坦度 (几何平均/算术平均)
    geo_mean = np.exp(np.mean(np.log(magnitude + eps)))
    arith_mean = np.mean(magnitude)
    flatness = float(geo_mean / (arith_mean + eps))

    # 谱峰因子
    crest = float(np.max(magnitude) / (arith_mean + eps))

    # 谱偏度
    skewness = float(np.sum(((freqs - centroid) / (bandwidth + eps)) ** 3 * power) / total_power)

    return {
        "centroid": float(centroid),
        "bandwidth": float(bandwidth),
        "rolloff": float(rolloff),
        "flatness": float(flatness),
        "crest": float(crest),
        "skewness": float(skewness),
    }


def compute_mfcc(samples: np.ndarray, sample_rate: float,
                 n_mfcc: int = 13, n_fft: int = 512) -> np.ndarray:
    """
    计算MFCC特征（梅尔倒谱系数）

    MFCC是语音/音频分类最常用的特征，模拟人耳非线性频率感知。

    Args:
        samples: 时域采样点
        sample_rate: 采样率
        n_mfcc: MFCC系数数量（通常13）
        n_fft: FFT点数

    Returns:
        (n_mfcc,) MFCC系数
    """
    n = len(samples)
    if n < n_fft:
        # 零填充
        padded = np.zeros(n_fft)
        padded[:n] = samples
        samples = padded
        n = n_fft

    # 加窗 + FFT
    windowed = samples[:n_fft] * np.hanning(n_fft)
    spectrum = np.abs(np.fft.rfft(windowed, n=n_fft))
    power = spectrum ** 2

    # Mel滤波器组
    n_mels = 26
    low_freq = 0
    high_freq = sample_rate / 2

    # Hz → Mel
    low_mel = 2595 * np.log10(1 + low_freq / 700)
    high_mel = 2595 * np.log10(1 + high_freq / 700)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    # 三角滤波器
    fbank = np.zeros((n_mels, len(power)))
    for m in range(n_mels):
        f_m_minus = bin_points[m]
        f_m = bin_points[m + 1]
        f_m_plus = bin_points[m + 2]

        for k in range(f_m_minus, f_m):
            fbank[m, k] = (k - bin_points[m]) / (bin_points[m + 1] - bin_points[m])
        for k in range(f_m, f_m_plus):
            fbank[m, k] = (bin_points[m + 2] - k) / (bin_points[m + 2] - bin_points[m + 1])

    # 滤波器组能量
    mel_energies = np.dot(fbank, power)
    mel_energies = np.where(mel_energies == 0, 1e-12, mel_energies)

    # log → DCT → MFCC
    log_mel = np.log(mel_energies)
    mfcc = np.zeros(n_mfcc)
    for i in range(n_mfcc):
        mfcc[i] = np.sum(log_mel * np.cos(np.pi * (i + 1) * (np.arange(n_mels) + 0.5) / n_mels))

    return mfcc


def detect_harmonic_structure(freqs: np.ndarray, magnitude: np.ndarray,
                               sample_rate: float) -> Dict:
    """
    检测是否具有谐波结构（用于区分人声和噪音）

    原理：
    - 人声和乐音有清晰的基频+谐波结构
    - 噪音通常频谱连续、无清晰谐波峰

    Returns:
        {
            "has_harmonic": bool,         # 是否存在谐波结构
            "harmonic_ratio": float,       # 谐波能量占比
            "fundamental_est": float,      # 估计基频 (Hz)
            "harmonic_count": int,         # 检测到的谐波数量
            "confidence": float,           # 置信度 0~1
        }
    """
    n = len(magnitude)
    if n < 10:
        return {"has_harmonic": False, "harmonic_ratio": 0,
                "fundamental_est": 0, "harmonic_count": 0, "confidence": 0}

    power = magnitude ** 2
    total_power = np.sum(power)

    # 只分析人声可能范围 80-500Hz 的峰值
    voice_mask = (freqs >= 60) & (freqs <= 600)
    voice_freqs = freqs[voice_mask]
    voice_mag = magnitude[voice_mask]
    voice_power = power[voice_mask]

    if len(voice_mag) < 5:
        return {"has_harmonic": False, "harmonic_ratio": 0,
                "fundamental_est": 0, "harmonic_count": 0, "confidence": 0}

    # 找峰值
    peak_indices = find_peaks_simple(voice_mag, height=np.max(voice_mag) * 0.15, distance=3)
    if len(peak_indices) < 2:
        return {"has_harmonic": False, "harmonic_ratio": 0,
                "fundamental_est": 0, "harmonic_count": 0, "confidence": 0}

    peak_freqs = voice_freqs[peak_indices]
    peak_mags = voice_mag[peak_indices]

    # 尝试估计基频：最低的强峰
    strong_mask = peak_mags > np.max(peak_mags) * 0.3
    if not np.any(strong_mask):
        return {"has_harmonic": False, "harmonic_ratio": 0,
                "fundamental_est": 0, "harmonic_count": 0, "confidence": 0}

    strong_peaks = peak_freqs[strong_mask]
    fundamental_est = strong_peaks[0]

    # 检验整数倍谐波
    harmonic_count = 0
    total_harmonic_power = 0.0

    for i, pf in enumerate(peak_freqs):
        for harmonic_n in range(1, 16):
            expected = fundamental_est * harmonic_n
            tolerance = max(5.0, fundamental_est * 0.15)
            if abs(pf - expected) < tolerance:
                harmonic_count += 1
                total_harmonic_power += voice_power[peak_indices[i]]
                break

    harmonic_ratio = total_harmonic_power / (np.sum(voice_power) + 1e-12)
    has_harmonic = harmonic_count >= 3 and harmonic_ratio > 0.3
    confidence = min(1.0, harmonic_count / 8.0) * min(1.0, harmonic_ratio / 0.5)

    return {
        "has_harmonic": has_harmonic,
        "harmonic_ratio": float(harmonic_ratio),
        "fundamental_est": float(fundamental_est),
        "harmonic_count": int(harmonic_count),
        "confidence": float(confidence),
    }


def compute_temporal_features(samples: np.ndarray, sample_rate: float) -> Dict[str, float]:
    """
    计算时域特征

    Returns:
        {
            "rms": RMS能量,
            "zcr": 过零率,
            "peak_factor": 峰值因子 (峰值/RMS),
            "rms_db": RMS相对满幅度的dB值
        }
    """
    eps = 1e-12

    rms = np.sqrt(np.mean(samples ** 2))

    # 过零率
    zcr = float(np.sum(np.abs(np.diff(np.sign(samples)))) / (2 * len(samples)))

    # 峰值因子
    peak = np.max(np.abs(samples))
    peak_factor = peak / (rms + eps)

    # RMS → dB (相对于满幅度 1.0)
    if rms > 0:
        rms_db = 20 * np.log10(rms + eps)
    else:
        rms_db = -96.0

    return {
        "rms": float(rms),
        "zcr": float(zcr),
        "peak_factor": float(peak_factor),
        "rms_db": float(rms_db),
    }


def extract_all_features(samples: np.ndarray, sample_rate: float) -> Dict:
    """
    提取所有音频特征（一站式接口）

    Args:
        samples: float数组，归一化到[-1, 1]或16bit整数
        sample_rate: 采样率 (Hz)

    Returns:
        完整特征字典
    """
    # 如果是16bit整数，转为float [-1, 1]
    if samples.dtype in [np.int16, np.int32]:
        samples = samples.astype(np.float32) / 32768.0

    n = len(samples)

    # FFT频谱
    freqs, magnitude, phase = compute_fft(samples, sample_rate)

    # 频段能量
    band_energies = compute_band_energies(freqs, magnitude)
    band_ratios = compute_band_ratios(band_energies)

    # 频谱特征
    spectral_feats = compute_spectral_features(freqs, magnitude)

    # MFCC
    mfcc = compute_mfcc(samples, sample_rate)

    # 谐波检测
    harmonic = detect_harmonic_structure(freqs, magnitude, sample_rate)

    # 时域特征
    temporal = compute_temporal_features(samples, sample_rate)

    return {
        "frame_size": len(samples),
        "sample_rate": sample_rate,
        "duration_ms": n / sample_rate * 1000,
        "freqs": freqs,
        "magnitude": magnitude,
        "band_energies": band_energies,
        "band_ratios": band_ratios,
        "spectral": spectral_feats,
        "mfcc": mfcc.tolist(),
        "harmonic": harmonic,
        "temporal": temporal,
    }
