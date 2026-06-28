"""
音库加载与查询模块

加载 sound_library.json，提供噪音类型查询、频段能量匹配等功能。
"""

import json
import os
from typing import Dict, List, Optional, Tuple


class SoundLibrary:
    """噪音音库管理器"""

    def __init__(self, library_path: Optional[str] = None):
        if library_path is None:
            library_path = os.path.join(os.path.dirname(__file__), "sound_library.json")
        with open(library_path, "r", encoding="utf-8") as f:
            self._data = json.load(f)

        self.categories: List[Dict] = self._data["categories"]
        self.freq_bands: Dict[str, Tuple[float, float]] = {
            k: tuple(v) for k, v in self._data["frequency_bands"].items()
            if not k.startswith("_")
        }
        self.annoyance_curve: Dict[str, float] = self._data["human_annoyance_curve"]
        self.a_weighting: Dict = self._data["a_weighting"]

    def get_category(self, cat_id: str) -> Optional[Dict]:
        """根据ID获取噪音类别"""
        for cat in self.categories:
            if cat["id"] == cat_id:
                return cat
        return None

    def get_all_categories(self) -> List[Dict]:
        return self.categories

    def get_speech_categories(self) -> List[Dict]:
        """获取所有与人声相关的类别"""
        return [c for c in self.categories if c.get("is_speech", False)]

    def get_non_speech_categories(self) -> List[Dict]:
        """获取所有非人声噪音类别"""
        return [c for c in self.categories if not c.get("is_speech", False)]

    def get_band_center(self, band_name: str) -> float:
        """获取频段的几何中心频率"""
        lo, hi = self.freq_bands[band_name]
        return (lo * hi) ** 0.5

    def compare_band_profile(self, measured: Dict[str, float], target: Dict[str, float]) -> float:
        """
        比较实测频段能量分布与目标分布的相似度

        Args:
            measured: 实测各频段能量比 {band_name: ratio}
            target: 目标频段能量比 {band_name: ratio}

        Returns:
            相似度 0~1，1=完全匹配
        """
        bands = list(self.freq_bands.keys())
        diff = 0.0
        for band in bands:
            m = measured.get(band, 0.0)
            t = target.get(band, 0.0)
            diff += (m - t) ** 2
        diff = (diff / len(bands)) ** 0.5
        # 将欧氏距离转为相似度（距离0→相似度1，距离0.5→相似度0）
        similarity = max(0.0, 1.0 - diff * 2.0)
        return similarity

    def get_a_weight(self, freq: float) -> float:
        """
        获取A计权衰减值（线性插值）

        A计权模拟人耳对不同频率的敏感度差异。
        """
        freqs = self.a_weighting["frequencies"]
        atten = self.a_weighting["attenuation"]

        if freq <= freqs[0]:
            return atten[0]
        if freq >= freqs[-1]:
            return atten[-1]

        for i in range(len(freqs) - 1):
            if freqs[i] <= freq <= freqs[i + 1]:
                t = (freq - freqs[i]) / (freqs[i + 1] - freqs[i])
                return atten[i] + t * (atten[i + 1] - atten[i])

        return 0.0

    def __repr__(self):
        return f"SoundLibrary({len(self.categories)} categories, {len(self.freq_bands)} bands)"


if __name__ == "__main__":
    lib = SoundLibrary()
    print(lib)
    print(f"\n噪音类别:")
    for cat in lib.categories:
        print(f"  [{cat['id']}] {cat['name']}: {cat['freq_range'][0]}-{cat['freq_range'][1]}Hz, "
              f"烦恼度={cat['annoyance_weight']}, 人声={'是' if cat.get('is_speech') else '否'}")
