"""
噪音烦恼度/危害评估器

综合评估维度：
  1. A计权声压级 — 模拟人耳对不同频率的敏感度
  2. 噪音类型烦恼权重 — 不同噪音类型的基础烦恼度
  3. 时间情境调整 — 夜间加倍
  4. 暴露时长效应 — 持续暴露累积伤害
  5. 人声特殊处理 — 带语义信息的噪音更恼人
"""

from typing import Dict, Optional
import numpy as np

from sound_library import SoundLibrary


class AnnoyanceEvaluator:
    """噪音烦恼度/危害综合评估器"""

    def __init__(self, library: SoundLibrary):
        self.library = library

    def compute_a_weighted_spl(self, freqs: np.ndarray, magnitude: np.ndarray,
                                ref_pressure: float = 20e-6) -> Dict:
        """
        计算A计权声压级

        A计权模拟人耳对不同频率的敏感度：
        - 1-4kHz 最敏感（0dB，不加衰减）
        - <500Hz 逐渐不敏感（衰减大）
        - >8kHz 也不敏感

        Args:
            freqs: 频率轴 (Hz)
            magnitude: 幅度谱（线性）
            ref_pressure: 参考声压 (Pa)，默认20μPa

        Returns:
            {overall_spl_dba, band_spls: {...}}
        """
        eps = 1e-12

        # 幅度 → 声压级 (dB SPL，未计权)
        # 注：此处为相对计算，非校准后的绝对SPL
        magnitude_safe = np.maximum(magnitude, eps)
        spl = 20 * np.log10(magnitude_safe / ref_pressure)

        # A计权衰减
        a_weights = np.array([self.library.get_a_weight(f) for f in freqs])
        spl_weighted = spl + a_weights

        # 总A计权SPL（能量和）
        power_weighted = np.sum(10 ** (spl_weighted / 10))
        overall_spl_dba = float(10 * np.log10(power_weighted + 1e-12))

        # 各频段A计权SPL
        band_spls = {}
        for band_name, (lo, hi) in self.library.freq_bands.items():
            mask = (freqs >= lo) & (freqs < hi)
            if np.any(mask):
                band_power = np.sum(10 ** (spl_weighted[mask] / 10))
                band_spls[band_name] = float(10 * np.log10(band_power + 1e-12))
            else:
                band_spls[band_name] = -96.0

        return {
            "overall_spl_dba": round(overall_spl_dba, 1),
            "band_spls": {k: round(v, 1) for k, v in band_spls.items()},
        }

    def evaluate(self, features: Dict, classification: Dict,
                 exposure_duration_s: float = 0,
                 time_of_day: Optional[str] = None,
                 environment: str = "indoor") -> Dict:
        """
        综合烦恼度/危害评估

        Args:
            features: 音频特征
            classification: classify()输出
            exposure_duration_s: 已暴露时长（秒），0=瞬时
            time_of_day: "day"/"night"，None=自动
            environment: "indoor"/"outdoor"

        Returns:
            {
                overall_score:      综合烦恼度 0~1
                level:              等级标签
                health_warning:     健康提示
                breakdown:          各维度得分明细
                recommendation:     建议
            }
        """
        freqs = features["freqs"]
        magnitude = features["magnitude"]

        # 1. A计权SPL
        spl_info = self.compute_a_weighted_spl(freqs, magnitude)
        spl_dba = spl_info["overall_spl_dba"]

        # 2. SPL → 烦恼度映射
        # <30dBA: 极安静 → 0.05
        # 30-45: 安静 → 0.15
        # 45-55: 正常 → 0.30
        # 55-65: 较吵 → 0.50
        # 65-75: 吵闹 → 0.70
        # >75: 非常吵 → 0.90
        if spl_dba < 30:
            spl_annoyance = 0.05
        elif spl_dba < 45:
            spl_annoyance = 0.05 + (spl_dba - 30) / 15 * 0.10
        elif spl_dba < 55:
            spl_annoyance = 0.15 + (spl_dba - 45) / 10 * 0.15
        elif spl_dba < 65:
            spl_annoyance = 0.30 + (spl_dba - 55) / 10 * 0.20
        elif spl_dba < 75:
            spl_annoyance = 0.50 + (spl_dba - 65) / 10 * 0.20
        else:
            spl_annoyance = 0.70 + min(0.25, (spl_dba - 75) / 25 * 0.10)

        # 3. 噪音类型基础烦恼度
        primary = classification.get("primary", {})
        cat_id = primary.get("category_id", "unknown")
        cat = self.library.get_category(cat_id)
        type_annoyance = cat["annoyance_weight"] if cat else 0.5

        # 4. 时间情境调整
        if time_of_day is None:
            from datetime import datetime
            hour = datetime.now().hour
            time_of_day = "night" if (hour >= 22 or hour < 6) else "day"

        time_multiplier = 1.0
        if time_of_day == "night":
            time_multiplier = 1.5  # 夜间烦恼×1.5

        # 5. 暴露时长效应
        # 短期暴露 (<60s): 仅瞬时反应
        # 中期暴露 (1-30min): 开始累积
        # 长期暴露 (>30min): 显著累积
        if exposure_duration_s < 60:
            duration_multiplier = 1.0
        elif exposure_duration_s < 1800:
            duration_multiplier = 1.0 + (exposure_duration_s - 60) / 1740 * 0.20
        else:
            duration_multiplier = 1.20 + min(0.30, (exposure_duration_s - 1800) / 7200 * 0.10)

        # 6. 环境调整
        env_multiplier = 1.0
        if environment == "indoor":
            # 室内对噪音更敏感
            if spl_dba > 45:
                env_multiplier = 1.15

        # 7. 说话声特殊处理
        speech_context = classification.get("speech_context", {})
        if speech_context and speech_context.get("is_speech"):
            # 覆盖type_annoyance为情境调整后的值
            type_annoyance = speech_context.get("adjusted_annoyance", type_annoyance)

        # 8. 综合计算
        raw_score = (spl_annoyance * 0.40 +
                     type_annoyance * 0.35 +
                     spl_annoyance * type_annoyance * 0.25)  # 交互项
        raw_score = raw_score * time_multiplier * duration_multiplier * env_multiplier
        overall_score = round(min(1.0, raw_score), 2)

        # 9. 等级判定
        if overall_score < 0.20:
            level = "舒适"
            emoji = "🟢"
        elif overall_score < 0.40:
            level = "可接受"
            emoji = "🟡"
        elif overall_score < 0.60:
            level = "轻度烦恼"
            emoji = "🟠"
        elif overall_score < 0.80:
            level = "中度烦恼"
            emoji = "🔴"
        else:
            level = "严重干扰"
            emoji = "⛔"

        # 10. 健康提示
        health_warning = self._generate_health_warning(overall_score, spl_dba, cat, exposure_duration_s)

        # 11. 建议
        recommendation = self._generate_recommendation(overall_score, cat_id, spl_dba)

        return {
            "overall_score": overall_score,
            "level": level,
            "level_emoji": emoji,
            "health_warning": health_warning,
            "recommendation": recommendation,
            "breakdown": {
                "spl_dba": round(spl_dba, 1),
                "spl_annoyance": round(spl_annoyance, 2),
                "type_annoyance": round(type_annoyance, 2),
                "time_multiplier": round(time_multiplier, 2),
                "duration_multiplier": round(duration_multiplier, 2),
                "env_multiplier": round(env_multiplier, 2),
                "time_of_day": time_of_day,
                "environment": environment,
            },
            "band_spls": spl_info["band_spls"],
        }

    def _generate_health_warning(self, score: float, spl_dba: float,
                                  cat: Optional[Dict], duration_s: float) -> str:
        """生成健康提示"""
        warnings = []

        if spl_dba > 85:
            warnings.append("声压级超过85dBA，短时间暴露即可造成听力损伤")
        elif spl_dba > 70:
            warnings.append("声压级超过70dBA，长期暴露有听力损伤风险")

        if duration_s > 3600:
            warnings.append("持续暴露超过1小时，建议休息或佩戴降噪设备")

        if score >= 0.80:
            warnings.append("烦恼度极高，可能引发焦虑和睡眠障碍")
        elif score >= 0.60:
            warnings.append("烦恼度较高，长期处于此环境可能影响心理健康")

        if cat and cat.get("is_speech"):
            warnings.append("说话声含语义信息，对需要专注的任务影响更大")

        if not warnings:
            warnings.append("当前噪声水平对健康无明显影响")

        return "；".join(warnings)

    def _generate_recommendation(self, score: float, cat_id: str, spl_dba: float) -> str:
        """生成建议"""
        if score < 0.20:
            return "环境安静适宜，保持当前状态"
        elif score < 0.40:
            return "轻微噪音，无需特殊处理"
        elif score < 0.60:
            if cat_id == "speech":
                return "建议佩戴耳机或移至安静区域以减少话语干扰"
            elif cat_id == "hvac" or cat_id == "electrical_hum":
                return "低频持续噪音，建议降噪耳塞或白噪掩蔽"
            else:
                return "噪音干扰明显，建议短暂休息或降噪处理"
        elif score < 0.80:
            if cat_id == "speech":
                return "严重话语干扰！建议立即佩戴降噪耳机或更换环境"
            elif cat_id == "construction":
                return "施工噪音严重！建议佩戴专业降噪耳罩或远离噪音源"
            else:
                return "噪音干扰严重！建议降噪防护或更换环境"
        else:
            return "环境不适合正常活动！请立即采取降噪措施或离开此环境"
