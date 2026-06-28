"""
噪音分类器

两层分类策略：
  Layer 1 — 规则匹配：利用频段能量比 + 频谱特征做快速初筛
  Layer 2 — ML分类：提取特征向量 → 训练好的分类模型 → 最终判定

当前实现：
  - 规则匹配（开箱即用）
  - ML分类器骨架（RandomForest，需标注数据训练）
  - 说话声情境判定（is_speech + 时间/时长规则）
"""

import json
import os
import pickle
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime

from sound_library import SoundLibrary
from audio_features import (
    FREQ_BANDS, compute_band_ratios, compute_band_energies,
    compute_spectral_features, compute_mfcc, detect_harmonic_structure,
    compute_temporal_features, extract_all_features,
)


class NoiseClassifier:
    """噪音分类器 — 规则匹配 + ML"""

    def __init__(self, library: SoundLibrary):
        self.library = library
        self.ml_model = None
        self.ml_scaler = None

        # 规则匹配的决策逻辑
        self._build_rule_definitions()

    def _build_rule_definitions(self):
        """构建规则匹配的决策树"""

        # 频段能量比阈值（用于初筛）
        self.rules = [
            # (检测函数名, 匹配类别ID, 优先级)
            ("_detect_quiet", "quiet", 110),
            ("_detect_shrill", "shrill", 100),
            ("_detect_impact", "impact", 85),
            ("_detect_electrical", "electrical_hum", 80),
            ("_detect_speech", "speech", 70),
            ("_detect_construction", "construction", 65),
            ("_detect_traffic", "traffic", 55),
            ("_detect_hvac", "hvac", 45),
            ("_detect_crowd", "crowd", 35),
            ("_detect_natural", "natural", 25),
        ]

    # -----------------------------------------------------------
    # 规则检测函数（每个返回 True/False + confidence）
    # -----------------------------------------------------------

    def _detect_shrill(self, features: Dict) -> Tuple[bool, float]:
        """检测尖锐啸叫：高频能量占比极高"""
        ratios = features["band_ratios"]
        # high_mid + high 占比 > 60%
        hf_ratio = ratios.get("high_mid", 0) + ratios.get("high", 0)
        if hf_ratio > 0.55:
            return True, min(1.0, hf_ratio / 0.7)
        return False, 0.0

    def _detect_impact(self, features: Dict) -> Tuple[bool, float]:
        """检测撞击/瞬态：高峰值因子 + 短暂持续时间"""
        temporal = features["temporal"]
        spectral = features["spectral"]
        # 高峰值因子 + 短时长 + 高频宽
        score = 0.0
        if temporal["peak_factor"] > 4.0:
            score += 0.4
        if features["duration_ms"] < 500:
            score += 0.3
        if spectral["bandwidth"] > 800:
            score += 0.3
        return score > 0.5, score

    def _detect_electrical(self, features: Dict) -> Tuple[bool, float]:
        """检测工频噪声：50Hz基频 + 谐波，能量集中在低频"""
        ratios = features["band_ratios"]
        harmonic = features["harmonic"]
        # 基频在50Hz附近 + 有明显的谐波结构
        score = 0.0
        bass_energy = ratios.get("bass", 0)
        if bass_energy > 0.4:
            score += 0.3

        if harmonic["has_harmonic"]:
            fund = harmonic["fundamental_est"]
            if 45 <= fund <= 65:  # 50Hz工频
                score += 0.5
            elif abs(fund - 100) < 15 or abs(fund - 150) < 20:
                score += 0.2  # 可能是工频谐波
        # 高频能量极低
        hf_ratio = ratios.get("high_mid", 0) + ratios.get("high", 0)
        if hf_ratio < 0.08:
            score += 0.2

        return score > 0.5, min(1.0, score)

    def _detect_speech(self, features: Dict) -> Tuple[bool, float]:
        """检测单人口音：谐波结构 + 中频能量 + 合适的基频范围"""
        harmonic = features["harmonic"]
        ratios = features["band_ratios"]
        spectral = features["spectral"]
        temporal = features["temporal"]

        score = 0.0

        # 谐波结构
        if harmonic["has_harmonic"]:
            score += 0.35
            fund = harmonic["fundamental_est"]
            if 80 <= fund <= 300:  # 人声基频范围
                score += 0.15
        else:
            return False, 0.0  # 没有谐波结构 → 大概率不是说话声

        # 中频能量突出（说话声核心频段 250-2000Hz）
        mid_energy = ratios.get("low_mid", 0) + ratios.get("mid", 0)
        if 0.3 < mid_energy < 0.7:
            score += 0.15

        # 谱质心在人声范围
        if 500 < spectral["centroid"] < 1500:
            score += 0.1

        # 过零率中等（纯噪音ZCR更极端）
        if 0.05 < temporal["zcr"] < 0.4:
            score += 0.1

        # 低高频都不极端
        if ratios.get("sub_bass", 0) < 0.2 and ratios.get("high", 0) < 0.1:
            score += 0.15

        return score > 0.5, min(1.0, score)

    def _detect_construction(self, features: Dict) -> Tuple[bool, float]:
        """检测施工噪音：宽频段能量 + 高峰值因子 + 频谱不平坦"""
        ratios = features["band_ratios"]
        spectral = features["spectral"]
        temporal = features["temporal"]

        score = 0.0

        # 宽频分布：各频段都有能量
        band_spread = sum(1 for v in ratios.values() if v > 0.03)
        if band_spread >= 5:
            score += 0.3

        # 高峰值因子（冲击性）
        if temporal["peak_factor"] > 3.5:
            score += 0.25

        # 频谱不平坦（非白噪）
        if spectral["flatness"] < 0.3:
            score += 0.2

        # 中低频都有分量
        if ratios.get("bass", 0) > 0.1 and ratios.get("low_mid", 0) > 0.1 and ratios.get("mid", 0) > 0.1:
            score += 0.25

        return score > 0.5, min(1.0, score)

    def _detect_traffic(self, features: Dict) -> Tuple[bool, float]:
        """检测交通噪音：低频能量主导 + 连续 + 谱质心低"""
        ratios = features["band_ratios"]
        spectral = features["spectral"]
        harmonic = features["harmonic"]

        score = 0.0

        # 低频主导
        bass_dominant = ratios.get("sub_bass", 0) + ratios.get("bass", 0)
        if bass_dominant > 0.5:
            score += 0.4

        # 谱质心低
        if spectral["centroid"] < 600:
            score += 0.2

        # 无清晰谐波结构
        if not harmonic["has_harmonic"]:
            score += 0.2

        # 频谱较平坦（噪声特性）
        if spectral["flatness"] > 0.4:
            score += 0.2

        return score > 0.5, min(1.0, score)

    def _detect_hvac(self, features: Dict) -> Tuple[bool, float]:
        """检测空调/风扇：低频 + 低能量 + 极平坦 + 无谐波"""
        ratios = features["band_ratios"]
        spectral = features["spectral"]
        temporal = features["temporal"]
        harmonic = features["harmonic"]

        score = 0.0

        # 低频极端主导
        bass_dominant = ratios.get("sub_bass", 0) + ratios.get("bass", 0)
        if bass_dominant > 0.65:
            score += 0.3

        # 能量较低
        if temporal["rms_db"] < -30:
            score += 0.2

        # 频谱非常平坦
        if spectral["flatness"] > 0.6:
            score += 0.25

        # 无谐波
        if not harmonic["has_harmonic"]:
            score += 0.25

        return score > 0.55, min(1.0, score)

    def _detect_crowd(self, features: Dict) -> Tuple[bool, float]:
        """检测人群嘈杂：中频能量 + 无清晰谐波 + 中等平坦度"""
        ratios = features["band_ratios"]
        harmonic = features["harmonic"]
        spectral = features["spectral"]

        score = 0.0

        # 中频能量突出
        mid_energy = ratios.get("low_mid", 0) + ratios.get("mid", 0)
        if 0.3 < mid_energy < 0.65:
            score += 0.3

        # 无清晰谐波（多人混合 → 谐波被抹平）
        if not harmonic["has_harmonic"] or harmonic["harmonic_ratio"] < 0.25:
            score += 0.3

        # 中等平坦度
        if 0.3 < spectral["flatness"] < 0.6:
            score += 0.2

        # 带宽适中
        if 400 < spectral["bandwidth"] < 1500:
            score += 0.2

        return score > 0.55, min(1.0, score)

    def _detect_natural(self, features: Dict) -> Tuple[bool, float]:
        """检测自然声：宽频 + 高平坦度 + 常规峰值因子"""
        spectral = features["spectral"]
        temporal = features["temporal"]

        score = 0.0

        # 高平坦度（噪声特性）
        if spectral["flatness"] > 0.5:
            score += 0.35

        # 带宽宽
        if spectral["bandwidth"] > 1000:
            score += 0.2

        # 峰值因子适中（不尖锐）
        if 2.0 < temporal["peak_factor"] < 5.0:
            score += 0.15

        # 谱质心偏高（风/雨/虫鸣都偏高）
        if spectral["centroid"] > 500:
            score += 0.15

        # 不是工频
        harmonic = features["harmonic"]
        if harmonic.get("fundamental_est", 0) < 40 or harmonic["fundamental_est"] > 65:
            score += 0.15

        return score > 0.5, min(1.0, score)

    def _detect_quiet(self, features: Dict) -> Tuple[bool, float]:
        """检测安静环境：极低能量"""
        temporal = features["temporal"]
        if temporal["rms_db"] < -35:
            return True, min(1.0, abs(temporal["rms_db"]) / 45.0)
        return False, 0.0

    # -----------------------------------------------------------
    # 规则匹配主流程
    # -----------------------------------------------------------

    def classify_by_rules(self, features: Dict) -> List[Dict]:
        """
        基于规则的噪音分类

        Returns:
            匹配结果列表，按置信度降序
            [{category_id, name, confidence, is_speech}, ...]
        """
        results = []

        for rule_name, cat_id, priority in sorted(self.rules, key=lambda x: -x[2]):
            detect_fn = getattr(self, rule_name)
            matched, confidence = detect_fn(features)
            if matched and confidence > 0:
                cat = self.library.get_category(cat_id)
                if cat:
                    results.append({
                        "category_id": cat_id,
                        "name": cat["name"],
                        "confidence": round(confidence, 3),
                        "is_speech": cat.get("is_speech", False),
                        "rule": rule_name,
                    })

        # 如果没有任何类别匹配，fallback到最佳匹配
        if not results:
            results = self._fallback_match(features)

        return results

    def _fallback_match(self, features: Dict) -> List[Dict]:
        """后备匹配：用频段能量比找最接近的类别"""
        ratios = features["band_ratios"]
        best_cat = None
        best_sim = -1

        for cat in self.library.categories:
            target = cat.get("band_profile", {})
            sim = self.library.compare_band_profile(ratios, target)
            if sim > best_sim:
                best_sim = sim
                best_cat = cat

        if best_cat:
            return [{
                "category_id": best_cat["id"],
                "name": best_cat["name"],
                "confidence": round(best_sim, 3),
                "is_speech": best_cat.get("is_speech", False),
                "rule": "fallback_band_profile",
            }]
        return [{
            "category_id": "unknown",
            "name": "未知噪音",
            "confidence": 0.0,
            "is_speech": False,
            "rule": "fallback",
        }]

    # -----------------------------------------------------------
    # ML分类器（需训练）
    # -----------------------------------------------------------

    def _build_feature_vector(self, features: Dict) -> np.ndarray:
        """从特征字典构建ML特征向量"""
        vec = []
        # 频段能量比 6维
        for band in FREQ_BANDS:
            vec.append(features["band_ratios"].get(band, 0.0))
        # 频谱特征 6维
        for key in ["centroid", "bandwidth", "rolloff", "flatness", "crest", "skewness"]:
            vec.append(features["spectral"].get(key, 0.0))
        # 时域特征 4维
        for key in ["rms", "zcr", "peak_factor", "rms_db"]:
            vec.append(features["temporal"].get(key, 0.0))
        # 谐波特征 3维
        for key in ["harmonic_ratio", "fundamental_est", "harmonic_count"]:
            vec.append(features["harmonic"].get(key, 0.0))
        # MFCC 13维
        mfcc = features.get("mfcc", [])
        if isinstance(mfcc, list):
            vec.extend(mfcc[:13])
        else:
            vec.extend([0.0] * 13)
        return np.array(vec, dtype=np.float32)

    def train_ml(self, features_list: List[Dict], labels: List[int],
                 model_path: Optional[str] = None):
        """
        训练ML分类器

        Args:
            features_list: 特征字典列表
            labels: 类别标签列表 (0=traffic, 1=construction, ...)
            model_path: 模型保存路径，默认使用 default_ml_model.pkl
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import cross_val_score
        except ImportError:
            raise ImportError("ML训练需要 scikit-learn。运行: pip install scikit-learn")

        X = np.array([self._build_feature_vector(f) for f in features_list])
        y = np.array(labels)

        self.ml_scaler = StandardScaler()
        X_scaled = self.ml_scaler.fit_transform(X)

        self.ml_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
        )
        self.ml_model.fit(X_scaled, y)

        # 交叉验证
        scores = cross_val_score(self.ml_model, X_scaled, y, cv=3)
        acc = scores.mean()

        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "noise_classifier_model.pkl")

        with open(model_path, "wb") as f:
            pickle.dump({
                "model": self.ml_model,
                "scaler": self.ml_scaler,
                "accuracy": acc,
                "categories": [c["id"] for c in self.library.categories],
            }, f)

        return {"accuracy": float(acc), "model_path": model_path, "n_samples": len(y)}

    def load_ml(self, model_path: Optional[str] = None):
        """加载预训练的ML模型"""
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "noise_classifier_model.pkl")
        if not os.path.exists(model_path):
            return False
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        self.ml_model = data["model"]
        self.ml_scaler = data["scaler"]
        return True

    def classify_by_ml(self, features: Dict) -> Optional[Dict]:
        """ML分类（需要已训练/加载的模型）"""
        if self.ml_model is None:
            return None

        vec = self._build_feature_vector(features).reshape(1, -1)
        vec_scaled = self.ml_scaler.transform(vec)
        proba = self.ml_model.predict_proba(vec_scaled)[0]
        pred_class = int(self.ml_model.predict(vec_scaled)[0])
        cat_id = self.library.categories[pred_class]["id"]
        cat = self.library.get_category(cat_id)

        return {
            "category_id": cat_id,
            "name": cat["name"] if cat else "unknown",
            "confidence": round(float(proba[pred_class]), 3),
            "is_speech": cat.get("is_speech", False) if cat else False,
            "method": "ml",
        }

    # -----------------------------------------------------------
    # 综合分类
    # -----------------------------------------------------------

    def classify(self, features: Dict, use_ml: bool = False,
                 time_of_day: Optional[str] = None) -> Dict:
        """
        综合分类 — 规则 + 可选ML，含说话声情境判定

        Args:
            features: extract_all_features()的输出
            use_ml: 是否使用ML模型（需先训练/加载）
            time_of_day: "day" / "night" / None (auto-infer)

        Returns:
            {
                "primary": 最可能的类别,
                "candidates": 候选列表,
                "speech_context": 说话声情境判定,
                "feature_summary": 关键特征摘要
            }
        """
        # 规则匹配
        rule_results = self.classify_by_rules(features)

        # ML分类（如可用）
        ml_result = None
        if use_ml:
            ml_result = self.classify_by_ml(features)

        # 说话声情境判定
        speech_context = None
        primary = rule_results[0] if rule_results else {"category_id": "unknown", "name": "未知", "confidence": 0}

        if primary.get("is_speech"):
            speech_context = self._evaluate_speech_context(features, time_of_day)

        # 如果ML可用且置信度高，考虑覆盖规则结果
        if ml_result and ml_result["confidence"] > 0.7 and (not primary or ml_result["confidence"] > primary.get("confidence", 0) + 0.15):
            primary = ml_result

        results = {
            "primary": primary,
            "candidates": rule_results[:3],
            "ml_result": ml_result,
            "speech_context": speech_context,
            "feature_summary": {
                "rms_db": round(features["temporal"]["rms_db"], 1),
                "centroid": round(features["spectral"]["centroid"], 0),
                "has_harmonic": features["harmonic"]["has_harmonic"],
                "harmonic_ratio": round(features["harmonic"]["harmonic_ratio"], 3),
                "band_dominant": max(features["band_ratios"], key=features["band_ratios"].get),
            }
        }

        return results

    def _evaluate_speech_context(self, features: Dict, time_of_day: Optional[str] = None) -> Dict:
        """
        判定说话声的情境：是自己的声音还是他人的干扰

        规则：
        - 深夜(22:00-06:00)听到说话声 → 高烦恼
        - 白天持续15分钟以上 → 中烦恼
        - 短暂出现（<5s）→ 可能只是路过说话
        """
        if time_of_day is None:
            hour = datetime.now().hour
            time_of_day = "night" if (hour >= 22 or hour < 6) else "day"

        duration = features.get("duration_ms", 0) / 1000.0  # 秒

        context = {
            "is_speech": True,
            "time_of_day": time_of_day,
            "duration_s": round(duration, 1),
        }

        if time_of_day == "night":
            context["noise_level"] = "high"
            context["reason"] = "深夜时段他人说话，严重干扰休息"
            context["adjusted_annoyance"] = 0.80
        elif duration > 900:  # 15分钟
            context["noise_level"] = "medium"
            context["reason"] = "持续15分钟以上的他人交谈，干扰注意力"
            context["adjusted_annoyance"] = 0.65
        elif duration < 5:
            context["noise_level"] = "low"
            context["reason"] = "短暂路过说话，轻微干扰"
            context["adjusted_annoyance"] = 0.20
        else:
            context["noise_level"] = "medium"
            context["reason"] = "他人正常交谈"
            context["adjusted_annoyance"] = 0.50

        return context
