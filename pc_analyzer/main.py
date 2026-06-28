#!/usr/bin/env python3
"""
噪音识别分析系统 — 主入口

============================================================
架构：STM32(ADC采集) → 串口/USB → PC Python(分析引擎)
============================================================

用法：
    # 分析WAV文件（离线模式）
    python main.py --input test.wav

    # 实时串口模式（需要STM32连接）
    python main.py --serial COM3 --baud 921600

    # 生成模拟测试音频并分析
    python main.py --demo

    # 输出JSON格式报告
    python main.py --input test.wav --json
"""

import argparse
import json
import os
import sys
import time
import wave
import struct
from typing import Optional

import numpy as np

from sound_library import SoundLibrary
from audio_features import extract_all_features
from noise_classifier import NoiseClassifier
from annoyance_evaluator import AnnoyanceEvaluator


def read_wav(filepath: str) -> tuple:
    """读取WAV文件，返回 (samples, sample_rate)"""
    with wave.open(filepath, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()

        raw = wf.readframes(n_frames)

        if sample_width == 2:
            fmt = f"<{n_frames * n_channels}h"
            samples = np.array(struct.unpack(fmt, raw), dtype=np.float32)
        elif sample_width == 1:
            samples = np.array(struct.unpack(f"{n_frames * n_channels}B", raw), dtype=np.float32)
            samples = (samples - 128) * 256
        else:
            raise ValueError(f"不支持的采样位深: {sample_width}")

        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)

        samples = samples.astype(np.float32) / 32768.0

    return samples, sample_rate


def generate_demo_signals(sample_rate: int = 16000, duration_s: float = 3.0) -> list:
    """
    生成多种噪音类型演示信号

    Returns:
        [(label, samples), ...]
    """
    n = int(sample_rate * duration_s)
    t = np.arange(n) / sample_rate
    demo_signals = []

    # 1. 交通噪音（低频主导）
    traffic = (0.3 * np.sin(2 * np.pi * 80 * t) +
               0.2 * np.sin(2 * np.pi * 150 * t) +
               0.1 * np.sin(2 * np.pi * 300 * t) +
               np.random.normal(0, 0.15, n))
    demo_signals.append(("traffic", traffic))

    # 2. 施工噪音（宽带冲击）
    construction = np.zeros(n)
    for i in range(0, n, sample_rate // 4):  # 每秒4次冲击
        if i + 200 < n:
            # 噪声冲击
            impulse = np.random.normal(0, 1.5, 200)
            envelope = np.exp(-np.arange(200) * 0.02)
            construction[i:i+200] = impulse * envelope
    construction = construction * 0.3 + np.random.normal(0, 0.05, n)
    demo_signals.append(("construction", construction))

    # 3. 空调嗡鸣（低频稳态）
    hvac = (0.25 * np.sin(2 * np.pi * 60 * t) +
            0.15 * np.sin(2 * np.pi * 120 * t) +
            0.08 * np.sin(2 * np.pi * 180 * t) +
            np.random.normal(0, 0.05, n))
    demo_signals.append(("hvac", hvac))

    # 4. 说话声（模拟人声基频+谐波+共振峰）
    # 基频150Hz + 若干谐波，加一个2.5kHz的共振峰
    speech = (0.2 * np.sin(2 * np.pi * 150 * t) +
              0.12 * np.sin(2 * np.pi * 300 * t) +
              0.08 * np.sin(2 * np.pi * 450 * t) +
              0.05 * np.sin(2 * np.pi * 600 * t) +
              0.04 * np.sin(2 * np.pi * 2500 * t) +
              0.02 * np.sin(2 * np.pi * 3500 * t) +
              np.random.normal(0, 0.02, n))
    # 模拟语句间歇
    speech[int(0.5 * sample_rate):int(0.8 * sample_rate)] = 0
    speech[int(1.2 * sample_rate):int(1.4 * sample_rate)] = 0
    speech[int(2.0 * sample_rate):int(2.5 * sample_rate)] = 0
    demo_signals.append(("speech", speech))

    # 5. 尖锐啸叫（高频窄带）
    shrill = (0.3 * np.sin(2 * np.pi * 4000 * t) +
              0.15 * np.sin(2 * np.pi * 6000 * t) +
              np.random.normal(0, 0.03, n))
    demo_signals.append(("shrill", shrill))

    # 6. 工频噪声（50Hz谐波）
    hum = (0.3 * np.sin(2 * np.pi * 50 * t) +
           0.15 * np.sin(2 * np.pi * 150 * t) +
           0.08 * np.sin(2 * np.pi * 250 * t) +
           np.random.normal(0, 0.03, n))
    demo_signals.append(("electrical_hum", hum))

    # 7. 安静环境
    quiet = np.random.normal(0, 0.003, n)
    demo_signals.append(("quiet", quiet))

    return demo_signals


def analyze_frame(samples: np.ndarray, sample_rate: float,
                  library: SoundLibrary, classifier: NoiseClassifier,
                  evaluator: AnnoyanceEvaluator,
                  frame_idx: int = 0, use_ml: bool = False) -> dict:
    """分析一帧音频数据"""
    # 特征提取
    features = extract_all_features(samples, sample_rate)

    # 分类
    classification = classifier.classify(features, use_ml=use_ml)

    # 烦恼度评估
    annoyance = evaluator.evaluate(features, classification)

    # 组装结果
    result = {
        "frame": frame_idx,
        "duration_ms": round(features["duration_ms"], 1),
        "classification": classification["primary"],
        "candidates": classification["candidates"],
        "speech_context": classification["speech_context"],
        "annoyance": annoyance,
        "feature_summary": classification["feature_summary"],
    }

    return result


def print_report(result: dict):
    """打印人类可读的分析报告"""
    primary = result["classification"]
    anno = result["annoyance"]
    summary = result["feature_summary"]
    speech_ctx = result.get("speech_context")

    print("=" * 60)
    print(f"  噪音分析报告 — 帧 #{result['frame']} (时长 {result['duration_ms']:.0f}ms)")
    print("=" * 60)

    # 分类结果
    print(f"\n  [?25h 检测结果]: {primary['name']}")
    print(f"    置信度: {primary['confidence']:.2%}")
    if primary.get("is_speech") and speech_ctx:
        print(f"    [?25h 人声情境]: {speech_ctx['reason']}")
        print(f"    情境烦恼度: {speech_ctx['adjusted_annoyance']:.2f}")

    # 候选分类
    if len(result["candidates"]) > 1:
        print(f"\n  [?25h 其他可能]:")
        for c in result["candidates"][1:]:
            print(f"    - {c['name']} ({c['confidence']:.2%})")

    # 烦恼度
    print(f"\n  [?25h 综合评估]: {anno['level_emoji']} {anno['level']} (得分: {anno['overall_score']:.2f})")

    # 详细指标
    print(f"\n  [?25h 声学指标]:")
    print(f"    A计权声压级: {anno['breakdown']['spl_dba']} dBA")
    print(f"    谱质心:      {summary['centroid']:.0f} Hz")
    print(f"    主导频段:     {summary['band_dominant']}")
    print(f"    谐波结构:     {'有' if summary['has_harmonic'] else '无'} (比例: {summary['harmonic_ratio']:.2f})")

    # 分频段SPL
    print(f"\n  [?25h 各频段声压级 (dBA)]:")
    band_spls = anno["band_spls"]
    max_spl = max(band_spls.values())
    for band, spl in band_spls.items():
        bar_len = max(0, int((spl + 96) / max(96 + max_spl, 1) * 20))
        bar = "█" * bar_len
        print(f"    {band:10s}: {spl:5.1f} {bar}")

    # 健康提示
    if anno["health_warning"]:
        print(f"\n  [?25h 健康提示]: {anno['health_warning']}")

    # 建议
    if anno["recommendation"]:
        print(f"\n  💡 建议: {anno['recommendation']}")

    print("\n" + "=" * 60 + "\n")


STM32_SAMPLE_RATE = 18868  # v6: TIM2触发频率

def main():
    parser = argparse.ArgumentParser(
        description="噪音识别分析系统 — STM32频谱分析仪上位机",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python main.py --demo                    生成演示信号并分析
    python main.py --input test.wav           分析WAV文件
    python main.py --input test.wav --json    输出JSON格式
    python main.py --serial COM3              实时串口模式(需STM32)
        """
    )
    parser.add_argument("--input", "-i", help="输入WAV文件路径")
    parser.add_argument("--serial", "-s", help="串口设备 (如 COM3 或 /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=921600, help="串口波特率 (默认921600)")
    parser.add_argument("--sample-rate", type=int, default=16000, help="采样率 (默认16000)")
    parser.add_argument("--demo", action="store_true", help="生成演示信号并分析")
    parser.add_argument("--json", action="store_true", help="输出JSON格式")
    parser.add_argument("--ml", action="store_true", help="尝试使用ML模型（需已训练）")
    parser.add_argument("--frame-size", type=int, default=512, help="每帧采样点数 (默认512)")
    parser.add_argument("--hop-size", type=int, default=256, help="帧移采样点数 (默认256)")
    args = parser.parse_args()

    # 加载组件
    print("Loading sound library...", file=sys.stderr)
    library = SoundLibrary()
    classifier = NoiseClassifier(library)
    evaluator = AnnoyanceEvaluator(library)

    # 尝试加载ML模型
    if args.ml:
        if classifier.load_ml():
            print("ML model loaded.", file=sys.stderr)
        else:
            print("ML model not found, using rules only.", file=sys.stderr)

    results = []

    if args.demo:
        # 演示模式
        print("=" * 60)
        print("  演示模式 — 生成模拟噪音信号并分类")
        print("=" * 60)
        print()
        demo_signals = generate_demo_signals(sample_rate=args.sample_rate, duration_s=2.0)

        for label, samples in demo_signals:
            expected = library.get_category(label)
            print(f"\n{'─' * 60}")
            print(f"  真实标签: {expected['name'] if expected else label}")
            print(f"{'─' * 60}")

            # 分帧分析
            frame_size = args.frame_size
            hop_size = args.hop_size
            n_frames = (len(samples) - frame_size) // hop_size

            # 只分析第一帧（快速演示）
            if n_frames > 0:
                frame = samples[:frame_size]
                result = analyze_frame(frame, args.sample_rate,
                                       library, classifier, evaluator,
                                       use_ml=args.ml)
                results.append({"true_label": label, **result})
                if not args.json:
                    print_report(result)
            else:
                print(f"  (信号太短，跳过)")

    elif args.input:
        # WAV文件模式
        print(f"Reading {args.input}...", file=sys.stderr)
        samples, sr = read_wav(args.input)

        # 如果需要，重采样
        if sr != args.sample_rate:
            print(f"Resampling {sr} → {args.sample_rate} Hz...", file=sys.stderr)
            # 简单重采样（线性插值）
            old_len = len(samples)
            new_len = int(old_len * args.sample_rate / sr)
            samples = np.interp(
                np.linspace(0, old_len - 1, new_len),
                np.arange(old_len),
                samples
            )
        sample_rate = args.sample_rate

        # 分帧处理
        frame_size = args.frame_size
        hop_size = args.hop_size
        total_frames = max(1, (len(samples) - frame_size) // hop_size + 1)

        print(f"Analyzing... ({total_frames} frames)", file=sys.stderr)
        for i in range(total_frames):
            start = i * hop_size
            end = start + frame_size
            if end > len(samples):
                break
            frame = samples[start:end]
            result = analyze_frame(frame, sample_rate,
                                   library, classifier, evaluator,
                                   frame_idx=i, use_ml=args.ml)
            results.append(result)

            if not args.json:
                # 只打印第一帧和噪音类型变化的帧
                if i == 0 or (i > 0 and result["classification"]["category_id"] !=
                              results[-2]["classification"]["category_id"]):
                    print_report(result)

    elif args.serial:
        # 串口实时模式
        try:
            from serial_receiver import SerialReceiver, frame_to_samples
        except ImportError:
            print("需要 pyserial: pip install pyserial", file=sys.stderr)
            return

        print("=" * 60)
        print("  实时串口模式 — STM32 → PC 噪音分析")
        print("=" * 60)
        print(f"  串口: {args.serial} @ {args.baud}")
        print(f"  帧大小: {args.frame_size} 采样点")
        print(f"  (STM32端: 500000bps, TIM2触发18868Hz, 每帧256采样点)")
        print()

        rx = SerialReceiver(args.serial, args.baud)
        if not rx.connect():
            return
        rx.start()

        # 累积缓冲区
        buffer = []
        frame_count = 0

        print("等待STM32数据... (Ctrl+C 停止)")
        print()

        try:
            while True:
                frame = rx.get_frame(timeout=2.0)
                if frame is None:
                    if rx.frame_count == 0:
                        print("  等待超时 — 检查STM32是否已烧录v5固件并运行")
                        continue
                    else:
                        print("\n  数据流中断")
                        break

                # 转为float采样 (DC偏移2048)
                samples = frame_to_samples(frame, dc_offset=2048)
                buffer.extend(samples.tolist())

                # 积累足够采样点后分析
                while len(buffer) >= args.frame_size:
                    chunk = np.array(buffer[:args.frame_size], dtype=np.float32)
                    buffer = buffer[args.hop_size:]  # 滑动窗口

                    result = analyze_frame(chunk, STM32_SAMPLE_RATE,  # 18868 Hz
                                           library, classifier, evaluator,
                                           frame_idx=frame_count, use_ml=args.ml)
                    results.append(result)
                    frame_count += 1

                    # 实时打印
                    primary = result["classification"]
                    anno = result["annoyance"]
                    print(f"\r[{frame_count:4d}] {anno['level_emoji']} "
                          f"{primary['name']:16s} "
                          f"conf={primary['confidence']:.0%} "
                          f"| {anno['breakdown']['spl_dba']:.0f}dBA "
                          f"| {anno['level']} ({anno['overall_score']:.2f})  ",
                          end="", flush=True)

                    if frame_count % 10 == 0:
                        print()  # 每10帧换行

                # 显示统计
                if frame_count > 0 and frame_count % 50 == 0:
                    stats = rx.stats()
                    print(f"\n  --- 已接收: {stats['frames_received']} STM32帧, "
                          f"{stats['errors']} 错误, 分析: {frame_count} 帧 ---")

        except KeyboardInterrupt:
            print("\n\n用户停止")
        finally:
            stats = rx.stats()
            print(f"\n统计: 接收{stats['frames_received']} STM32帧, "
                  f"分析{frame_count} PC帧, "
                  f"{stats['errors']} 通信错误")
            rx.disconnect()

    else:
        parser.print_help()
        return

    # JSON输出
    if args.json and results:
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        output = {
            "system": "噪音识别分析系统 v1.0",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "demo" if args.demo else "file",
            "results": results,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2, cls=NpEncoder))


if __name__ == "__main__":
    main()
