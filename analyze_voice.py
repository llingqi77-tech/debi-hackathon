#!/usr/bin/env python3
"""24镜视频音高（F0）分析脚本：检测语音基频，按角色和年龄段分组比对"""

import numpy as np
import wave
import os
import json
from pathlib import Path

AUDIO_DIR = Path("/Users/linglingqi/Downloads/她的花向上开-24镜完整版/audio")
SR = 16000  # 采样率

# 镜头台词映射（来自提示词文档）
SHOT_INFO = {
    1:  {"dialogue": "无", "speakers": []},
    2:  {"dialogue": "渊画外音", "speakers": ["yuan_20"]},
    3:  {"dialogue": "精灵+渊", "speakers": ["fairy", "yuan_20"]},
    4:  {"dialogue": "精灵", "speakers": ["fairy"]},
    5:  {"dialogue": "无", "speakers": []},
    6:  {"dialogue": "渊画外音", "speakers": ["yuan_20"]},
    7:  {"dialogue": "精灵+渊", "speakers": ["fairy", "yuan_20"]},
    8:  {"dialogue": "精灵+渊", "speakers": ["fairy", "yuan_20"]},
    9:  {"dialogue": "无", "speakers": []},
    10: {"dialogue": "渊画外音", "speakers": ["yuan_30"]},
    11: {"dialogue": "渊", "speakers": ["yuan_30"]},
    12: {"dialogue": "精灵+渊", "speakers": ["fairy", "yuan_30"]},
    13: {"dialogue": "无", "speakers": []},
    14: {"dialogue": "渊画外音", "speakers": ["yuan_30"]},
    15: {"dialogue": "精灵", "speakers": ["fairy"]},
    16: {"dialogue": "渊", "speakers": ["yuan_40"]},
    17: {"dialogue": "渊", "speakers": ["yuan_40"]},
    18: {"dialogue": "精灵", "speakers": ["fairy"]},
    19: {"dialogue": "无", "speakers": []},
    20: {"dialogue": "精灵+渊", "speakers": ["fairy", "yuan_40"]},
    21: {"dialogue": "精灵", "speakers": ["fairy"]},
    22: {"dialogue": "渊画外音", "speakers": ["yuan_40"]},
    23: {"dialogue": "精灵+渊", "speakers": ["fairy", "yuan_40"]},
    24: {"dialogue": "渊画外音", "speakers": ["yuan_40"]},
}


def read_wav(path):
    """读取 WAV 文件，返回 float32 数组"""
    with wave.open(str(path), 'rb') as wf:
        n = wf.getnframes()
        raw = wf.readframes(n)
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return data


def frame_signal(signal, frame_len=400, hop_len=160):
    """分帧"""
    n_frames = 1 + (len(signal) - frame_len) // hop_len
    frames = np.zeros((n_frames, frame_len))
    for i in range(n_frames):
        start = i * hop_len
        frames[i] = signal[start:start + frame_len]
    return frames


def compute_energy(frames):
    """计算每帧能量（dB）"""
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-10)
    return 20 * np.log10(rms + 1e-10)


def autocorr_pitch(frame, sr=SR, fmin=80, fmax=500):
    """自相关法估计单帧基频"""
    frame = frame - np.mean(frame)
    # 加窗
    window = np.hanning(len(frame))
    frame = frame * window

    corr = np.correlate(frame, frame, mode='full')
    corr = corr[len(corr) // 2:]

    # 只在有效范围内找峰值
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    if max_lag >= len(corr):
        max_lag = len(corr) - 1

    segment = corr[min_lag:max_lag]
    if len(segment) == 0 or np.max(segment) <= 0:
        return 0

    peak_idx = np.argmax(segment) + min_lag
    # 清晰度检查：峰值要明显高于零延迟值的一定比例
    clarity = corr[peak_idx] / (corr[0] + 1e-10)
    if clarity < 0.25:
        return 0

    # 抛物线插值提高精度
    if 0 < peak_idx < len(corr) - 1:
        x0, x1, x2 = corr[peak_idx - 1], corr[peak_idx], corr[peak_idx + 1]
        denom = (x0 - 2 * x1 + x2)
        if abs(denom) > 1e-10:
            peak_idx = peak_idx + 0.5 * (x0 - x2) / denom

    return sr / peak_idx


def analyze_shot(wav_path):
    """分析单个镜头的语音特征"""
    signal = read_wav(wav_path)
    duration = len(signal) / SR

    frame_len = int(0.025 * SR)  # 25ms
    hop_len = int(0.010 * SR)    # 10ms
    frames = frame_signal(signal, frame_len, hop_len)
    energy = compute_energy(frames)

    # 语音帧判定：能量高于阈值
    voice_threshold = -45  # dB
    voice_mask = energy > voice_threshold

    # 对语音帧估计音高
    pitches = []
    voice_energies = []
    for i in range(len(frames)):
        if voice_mask[i]:
            f0 = autocorr_pitch(frames[i], SR)
            if f0 > 0:
                pitches.append(f0)
                voice_energies.append(energy[i])

    # 语音占比
    voice_ratio = np.sum(voice_mask) / len(voice_mask)

    # 语音起止时间
    voice_times = np.where(voice_mask)[0] * hop_len / SR
    if len(voice_times) > 0:
        voice_start = voice_times[0]
        voice_end = voice_times[-1]
    else:
        voice_start = voice_end = 0

    result = {
        "duration": round(duration, 2),
        "voice_ratio": round(voice_ratio, 3),
        "voice_start": round(voice_start, 2),
        "voice_end": round(voice_end, 2),
        "mean_energy_db": round(np.mean(energy), 1),
        "n_voiced_frames": len(pitches),
    }

    if len(pitches) > 0:
        pitches = np.array(pitches)
        result.update({
            "f0_mean": round(float(np.mean(pitches)), 1),
            "f0_median": round(float(np.median(pitches)), 1),
            "f0_std": round(float(np.std(pitches)), 1),
            "f0_min": round(float(np.min(pitches)), 1),
            "f0_max": round(float(np.max(pitches)), 1),
            "f0_p25": round(float(np.percentile(pitches, 25)), 1),
            "f0_p75": round(float(np.percentile(pitches, 75)), 1),
        })
    else:
        result.update({
            "f0_mean": None, "f0_median": None, "f0_std": None,
            "f0_min": None, "f0_max": None,
        })

    return result


def main():
    results = {}
    print("=" * 90)
    print(f"{'镜头':<6} {'台词':<10} {'语音占比':<8} {'F0均值':<8} {'F0中位':<8} {'F0标准差':<8} {'F0范围':<14} {'语音段':<12}")
    print("-" * 90)

    for num in range(1, 25):
        wav_path = AUDIO_DIR / f"镜头{num:02d}.wav"
        if not wav_path.exists():
            continue
        info = SHOT_INFO[num]
        r = analyze_shot(wav_path)
        r["dialogue"] = info["dialogue"]
        r["speakers"] = info["speakers"]
        results[num] = r

        f0_str = f"{r['f0_mean']}" if r['f0_mean'] else "N/A"
        f0med_str = f"{r['f0_median']}" if r['f0_median'] else "N/A"
        f0std_str = f"{r['f0_std']}" if r['f0_std'] else "N/A"
        f0range_str = f"{r['f0_min']}-{r['f0_max']}" if r['f0_min'] else "N/A"
        voice_seg = f"{r['voice_start']}-{r['voice_end']}s"

        print(f"{num:<6} {info['dialogue']:<10} {r['voice_ratio']:<8.3f} "
              f"{f0_str:<8} {f0med_str:<8} {f0std_str:<8} {f0range_str:<14} {voice_seg:<12}")

    # 按角色分组统计
    print("\n" + "=" * 90)
    print("按角色/年龄段分组统计")
    print("=" * 90)

    groups = {
        "精灵(全部出场)": [3, 4, 7, 8, 12, 15, 18, 20, 21, 23],
        "渊-20岁": [2, 3, 6, 7, 8],
        "渊-30岁": [10, 11, 12, 14],
        "渊-40岁": [16, 17, 20, 22, 23, 24],
        "渊-20岁(单人)": [2, 6],
        "渊-30岁(单人)": [10, 11, 14],
        "渊-40岁(单人)": [16, 17, 22, 24],
        "精灵(单人)": [4, 15, 18, 21],
    }

    group_stats = {}
    for gname, shot_nums in groups.items():
        f0_all = []
        for n in shot_nums:
            if n in results and results[n]["f0_mean"]:
                f0_all.append(results[n]["f0_mean"])
        if f0_all:
            f0_all = np.array(f0_all)
            group_stats[gname] = {
                "shots": shot_nums,
                "n": len(f0_all),
                "mean_of_means": round(float(np.mean(f0_all)), 1),
                "std_of_means": round(float(np.std(f0_all)), 1),
                "min": round(float(np.min(f0_all)), 1),
                "max": round(float(np.max(f0_all)), 1),
            }
            print(f"\n{gname} ({len(f0_all)}镜有语音):")
            print(f"  各镜F0均值: {[results[n]['f0_mean'] for n in shot_nums if n in results and results[n]['f0_mean']]}")
            print(f"  均值的均值: {group_stats[gname]['mean_of_means']} Hz")
            print(f"  镜间标准差: {group_stats[gname]['std_of_means']} Hz")
            print(f"  范围: {group_stats[gname]['min']} - {group_stats[gname]['max']} Hz")
        else:
            print(f"\n{gname}: 无有效语音数据")

    # 保存完整结果
    output = {"per_shot": results, "group_stats": group_stats}
    out_path = AUDIO_DIR.parent / "voice_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n完整结果已保存: {out_path}")


if __name__ == "__main__":
    main()
