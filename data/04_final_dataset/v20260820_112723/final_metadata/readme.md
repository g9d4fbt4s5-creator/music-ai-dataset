# 数据集元数据

## 基本信息

| 项目 | 内容 |
|------|------|
| 生成时间 | 2026-08-20T11:27:23.774232+08:00 |
| 来源快照 | gpu_backup_20260820_173500 |
| 数据来源 | mtg-jamendo |
| 样本总数 | 20 |
| 特征总数 | 32 |

## 文件清单

| 文件 | 说明 |
|------|------|
| dataset_joined.csv | 完整 join 结果（CSV） |
| dataset_joined.json | 完整 join 结果（JSON） |
| audio_manifest.csv | 音频清单 |
| feature_summary.json | 特征统计摘要 |

## 特征字段

### 基础特征（来自 all_features.csv）
track, status, lufs, bpm, n_beats, clap_dim, n_f0_frames, f0_conf_mean, rms_mean, centroid_mean, zcr_mean, mfcc1_mean, chroma_mean, dur_s, secs

### 节拍详细特征（来自 beats.csv）
n_beats_detailed, beat_interval_mean, beat_interval_std, beat_interval_min, beat_interval_max, first_beat_time, last_beat_time

### 基频详细特征（来自 f0.csv）
f0_conf_mean, f0_hz_mean, f0_hz_std, f0_hz_min, f0_hz_max, f0_hz_median, f0_confidence_mean, f0_confidence_std, f0_high_conf_ratio, f0_vocal_ratio

---

*由 join_snapshot_features.py 自动生成*
