# 版本快照 v20260820_085303

## 来源
- 上游产物目录：`~/Downloads/mtg_jamendo_meta/`（mtg_jamendo_meta 流水线产出）
- GPU 备份来源：`~/Downloads/mtg_jamendo_meta/gpu_backup/`（07:35 由 backup_gpu_mir.sh 从 AutoDL 实例拉取）
- OSS 冷备份前缀：`dataset/mtg-jamendo/jazz_test_500_audio-low/`（mp3，只写不读）

## 内容清单

| 文件 | 说明 | 行数/大小 |
|---|---|---|
| `dataset_joined.csv` | MIR 特征 × 官方 GT 元数据 联合表（track_id 主键 join） | 20 行 |
| `dataset_joined.json` | 同上 JSON 版（records） | 20 条 |
| `audio_manifest.csv` | 500 首音频 audio_id → 多源路径映射 | 500 行 |
| `readme.md` | 本说明 | - |

## audio_manifest.csv 字段

| 字段 | 说明 |
|---|---|
| audio_id | track_id（track_XXXXXXX，文件名去后缀，唯一主键） |
| oss_key | OSS 冷备份对象 key（`dataset/mtg-jamendo/jazz_test_500_audio-low/track_XXXXXXX.mp3`） |
| gpu_local_path | AutoDL 数据盘本地路径（实例销毁后失效，需重新下载源数据集） |
| official_source_path | MTG-Jamendo 官方相对路径（如 `55/955.mp3`） |
| duration_sec | 官方标注时长（秒） |
| file_bytes | OSS 对象字节数（list 获取，未下载实体） |
| oss_etag | OSS 对象 etag（完整性指纹） |
| format / sample_rate / channels | mp3 / 44100 / 2 |
| status | archived=已入OSS冷备份；missing_oss=OSS缺失需补传 |

## 数据规模与状态

- 音频全集：jazz 前 500 首（audio-low，30s 片段），OSS 500/500 完整
- MIR 特征：当前仅前 20 首已提取（CLAP-fusion 512d / torchcrepe f0 / madmom beats+BPM / pyloudnorm LUFS / librosa 统计）
- 元数据：已按 GPU 实际音频清单裁剪至 500 行（双向一一对应，零漏零余）
- track_id 唯一关联：禁止读 mp3 ID3 标签；所有 join/校验以文件名 track_id 为主键

## 关联键

`audio_manifest.audio_id` = `dataset_joined.track` = 官方元数据 `TRACK_ID` = mp3 文件名去后缀

## 强制约束（继承项目规则）

1. 🔒 OSS 音频只写不读：mp3 仅为冷备份，禁止读取/下载/流式拉取（训练推理读 GPU 本地）
2. 🔒 备份脚本 backup_gpu_mir.sh 仅用户明确指令时手动执行
3. 实例不用必须关机停计费

## 下一步
- 全量 500 首 MIR 提取（复用 GPU `/root/feat_20.py` 改 N=500，前 20 首断点跳过）
- 完成后重新生成 dataset_joined（500 行）并发布新版本快照
