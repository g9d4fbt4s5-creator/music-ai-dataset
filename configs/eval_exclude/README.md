# Eval Exclude 配置清单

## voided_genre_22.txt
- 22首La La Land传播的Musical Theatre样本
- 用户裁定：genre真清空（unlabeled），不是换种子重标
- L4 fuse_single_sample中自动检测，genre置空

## ace_studio_exclude.txt
- 5首ACE Studio生成音乐（source_type: ace_studio_generated_demucs_vocals）
- ADR-003：AI生成样本排除出训练/评估集
- 独立eval清单，用于OOD分析
