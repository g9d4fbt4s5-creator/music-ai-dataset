# OSS RAM 权限分离配置指引

## 为什么要分离账号？

| 风险场景 | 单账号（全权限） | 双账号分离 |
|----------|------------------|------------|
| 备份脚本密钥泄露 | 攻击者可下载/删除所有数据 | 只能上传，不能下载或删除 |
| 恢复脚本密钥泄露 | 攻击者可下载/删除所有数据 | 只能下载，不能上传或删除 |
| 脚本 bug 误删 | 可能删除 OSS 上的备份 | 备份账号无删除权限，删不了 |

**核心原则**：日常运行的脚本只用最小权限的账号，高权限账号只在必要时使用。

---

## 账号设计

### 账号 1：OSS 备份账号（只写）

**用途**：日常上传备份（`upload_cache_to_oss.py`、`oss_gpu_backup_client.py`、`oss_local_client.py`）

**权限**：
- ✅ `PutObject`（上传对象）
- ✅ `ListObjects`（列出对象，用于校验）
- ✅ `HeadObject`（获取对象元数据，用于校验）
- ❌ `GetObject`（禁止下载）
- ❌ `DeleteObject`（禁止删除）

**环境变量**：
```
OSS_BACKUP_ACCESS_KEY_ID=xxx
OSS_BACKUP_ACCESS_KEY_SECRET=xxx
```

### 账号 2：OSS 恢复账号（只读）

**用途**：灾难恢复（`disaster_recovery.py`），新协作者初始化项目

**权限**：
- ✅ `GetObject`（下载对象）
- ✅ `ListObjects`（列出对象）
- ✅ `HeadObject`（获取对象元数据）
- ❌ `PutObject`（禁止上传）
- ❌ `DeleteObject`（禁止删除）

**环境变量**：
```
OSS_RECOVERY_ACCESS_KEY_ID=xxx
OSS_RECOVERY_ACCESS_KEY_SECRET=xxx
```

---

## 配置步骤（阿里云控制台）

### 第一步：创建 RAM 子用户

1. 登录阿里云控制台：https://ram.console.aliyun.com/users
2. 点击 **创建用户**
3. **登录名称**：`music-corpus-backup`（备份账号）
4. **显示名称**：`音乐语料项目-备份账号（只写）`
5. **访问方式**：勾选 **OpenAPI 调用访问**（生成 AccessKey）
6. 点击 **确定**
7. **保存 AccessKey ID 和 AccessKey Secret**（只显示一次，务必保存）

重复以上步骤，创建第二个用户：
- **登录名称**：`music-corpus-recovery`
- **显示名称**：`音乐语料项目-恢复账号（只读）`

### 第二步：创建自定义权限策略

#### 策略 1：OSS 备份只写策略

1. 进入 https://ram.console.aliyun.com/policies
2. 点击 **创建权限策略**
3. 选择 **脚本编辑**
4. 粘贴以下内容：

```json
{
    "Version": "1",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "oss:PutObject",
                "oss:ListObjects",
                "oss:GetObjectMeta",
                "oss:HeadObject"
            ],
            "Resource": [
                "acs:oss:*:*:music-ai-dataset-2026",
                "acs:oss:*:*:music-ai-dataset-2026/*"
            ]
        }
    ]
}
```

> **注意**：把 `music-ai-dataset-2026` 换成你的 bucket 名称。

5. **策略名称**：`MusicCorpusOSSBackupOnly`
6. **备注**：`音乐语料项目-OSS备份只写权限（Put+List+Head，禁止Get和Delete）`
7. 点击 **确定**

#### 策略 2：OSS 恢复只读策略

重复以上步骤，创建第二个策略：

```json
{
    "Version": "1",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "oss:GetObject",
                "oss:ListObjects",
                "oss:GetObjectMeta",
                "oss:HeadObject"
            ],
            "Resource": [
                "acs:oss:*:*:music-ai-dataset-2026",
                "acs:oss:*:*:music-ai-dataset-2026/*"
            ]
        }
    ]
}
```

- **策略名称**：`MusicCorpusOSSRecoveryReadOnly`
- **备注**：`音乐语料项目-OSS恢复只读权限（Get+List+Head，禁止Put和Delete）`

### 第三步：为子用户授权

1. 进入 https://ram.console.aliyun.com/users
2. 点击 `music-corpus-backup` 用户
3. 点击 **权限管理** 标签
4. 点击 **新增授权**
5. 搜索 `MusicCorpusOSSBackupOnly`，点击添加
6. 点击 **确定**

重复以上步骤，为 `music-corpus-recovery` 用户授权 `MusicCorpusOSSRecoveryReadOnly` 策略。

### 第四步：更新配置文件

把两个账号的 AccessKey 写入 `~/.config/music-corpus/.env`：

```bash
# 备份账号（只写）
OSS_BACKUP_ACCESS_KEY_ID=备份账号的AccessKey ID
OSS_BACKUP_ACCESS_KEY_SECRET=备份账号的AccessKey Secret

# 恢复账号（只读）
OSS_RECOVERY_ACCESS_KEY_ID=恢复账号的AccessKey ID
OSS_RECOVERY_ACCESS_KEY_SECRET=恢复账号的AccessKey Secret
```

---

## 验证权限分离

### 验证备份账号（只能上传，不能下载）

```bash
cd ~/music_corpus_project
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate labelstudio-env

# 测试上传（应该成功）
python3 -c "
import sys; sys.path.insert(0, 'scripts/utils')
from oss_local_client import OSSLocalClient
client = OSSLocalClient()
# 测试 list（应该成功）
keys = client.list_objects('')
print(f'ListObjects 成功，找到 {len(keys)} 个对象')
"

# 测试下载（应该失败，备份账号无 GetObject 权限）
python3 -c "
import sys; sys.path.insert(0, 'scripts/utils')
from config_loader import get_oss_config
import boto3
from botocore.config import Config
cfg = get_oss_config('backup')
s3 = boto3.client('s3', aws_access_key_id=cfg['access_key_id'],
    aws_secret_access_key=cfg['access_key_secret'],
    endpoint_url=cfg['endpoint'], region_name=cfg['region'],
    config=Config(s3={'addressing_style': 'virtual'}))
try:
    s3.get_object(Bucket=cfg['bucket'], Key='test')
    print('❌ 安全漏洞：备份账号能下载对象！')
except Exception as e:
    print(f'✅ 备份账号无法下载（预期行为）: {e}')
"
```

### 验证恢复账号（只能下载，不能上传）

```bash
python3 -c "
import sys; sys.path.insert(0, 'scripts/utils')
from config_loader import get_oss_config
import boto3
from botocore.config import Config
cfg = get_oss_config('recovery')
s3 = boto3.client('s3', aws_access_key_id=cfg['access_key_id'],
    aws_secret_access_key=cfg['access_key_secret'],
    endpoint_url=cfg['endpoint'], region_name=cfg['region'],
    config=Config(s3={'addressing_style': 'virtual'}))
try:
    s3.put_object(Bucket=cfg['bucket'], Key='test_security_check', Body=b'test')
    print('❌ 安全漏洞：恢复账号能上传对象！')
    # 清理测试文件
    s3.delete_object(Bucket=cfg['bucket'], Key='test_security_check')
except Exception as e:
    print(f'✅ 恢复账号无法上传（预期行为）: {e}')
"
```

---

## 权限矩阵速查表

| 操作 | 备份账号 (OSS_BACKUP) | 恢复账号 (OSS_RECOVERY) |
|------|----------------------|------------------------|
| 上传对象 (PutObject) | ✅ | ❌ |
| 列出对象 (ListObjects) | ✅ | ✅ |
| 获取元数据 (HeadObject) | ✅ | ✅ |
| 下载对象 (GetObject) | ❌ | ✅ |
| 删除对象 (DeleteObject) | ❌ | ❌ |
| 删除 Bucket | ❌ | ❌ |
| 修改 Bucket 权限 | ❌ | ❌ |

---

## 安全最佳实践

1. **定期轮换 AccessKey**：建议每 90 天轮换一次
2. **启用 MFA**：RAM 子用户登录控制台时启用多因素认证
3. **最小权限原则**：只授予必要的权限，不要给 AdministratorAccess
4. **监控异常访问**：在阿里云日志服务中配置 OSS 访问日志告警
5. **不要硬编码密钥**：所有脚本从环境变量或 `~/.config/music-corpus/.env` 读取
6. **.gitignore 覆盖 .env**：确保密钥文件不会被提交到 git

---

## 故障排查

### 问题：备份账号上传失败，提示 NoPermission

**原因**：权限策略中的 bucket 名称写错了，或者策略没有正确授权

**解决**：
1. 检查策略中的 `Resource` 字段，确认 bucket 名称正确
2. 确认策略已正确附加到用户
3. 在 RAM 控制台的 **权限管理** → **权限策略** 中查看策略内容

### 问题：恢复账号下载失败，提示 NoPermission

**原因**：同上，检查恢复账号的策略

### 问题：两个账号都无法访问 OSS

**原因**：可能是 bucket 权限设置了私有，且 RAM 策略未生效

**解决**：
1. 确认 bucket 是标准存储，不是归档存储
2. 确认 RAM 策略中的 `Resource` 包含 bucket 和 bucket/*
3. 等待 1-2 分钟让策略生效（RAM 策略有延迟）
