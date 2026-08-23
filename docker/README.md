# Docker 化部署指南

> AI音乐数据集处理流水线 — 容器化部署，消除"本地能跑，服务器跑崩"的环境不一致问题。

## 一、为什么用 Docker？

### 痛点

你的项目依赖复杂：
- **PyTorch 生态**：torch, torchaudio, transformers, demucs, MERT, CLAP
- **TensorFlow 生态**：YAMNet（TF 2.15，和 PyTorch 环境冲突）
- **音频处理**：librosa, soundfile, pyloudnorm, ffmpeg
- **ASR**：openai-whisper, funasr, faster-whisper
- **系统工具**：ffmpeg, chromaprint (fpcalc)

换一台新的 AutoDL 实例，手动装环境要 30 分钟+，还经常版本冲突。

### Docker 解决

```
新实例启动 → docker pull 镜像 → nvidia-docker run（挂载数据盘）→ 直接跑脚本
全程 5 分钟，零环境调试
```

## 二、镜像清单

| 镜像 | 用途 | 基础环境 | 预估体积 |
|------|------|---------|---------|
| `music-corpus-gpu` | 主镜像：特征提取/源分离/ASR/聚类 | PyTorch 2.2.2 + CUDA 12.1 | ~8-12GB |
| `music-corpus-yamnet` | YAMNet专用：音频分类 | TensorFlow 2.15 + CUDA | ~3-5GB |

### 为什么两个镜像？

YAMNet 用 TensorFlow 2.15，要求 numpy < 1.26；主镜像用 PyTorch，numpy 1.26.4。两者混在一起会有版本冲突，所以分开更干净。

## 三、快速开始

### 3.1 构建镜像

```bash
# 构建所有镜像
bash docker/build.sh

# 只构建GPU主镜像
bash docker/build.sh gpu

# 只构建YAMNet镜像
bash docker/build.sh yamnet

# 无缓存构建（强制重新下载所有依赖）
bash docker/build.sh --no-cache
```

### 3.2 登录 GitHub Container Registry

```bash
# 1. 在 GitHub 生成 Personal Access Token
#    地址：https://github.com/settings/tokens
#    权限：write:packages, read:packages, delete:packages

# 2. 登录
echo "YOUR_GITHUB_TOKEN" | docker login ghcr.io -u g9d4fbt4s5-creator --password-stdin
```

### 3.3 推送镜像

```bash
# 推送所有镜像
bash docker/push.sh

# 只推送GPU主镜像
bash docker/push.sh gpu
```

### 3.4 在 AutoDL 上使用

```bash
# 1. 拉取镜像
docker pull ghcr.io/g9d4fbt4s5-creator/music-corpus-gpu:latest

# 2. 运行脚本（挂载数据盘）
nvidia-docker run --rm \
    -v /root/autodl-tmp:/data \
    ghcr.io/g9d4fbt4s5-creator/music-corpus-gpu:latest \
    python scripts/01_preprocess/04_extract_features.py --input /data/manifest.csv

# 3. 进入交互模式（调试用）
nvidia-docker run -it --rm \
    -v /root/autodl-tmp:/data \
    ghcr.io/g9d4fbt4s5-creator/music-corpus-gpu:latest \
    bash
```

## 四、数据挂载说明

### 重要原则

**镜像只放代码 + 依赖，数据全部外部挂载。**

| 数据类型 | 容器内路径 | 宿主机路径（AutoDL） | 说明 |
|---------|-----------|---------------------|------|
| 原始音频 | `/data/raw_audio` | `/root/autodl-tmp/raw_audio` | 输入 |
| 母版FLAC | `/data/processed_master` | `/root/autodl-tmp/processed_master` | 中间产物 |
| 切片 | `/data/segments` | `/root/autodl-tmp/segments` | 输出 |
| 特征 | `/data/features` | `/root/autodl-tmp/features` | 输出 |
| 元数据 | `/data/meta` | `/root/autodl-tmp/meta` | 输出 |
| 模型缓存 | `/app/models` | `/root/autodl-tmp/models` | 避免重复下载 |

### 挂载示例

```bash
nvidia-docker run --rm \
    -v /root/autodl-tmp/raw_audio:/data/raw_audio \
    -v /root/autodl-tmp/processed_master:/data/processed_master \
    -v /root/autodl-tmp/segments:/data/segments \
    -v /root/autodl-tmp/features:/data/features \
    -v /root/autodl-tmp/meta:/data/meta \
    -v /root/autodl-tmp/models:/app/models \
    ghcr.io/g9d4fbt4s5-creator/music-corpus-gpu:latest \
    python scripts/etl_pipeline.py --input /data/raw_audio --output /data
```

## 五、本地开发（docker-compose）

### 5.1 启动开发环境

```bash
# 启动GPU主镜像（交互模式，代码热重载）
docker compose run gpu bash

# 启动YAMNet镜像
docker compose run yamnet bash
```

### 5.2 代码热重载

`docker-compose.yml` 中已经配置了：
```yaml
volumes:
  - ./scripts:/app/scripts    # 本地代码修改，容器内立即生效
  - ./configs:/app/configs
```

修改本地 `scripts/` 下的文件，容器内不需要重新构建镜像，直接生效。

## 六、常见问题

### Q1: `nvidia-docker: command not found`

AutoDL 实例通常预装了 nvidia-docker2。如果没有：
```bash
# 检查
docker info | grep -i nvidia

# 如果没有，安装nvidia-docker2
# 参考：https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Q2: 容器内看不到 GPU

```bash
# 检查容器内是否能看到GPU
nvidia-docker run --rm ghcr.io/g9d4fbt4s5-creator/music-corpus-gpu:latest \
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

# 如果输出 False 0，检查：
# 1. 是否用了 nvidia-docker run 而不是 docker run
# 2. 宿主机是否有GPU：nvidia-smi
# 3. 基础镜像的CUDA版本是否和驱动兼容
```

### Q3: 镜像太大，拉取慢

```bash
# 方法1：用阿里云镜像加速器（AutoDL国内访问快）
# 编辑 /etc/docker/daemon.json
{
  "registry-mirrors": ["https://your-mirror.mirror.aliyuncs.com"]
}

# 方法2：只拉取需要的标签
docker pull ghcr.io/g9d4fbt4s5-creator/music-corpus-gpu:latest
```

### Q4: 如何添加新的 Python 依赖？

```bash
# 方法1：修改 Dockerfile，重新构建
# 编辑 docker/Dockerfile.gpu，在 RUN pip install 中添加新包
bash docker/build.sh gpu --no-cache

# 方法2：容器内临时安装（调试用，不持久化）
nvidia-docker run -it --rm ghcr.io/g9d4fbt4s5-creator/music-corpus-gpu:latest bash
pip install new-package
```

### Q5: YAMNet 镜像和主镜像如何协作？

```bash
# 方案A：分别运行，通过共享数据目录传递结果
# 1. YAMNet镜像跑分类，结果存 /data/yamnet_output.csv
nvidia-docker run --rm -v /root/autodl-tmp:/data \
    ghcr.io/g9d4fbt4s5-creator/music-corpus-yamnet:latest \
    python scripts/00.5_cleaning/yamnet_infer.py --input /data/manifest.csv --output /data/yamnet_output.csv

# 2. 主镜像读取YAMNet结果，继续后续处理
nvidia-docker run --rm -v /root/autodl-tmp:/data \
    ghcr.io/g9d4fbt4s5-creator/music-corpus-gpu:latest \
    python scripts/etl_pipeline.py --yamnet-results /data/yamnet_output.csv
```

## 七、镜像体积优化

### 当前优化措施

- ✅ 基础镜像用 `runtime` 版（不是 `devel` 版，小很多）
- ✅ `pip install --no-cache-dir`（不保留pip缓存）
- ✅ `.dockerignore` 排除数据/模型/日志/音频
- ✅ apt-get 安装后 `rm -rf /var/lib/apt/lists/*`
- ✅ 模型权重不打进镜像，运行时下载

### 进一步优化（可选）

```dockerfile
# 多阶段构建（multi-stage build）
# 编译阶段装所有依赖，运行阶段只拷贝需要的文件
# 可以减少 30-50% 镜像体积
```

## 八、版本管理

```bash
# 构建指定版本
VERSION=v1.0.0 bash docker/build.sh

# 推送指定版本
VERSION=v1.0.0 bash docker/push.sh

# 查看所有版本
docker images ghcr.io/g9d4fbt4s5-creator/music-corpus-gpu
```

## 九、相关文件

| 文件 | 说明 |
|------|------|
| `docker/Dockerfile.gpu` | GPU主镜像构建文件 |
| `docker/Dockerfile.yamnet` | YAMNet专用镜像构建文件 |
| `docker/build.sh` | 构建脚本 |
| `docker/push.sh` | 推送脚本 |
| `docker/docker-compose.yml` | 本地开发用compose配置 |
| `.dockerignore` | 排除文件列表 |

---

**一句话总结**：Docker = 带全套环境的便携式盒子。构建一次，到处运行，再也不用"在我电脑能跑"。
