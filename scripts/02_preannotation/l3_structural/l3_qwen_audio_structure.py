#!/usr/bin/env python3
"""
l3_qwen_audio_structure.py — Qwen-Omni 多模态黄金集 L3 结构标注

功能：
- 用 Qwen3.5-Omni-Flash 多模态 API 直接听音频，生成段落结构/乐器/情绪/caption
- 音频预处理：FLAC > 8MB → 转 MP3 320kbps；仍 > 10MB → 取前3分钟
- 输出：原始 Qwen 输出 JSON + V4 标准格式 JSON
- 支持缓存、重试、单首指定

用法：
    python l3_qwen_audio_structure.py \
        --golden-manifest data/03_human_annotation/golden_set/golden_manifest.csv \
        --master-dir data/01_preprocess/processed_master/ \
        --output-dir data/02_preannotation/l3_structural/qwen_omni/ \
        --api-key sk-xxx \
        --model qwen3.5-omni-flash

依赖：
    - requests
    - ffmpeg / ffprobe
"""

import os
import sys
import json
import time
import base64
import argparse
import logging
import subprocess
import requests

# 自动加载项目根目录的 .env（API key 等配置）
# 脚本在 scripts/02_preannotation/l3_structural/ 下，往上3级是项目根
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent.parent.parent
_env_path = _project_root / ".env"

def _load_env_manually(env_path):
    """手动解析.env文件，不依赖python-dotenv（Mac沙箱环境可能未安装）"""
    if not env_path.exists():
        return
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)

try:
    from dotenv import load_dotenv
    if _env_path.exists():
        load_dotenv(_env_path)
    else:
        load_dotenv()  # fallback: 从当前目录查找
except ImportError:
    # python-dotenv 未安装时，手动解析.env
    _load_env_manually(_env_path)
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

# ===================== 配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

QWEN_OMNI_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

# Qwen-Omni 限制
MAX_AUDIO_SIZE_MB = 10.0  # base64 编码前
MAX_AUDIO_DURATION_SEC = 1200  # 20分钟
MP3_BITRATE = "320k"
SEGMENT_DURATION_SEC = 180  # 超长音频取前3分钟

# ===================== Prompt =====================
SYSTEM_PROMPT = """你是一个专业的音乐结构分析专家。请仔细听音频，输出结构化的音乐分析。

要求：
1. 输出严格的 JSON 格式，不要有额外文字或 markdown 标记
2. 段落时间戳要准确，单位为秒
3. 乐器要具体（如"中音萨克斯"、"原声钢琴"、"爵士鼓"）
4. 情绪要具体（如"忧郁沉思"、"热情奔放"、"宁静舒缓"）
5. caption 是一句话描述，包含流派、乐器、情绪、整体氛围

输出格式：
{
  "genre": "具体流派（如 latin jazz）",
  "subgenre": "子风格（如 bebop / cool jazz）",
  "vocal_presence": "instrumental 或 vocal",
  "tempo_bpm": 120,
  "key": "如 D minor",
  "segments": [
    {
      "start": 0.0,
      "end": 32.5,
      "label": "前奏",
      "instruments": ["钢琴", "贝斯"],
      "mood": "宁静",
      "confidence": 0.9
    }
  ],
  "instruments": ["钢琴", "贝斯", "鼓", "萨克斯"],
  "mood_tags": ["忧郁", "沉思"],
  "mood_vad": {"valence": 0.3, "arousal": 0.4, "dominance": 0.5},
  "caption": "这是一首...",
  "confidence": 0.85
}"""


# ===================== 音频预处理 =====================
def get_audio_duration(file_path: str) -> float:
    """获取音频时长（秒）"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", file_path],
            capture_output=True, text=True, timeout=10
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def get_file_size_mb(file_path: str) -> float:
    return os.path.getsize(file_path) / (1024 * 1024)


def convert_to_mp3(input_path: str, output_path: str, bitrate: str = MP3_BITRATE) -> bool:
    """转 MP3"""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-b:a", bitrate,
             "-ac", "2", "-ar", "44100", "-loglevel", "error", output_path],
            check=True, timeout=120
        )
        return os.path.exists(output_path)
    except subprocess.CalledProcessError as e:
        logger.warning(f"  转MP3失败: {e}")
        return False


def extract_segment(input_path: str, output_path: str,
                    start_sec: float = 0, duration_sec: float = SEGMENT_DURATION_SEC) -> bool:
    """提取音频片段"""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ss", str(start_sec),
             "-t", str(duration_sec), "-b:a", MP3_BITRATE, "-ac", "2",
             "-ar", "44100", "-loglevel", "error", output_path],
            check=True, timeout=60
        )
        return os.path.exists(output_path)
    except subprocess.CalledProcessError as e:
        logger.warning(f"  提取片段失败: {e}")
        return False


def prepare_audio_for_qwen(input_path: str, tmp_dir: Path) -> Tuple[str, bool]:
    """
    预处理音频为 Qwen-Omni 可用格式
    返回: (处理后的文件路径, 是否被截断)
    """
    size_mb = get_file_size_mb(input_path)
    duration = get_audio_duration(input_path)
    logger.info(f"  原始: {size_mb:.1f}MB, {duration:.0f}s")

    # 超长音频截断
    truncated = False
    work_path = input_path

    if duration > MAX_AUDIO_DURATION_SEC:
        logger.info(f"  时长 {duration:.0f}s > {MAX_AUDIO_DURATION_SEC}s，取前 {SEGMENT_DURATION_SEC}s")
        seg_path = str(tmp_dir / f"{Path(input_path).stem}_seg{SEGMENT_DURATION_SEC}.mp3")
        if extract_segment(input_path, seg_path, 0, SEGMENT_DURATION_SEC):
            work_path = seg_path
            truncated = True

    # 大文件转 MP3
    if get_file_size_mb(work_path) > MAX_AUDIO_SIZE_MB:
        logger.info(f"  文件 {get_file_size_mb(work_path):.1f}MB > {MAX_AUDIO_SIZE_MB}MB，转MP3")
        mp3_path = str(tmp_dir / f"{Path(work_path).stem}.mp3")
        if convert_to_mp3(work_path, mp3_path):
            work_path = mp3_path

    # 转 MP3 后仍太大，取片段
    if get_file_size_mb(work_path) > MAX_AUDIO_SIZE_MB:
        logger.info(f"  仍 {get_file_size_mb(work_path):.1f}MB，取前 {SEGMENT_DURATION_SEC}s")
        seg_path = str(tmp_dir / f"{Path(work_path).stem}_seg{SEGMENT_DURATION_SEC}.mp3")
        if extract_segment(work_path, seg_path, 0, SEGMENT_DURATION_SEC):
            work_path = seg_path
            truncated = True

    final_size = get_file_size_mb(work_path)
    final_dur = get_audio_duration(work_path)
    logger.info(f"  处理后: {final_size:.1f}MB, {final_dur:.0f}s, 截断={truncated}")
    return work_path, truncated


# ===================== API 调用 =====================
def call_qwen_omni(audio_path: str, api_key: str, model: str = "qwen3.5-omni-flash",
                    max_retries: int = 3) -> Optional[Dict]:
    """调用 Qwen-Omni 多模态 API"""
    # 读取音频并 base64 编码
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    # 检测音频格式
    ext = Path(audio_path).suffix.lower()
    mime_type = "audio/mpeg" if ext == ".mp3" else "audio/flac"
    audio_url = f"data:{mime_type};base64,{audio_b64}"

    payload = {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": [{"text": SYSTEM_PROMPT}]
                },
                {
                    "role": "user",
                    "content": [
                        {"audio": audio_url},
                        {"text": "请分析这段音频，输出结构化的JSON。"}
                    ]
                }
            ]
        },
        "parameters": {
            "max_tokens": 2000,
            "temperature": 0.3,
        }
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries):
        try:
            logger.info(f"  API调用 (尝试 {attempt+1}/{max_retries})...")
            resp = requests.post(QWEN_OMNI_ENDPOINT, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            # 解析输出
            if "output" in data and "choices" in data["output"]:
                content = data["output"]["choices"][0]["message"]["content"]
                # content 可能是列表或字符串
                if isinstance(content, list):
                    text = "".join(item.get("text", "") for item in content if isinstance(item, dict))
                else:
                    text = str(content)

                # 提取 JSON
                result = parse_json_from_text(text)
                if result:
                    result["_raw_text"] = text
                    result["_usage"] = data.get("usage", {})
                    return result
                else:
                    logger.warning(f"  无法解析JSON，原始文本前200字: {text[:200]}")
                    return {"_raw_text": text, "_usage": data.get("usage", {}), "parse_error": True}
            else:
                logger.error(f"  API返回格式异常: {json.dumps(data, ensure_ascii=False)[:300]}")
                return None

        except requests.exceptions.Timeout:
            logger.warning(f"  超时，重试...")
            time.sleep(5)
        except requests.exceptions.HTTPError as e:
            logger.error(f"  HTTP错误: {e}, 响应: {resp.text[:300]}")
            if resp.status_code == 429:
                time.sleep(10)
            else:
                return None
        except Exception as e:
            logger.error(f"  异常: {e}")
            time.sleep(3)

    return None


def parse_json_from_text(text: str) -> Optional[Dict]:
    """从文本中提取 JSON"""
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试提取 ```json ... ``` 块
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        try:
            return json.loads(text[start:end].strip())
        except json.JSONDecodeError:
            pass

    # 尝试提取第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass

    return None


# ===================== 主流程 =====================
def main():
    parser = argparse.ArgumentParser(description="Qwen-Omni 多模态黄金集 L3 结构标注")
    parser.add_argument("--golden-manifest", type=str, required=True)
    parser.add_argument("--master-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key（默认从 .env 的 DASHSCOPE_API_KEY 或 QWEN_OMNI_API_KEY 读取）")
    parser.add_argument("--model", type=str, default="qwen3.5-omni-flash")
    parser.add_argument("--audio-id", type=str, default=None, help="只处理指定 audio_id")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的输出")
    args = parser.parse_args()

    # 自动从环境变量读取 API key（支持 DASHSCOPE_API_KEY 和 QWEN_OMNI_API_KEY）
    if not args.api_key:
        args.api_key = os.environ.get("DASHSCOPE_API_KEY", "") or os.environ.get("QWEN_OMNI_API_KEY", "")

    if not args.api_key:
        logger.error("未提供 API key。请在 .env 中设置 DASHSCOPE_API_KEY，或用 --api-key 传入")
        sys.exit(1)

    # API key 预检：发一个最小请求验证 key 有效性，避免处理完所有音频才发现 key 无效
    logger.info("验证 API key 有效性...")
    try:
        import requests
        test_resp = requests.post(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
            headers={
                "Authorization": f"Bearer {args.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": args.model,
                "input": {"messages": [{"role": "user", "content": [{"text": "ping"}]}]},
            },
            timeout=10,
        )
        if test_resp.status_code == 401:
            logger.error("❌ API key 无效（401 Unauthorized）。请更新 .env 中的 DASHSCOPE_API_KEY")
            sys.exit(1)
        elif test_resp.status_code >= 400:
            logger.warning(f"⚠️ API 预检返回状态码 {test_resp.status_code}，继续尝试...")
        else:
            logger.info("✅ API key 验证通过")
    except requests.exceptions.RequestException as e:
        logger.warning(f"⚠️ API 预检网络错误: {e}，继续尝试（可能是网络波动）")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    # 加载黄金集 manifest
    import csv
    golden_samples = []
    with open(args.golden_manifest, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.audio_id and row["audio_id"] != args.audio_id:
                continue
            golden_samples.append(row)

    logger.info(f"黄金集: {len(golden_samples)} 首")

    # 处理每首
    results = []
    for idx, sample in enumerate(golden_samples):
        audio_id = sample["audio_id"]
        logger.info(f"\n{'='*60}")
        logger.info(f"[{idx+1}/{len(golden_samples)}] {audio_id}")

        # 检查输出
        out_file = output_dir / f"{audio_id}_l3_qwen.json"
        if out_file.exists() and not args.force:
            logger.info(f"  已存在，跳过（用 --force 覆盖）")
            continue

        # 查找母版文件
        master_dir = Path(args.master_dir)
        master_files = list(master_dir.rglob(f"*{audio_id}*"))
        if not master_files:
            logger.error(f"  未找到母版文件: {audio_id}")
            continue
        master_path = str(master_files[0])
        logger.info(f"  母版: {master_path}")

        # 预处理音频
        with tempfile.TemporaryDirectory(prefix="qwen_omni_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            prepared_audio, truncated = prepare_audio_for_qwen(master_path, tmp_path)

            # 调用 API
            result = call_qwen_omni(prepared_audio, args.api_key, args.model)

        if result is None:
            logger.error(f"  API调用失败")
            continue

        # 保存原始输出
        raw_file = raw_dir / f"{audio_id}_raw.json"
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # 构建标准输出
        output = {
            "audio_id": audio_id,
            "model": args.model,
            "truncated": truncated,
            "annotation": {
                "genre": result.get("genre", ""),
                "subgenre": result.get("subgenre", ""),
                "vocal_presence": result.get("vocal_presence", ""),
                "tempo_bpm": result.get("tempo_bpm"),
                "key": result.get("key", ""),
                "segments": result.get("segments", []),
                "instruments": result.get("instruments", []),
                "mood_tags": result.get("mood_tags", []),
                "mood_vad": result.get("mood_vad", {}),
                "caption": result.get("caption", ""),
                "confidence": result.get("confidence", 0.7),
            },
            "source": "qwen_omni",
            "usage": result.get("_usage", {}),
            "parse_error": result.get("parse_error", False),
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"  ✅ 完成: genre={output['annotation']['genre']}, "
                    f"instruments={len(output['annotation']['instruments'])}, "
                    f"segments={len(output['annotation']['segments'])}")
        results.append(output)

    # 汇总
    logger.info(f"\n{'='*60}")
    logger.info(f"完成: {len(results)}/{len(golden_samples)} 首")
    for r in results:
        logger.info(f"  {r['audio_id']}: {r['annotation']['genre']} | "
                    f"{len(r['annotation']['instruments'])}乐器 | "
                    f"{len(r['annotation']['segments'])}段落")


if __name__ == "__main__":
    import requests
    main()
