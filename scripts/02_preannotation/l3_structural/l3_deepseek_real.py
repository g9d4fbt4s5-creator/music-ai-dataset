"""
l3_deepseek_real.py
L3 结构标注层：用 DeepSeek API 基于 L1+L2 特征生成结构化标签

功能：
- 读取 L1 物理特征 + L2 语义候选
- 调用 DeepSeek V4 Flash 生成结构化标签（流派/情绪/乐器/场景/Caption）
- 支持并发、缓存、重试
- 输出：每个音频一个 JSON

用法：
    python l3_deepseek_real.py \
        --l1-dir data/02_preannotation/l1_physical \
        --l2-semantic-dir data/02_preannotation/l2_semantic \
        --output data/02_preannotation/l3_structural/text_labels \
        --api-key sk-xxx \
        --concurrency 5
"""
import os
import sys
import json
import time
import argparse
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================== 配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"  # DeepSeek V4 Flash

# ===================== Prompt 模板 =====================

SYSTEM_PROMPT = """你是一个专业的音乐标注专家。根据提供的音频物理特征和语义候选，生成结构化的音乐标签。

要求：
1. 输出严格的 JSON 格式，不要有额外文字
2. 标签要准确、具体，避免泛泛而谈
3. 流派要具体到子风格（如 latin jazz、bebop、cool jazz）
4. 情绪要具体（如 melancholic、energetic、introspective）
5. 乐器要列出实际听到的乐器
6. Caption 是一句话描述，包含流派、乐器、情绪、速度感

输出格式：
{
  "genre": "具体流派",
  "genre_candidates": ["候选1", "候选2", "候选3"],
  "mood": ["情绪1", "情绪2"],
  "instrumentation": ["乐器1", "乐器2", "乐器3"],
  "scene": "适用场景",
  "vocal_presence": "instrumental 或 vocal",
  "tempo_description": "慢速/中速/快速",
  "caption": "一句话描述"
}
"""


def build_user_prompt(l1_features: Dict, l2_semantic: Dict) -> str:
    """构建用户 prompt"""
    # L1 特征
    bpm = l1_features.get("bpm", "unknown")
    key = l1_features.get("key", "unknown")
    lufs = l1_features.get("lufs", "unknown")
    duration = l1_features.get("duration_sec", "unknown")
    spectral_centroid = l1_features.get("spectral_centroid_mean", "unknown")
    zcr = l1_features.get("zero_crossing_rate_mean", "unknown")

    # L2 语义候选
    genre_top3 = [g["label"] for g in l2_semantic.get("genre", [])[:3]]
    mood_top3 = [m["label"] for m in l2_semantic.get("mood", [])[:3]]
    instrument_top5 = [i["label"] for i in l2_semantic.get("instrumentation", [])[:5]]
    scene_top3 = [s["label"] for s in l2_semantic.get("scene", [])[:3]]

    prompt = f"""请根据以下音频特征生成结构化标签：

【物理特征】
- BPM: {bpm}
- Key: {key}
- LUFS (响度): {lufs}
- 时长: {duration}秒
- 频谱质心: {spectral_centroid} Hz
- 过零率: {zcr}

【CLAP zero-shot 候选标签】
- 流派候选: {genre_top3}
- 情绪候选: {mood_top3}
- 乐器候选: {instrument_top5}
- 场景候选: {scene_top3}

请综合以上信息，输出准确的结构化标签 JSON。
"""
    return prompt


# ===================== API 调用 =====================

def call_deepseek(api_key: str, prompt: str, max_retries: int = 3) -> Optional[Dict]:
    """
    调用 DeepSeek API

    Args:
        api_key: API key
        prompt: 用户 prompt
        max_retries: 最大重试次数

    Returns:
        解析后的 JSON 标签，失败返回 None
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 800,
        "temperature": 0.3,
        "response_format": {"type": "json_object"}
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(DEEPSEEK_BASE_URL, headers=headers, json=payload, timeout=30)

            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                # 解析 JSON
                try:
                    labels = json.loads(content)
                    return labels
                except json.JSONDecodeError:
                    # 尝试提取 JSON 部分
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    if start >= 0 and end > start:
                        return json.loads(content[start:end])
                    logger.warning(f"JSON 解析失败: {content[:100]}")
                    return None
            elif resp.status_code == 429:
                # 速率限制，退避
                wait_time = 2 ** attempt
                logger.warning(f"速率限制，等待 {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.warning(f"API 错误 {resp.status_code}: {resp.text[:100]}")
                time.sleep(1)

        except requests.exceptions.Timeout:
            logger.warning(f"超时 (尝试 {attempt+1}/{max_retries})")
            time.sleep(2)
        except Exception as e:
            logger.warning(f"异常: {e}")
            time.sleep(1)

    return None


# ===================== 数据加载 =====================

def load_l1_features(l1_dir: Path) -> Dict[str, Dict]:
    """加载 L1 物理特征"""
    features = {}
    for f in l1_dir.glob("*_physical.json"):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        raw_id = data.get("audio_id", "")
        # 提取纯 audio_id
        if "_" in raw_id:
            parts = raw_id.split("_")
            for part in reversed(parts):
                if len(part) == 26:
                    raw_id = part
                    break
        data["audio_id"] = raw_id
        features[raw_id] = data
    return features


def load_l2_semantic(l2_dir: Path) -> Dict[str, Dict]:
    """加载 L2 语义候选"""
    semantics = {}
    for f in l2_dir.glob("*_semantic.json"):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        raw_id = data.get("audio_id", f.stem.replace("_semantic", ""))
        # 提取纯 audio_id（去掉 hash32_ 前缀）
        if "_" in raw_id:
            parts = raw_id.split("_")
            for part in reversed(parts):
                if len(part) == 26:
                    raw_id = part
                    break
        data["audio_id"] = raw_id
        semantics[raw_id] = data
    return semantics


# ===================== 主流程 =====================

def main():
    parser = argparse.ArgumentParser(
        description="L3 结构标注层：DeepSeek API 生成结构化标签",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--l1-dir", type=str, default="data/02_preannotation/l1_physical",
                        help="L1 物理特征目录")
    parser.add_argument("--l2-semantic-dir", type=str, default="data/02_preannotation/l2_semantic",
                        help="L2 语义候选目录")
    parser.add_argument("--output", type=str, default="data/02_preannotation/l3_structural/text_labels",
                        help="输出目录")
    parser.add_argument("--api-key", type=str, default=None,
                        help="DeepSeek API key（默认从环境变量 DEEPSEEK_API_KEY 读取）")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="并发数（默认 5）")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制处理数量")
    parser.add_argument("--skip-existing", action="store_true",
                        help="跳过已存在的输出文件")
    args = parser.parse_args()

    # 获取 API key
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("未提供 API key，请使用 --api-key 或设置 DEEPSEEK_API_KEY 环境变量")
        return

    l1_dir = Path(args.l1_dir)
    l2_dir = Path(args.l2_semantic_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    logger.info("加载 L1 物理特征...")
    l1_features = load_l1_features(l1_dir)
    logger.info(f"  L1: {len(l1_features)} 首")

    logger.info("加载 L2 语义候选...")
    l2_semantics = load_l2_semantic(l2_dir)
    logger.info(f"  L2: {len(l2_semantics)} 首")

    # 取交集
    common_ids = set(l1_features.keys()) & set(l2_semantics.keys())
    logger.info(f"共同样本: {len(common_ids)} 首")

    if args.limit:
        common_ids = list(common_ids)[:args.limit]

    # 跳过已存在
    if args.skip_existing:
        existing = {f.stem.replace("_text_labels", "") for f in output_dir.glob("*_text_labels.json")}
        to_process = [aid for aid in common_ids if aid not in existing]
        logger.info(f"跳过已存在: {len(common_ids) - len(to_process)} 首，待处理: {len(to_process)} 首")
    else:
        to_process = list(common_ids)

    if not to_process:
        logger.info("没有需要处理的样本")
        return

    # 并发调用
    logger.info(f"\n开始调用 DeepSeek API（并发={args.concurrency}）...")
    results = {}
    failed = []

    def process_one(audio_id: str) -> tuple:
        l1 = l1_features[audio_id]
        l2 = l2_semantics[audio_id]
        prompt = build_user_prompt(l1, l2)
        labels = call_deepseek(api_key, prompt)
        return audio_id, labels

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(process_one, aid): aid for aid in to_process}

        for i, future in enumerate(as_completed(futures), 1):
            audio_id = futures[future]
            try:
                aid, labels = future.result()
                if labels:
                    labels["audio_id"] = aid
                    labels["source"] = "deepseek_v4_flash"
                    results[aid] = labels

                    # 保存单个文件
                    with open(output_dir / f"{aid}_text_labels.json", "w", encoding="utf-8") as f:
                        json.dump(labels, f, ensure_ascii=False, indent=2)

                    logger.info(f"[{i}/{len(to_process)}] ✅ {aid}: {labels.get('genre', 'N/A')}")
                else:
                    failed.append(audio_id)
                    logger.info(f"[{i}/{len(to_process)}] ❌ {aid}: API 调用失败")
            except Exception as e:
                failed.append(audio_id)
                logger.error(f"[{i}/{len(to_process)}] ❌ {audio_id}: {e}")

    # 汇总
    logger.info("\n" + "=" * 60)
    logger.info("L3 DeepSeek 标签提取完成")
    logger.info("=" * 60)
    logger.info(f"  总数: {len(to_process)}")
    logger.info(f"  成功: {len(results)}")
    logger.info(f"  失败: {len(failed)}")
    logger.info(f"  输出目录: {output_dir}")

    if failed:
        logger.info(f"  失败列表: {failed}")

    # 流派分布
    genre_dist = {}
    for labels in results.values():
        genre = labels.get("genre", "unknown")
        genre_dist[genre] = genre_dist.get(genre, 0) + 1

    logger.info("\n  流派分布:")
    for genre, count in sorted(genre_dist.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"    {genre}: {count}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
