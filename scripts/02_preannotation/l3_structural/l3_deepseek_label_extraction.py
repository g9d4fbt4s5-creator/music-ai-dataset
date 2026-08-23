"""
l3_deepseek_label_extraction.py
L3 结构层：DeepSeek V4 Flash 文本标签提取 + 伪 Caption 生成

功能：
- 读取 L1 物理特征 + L2 语义特征（CLAP zero-shot）
- 调用 DeepSeek V4 Flash API 生成结构化标签
- 生成伪 Caption（自然语言描述）
- 支持并发、缓存、重试、速率限制
- 输出 JSONL 格式

用法：
    # 全量标签提取
    python l3_deepseek_label_extraction.py \
        --input-dir data/02_preannotation/l1_physical \
        --l2-dir data/02_preannotation/l2_semantic \
        --output data/02_preannotation/l3_structural/text_labels \
        --config configs/preannotation/preannotation_config.yaml

    # 只处理指定样本
    python l3_deepseek_label_extraction.py \
        --input-dir data/02_preannotation/l1_physical \
        --output data/02_preannotation/l3_structural/text_labels \
        --sample-ids data/02_preannotation/sample_ids.txt

    # 疑难样本纠错（V4 Pro）
    python l3_deepseek_label_extraction.py \
        --input-dir data/02_preannotation/l3_structural/text_labels \
        --output data/02_preannotation/l3_structural/corrected_labels \
        --mode correction \
        --correction-ratio 0.10
"""
import os
import sys
import json
import time
import yaml
import argparse
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ===================== 工具函数 =====================

def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_cache_key(features: Dict, model: str, prompt_version: str = "v1") -> str:
    """生成缓存 key（基于特征内容的哈希）"""
    content = json.dumps(features, sort_keys=True, ensure_ascii=False)
    raw = f"{model}:{prompt_version}:{content}"
    return hashlib.md5(raw.encode()).hexdigest()


def load_cached_result(cache_dir: Path, cache_key: str) -> Optional[Dict]:
    """从缓存加载结果"""
    cache_file = cache_dir / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"缓存读取失败 {cache_key}: {e}")
    return None


def save_cached_result(cache_dir: Path, cache_key: str, result: Dict):
    """保存结果到缓存"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# ===================== DeepSeek API 客户端 =====================

class DeepSeekClient:
    """DeepSeek API 客户端（含重试、速率限制）"""

    def __init__(self, config: Dict):
        self.api_key = os.environ.get(config.get("api_key_env", "DEEPSEEK_API_KEY"), "")
        self.base_url = config.get("base_url", "https://api.deepseek.com/v1")
        self.model = config.get("model", "deepseek-chat")
        self.max_tokens = config.get("cost_control", {}).get("max_tokens_per_request", 1000)

        # 重试配置
        retry_config = config.get("concurrency", {}).get("retry", {})
        self.max_retries = retry_config.get("max_retries", 3)
        self.initial_delay = retry_config.get("initial_delay", 1.0)
        self.max_delay = retry_config.get("max_delay", 10.0)
        self.backoff_factor = retry_config.get("backoff_factor", 2.0)

        # 速率限制
        self.rate_limit = config.get("concurrency", {}).get("rate_limit_per_minute", 60)
        self.min_interval = 60.0 / self.rate_limit
        self.last_request_time = 0

        if not self.api_key:
            logger.warning(f"未设置 API Key（环境变量 {config.get('api_key_env')}），将使用模拟模式")

    def _wait_for_rate_limit(self):
        """速率限制等待"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()

    def chat_completion(self, messages: List[Dict], temperature: float = 0.3) -> Optional[str]:
        """
        调用 DeepSeek Chat Completion API（含重试）

        Args:
            messages: 消息列表 [{"role": "system/user", "content": "..."}]
            temperature: 温度参数

        Returns:
            回复文本，失败返回 None
        """
        if not self.api_key:
            # 模拟模式：返回模拟结果
            return self._mock_response(messages)

        for attempt in range(self.max_retries):
            try:
                self._wait_for_rate_limit()

                import requests
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": self.max_tokens,
                        "stream": False,
                    },
                    timeout=30,
                )

                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                elif response.status_code == 429:
                    # 速率限制，等待后重试
                    wait_time = min(self.initial_delay * (self.backoff_factor ** attempt), self.max_delay)
                    logger.warning(f"速率限制 (429)，等待 {wait_time:.1f}s 后重试 ({attempt+1}/{self.max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"API 错误 {response.status_code}: {response.text}")
                    if attempt < self.max_retries - 1:
                        wait_time = min(self.initial_delay * (self.backoff_factor ** attempt), self.max_delay)
                        time.sleep(wait_time)

            except Exception as e:
                logger.error(f"请求异常 (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    wait_time = min(self.initial_delay * (self.backoff_factor ** attempt), self.max_delay)
                    time.sleep(wait_time)

        return None

    def _mock_response(self, messages: List[Dict]) -> str:
        """模拟模式：返回模拟结果（用于测试）"""
        logger.info("模拟模式：返回模拟标签结果")
        return json.dumps({
            "genre": ["jazz", "bebop"],
            "mood": ["melancholic", "introspective"],
            "instrumentation": ["saxophone", "piano", "double bass", "drums"],
            "vocal_presence": False,
            "tempo_category": "medium",
            "era": "1950s",
            "subgenre": "hard bop",
            "caption": "A melancholic bebop jazz piece featuring saxophone and piano, with a medium tempo and introspective mood.",
            "confidence": 0.75,
        }, ensure_ascii=False)


# ===================== Prompt 模板 =====================

SYSTEM_PROMPT = """你是一个专业的音乐标签分析专家。根据提供的音频特征信息，生成结构化的音乐标签和自然语言描述。

要求：
1. 只输出 JSON 格式，不要输出其他内容
2. 标签要准确、具体，避免泛泛而谈
3. 基于提供的特征推断，不要编造无法从特征推断的信息
4. 置信度要反映你对标签的确定程度

输出 JSON 格式：
{
  "genre": ["流派1", "流派2"],
  "mood": ["情绪1", "情绪2"],
  "instrumentation": ["乐器1", "乐器2"],
  "vocal_presence": true/false,
  "tempo_category": "slow/medium/fast",
  "era": "年代",
  "subgenre": "子流派",
  "caption": "自然语言描述（100-200字）",
  "confidence": 0.0-1.0
}"""


def build_user_prompt(features: Dict) -> str:
    """构建用户 prompt（基于特征 JSON）"""
    features_str = json.dumps(features, ensure_ascii=False, indent=2)
    return f"""请根据以下音频特征信息，生成结构化的音乐标签：

音频特征：
{features_str}

请输出 JSON 格式的标签结果。"""


def build_correction_prompt(initial_labels: Dict, features: Dict) -> str:
    """构建纠错 prompt（用于 V4 Pro 疑难样本纠错）"""
    initial_str = json.dumps(initial_labels, ensure_ascii=False, indent=2)
    features_str = json.dumps(features, ensure_ascii=False, indent=2)
    return f"""请审核并修正以下音乐标签。初始标签可能存在错误或不准确之处，请根据音频特征进行修正。

初始标签：
{initial_str}

音频特征：
{features_str}

请输出修正后的 JSON 格式标签结果。如果初始标签正确，保持不变；如果有错误，请修正。"""


# ===================== 标签提取 =====================

def extract_labels(client: DeepSeekClient, features: Dict,
                   cache_dir: Path, mode: str = "extract") -> Optional[Dict]:
    """
    提取标签（含缓存）

    Args:
        client: DeepSeek 客户端
        features: 音频特征
        cache_dir: 缓存目录
        mode: 模式（extract/correction）

    Returns:
        标签结果，失败返回 None
    """
    # 生成缓存 key
    cache_key = get_cache_key(features, client.model, mode)

    # 检查缓存
    cached = load_cached_result(cache_dir, cache_key)
    if cached:
        logger.debug(f"缓存命中: {cache_key[:8]}")
        return cached

    # 构建 prompt
    if mode == "correction":
        # 纠错模式：features 包含 initial_labels + 原始特征
        initial_labels = features.get("initial_labels", {})
        original_features = features.get("features", {})
        user_prompt = build_correction_prompt(initial_labels, original_features)
    else:
        user_prompt = build_user_prompt(features)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # 调用 API
    response = client.chat_completion(messages)

    if response is None:
        logger.error("API 调用失败")
        return None

    # 解析 JSON
    try:
        # 清理响应（可能包含 markdown 代码块）
        response_clean = response.strip()
        if response_clean.startswith("```json"):
            response_clean = response_clean[7:]
        if response_clean.endswith("```"):
            response_clean = response_clean[:-3]
        response_clean = response_clean.strip()

        result = json.loads(response_clean)
        result["_raw_response"] = response
        result["_cache_key"] = cache_key
        result["_timestamp"] = datetime.now().isoformat()

        # 保存缓存
        save_cached_result(cache_dir, cache_key, result)

        return result

    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失败: {e}")
        logger.error(f"原始响应: {response[:500]}")
        return None


def process_sample(sample_id: str, features: Dict, client: DeepSeekClient,
                   cache_dir: Path, output_dir: Path, mode: str = "extract") -> Tuple[str, Optional[Dict]]:
    """
    处理单个样本

    Args:
        sample_id: 样本 ID
        features: 音频特征
        client: DeepSeek 客户端
        cache_dir: 缓存目录
        output_dir: 输出目录
        mode: 模式

    Returns:
        (sample_id, result)
    """
    logger.info(f"处理样本: {sample_id}")

    result = extract_labels(client, features, cache_dir, mode)

    if result:
        # 保存单个结果
        output_file = output_dir / f"{sample_id}.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"sample_id": sample_id, "labels": result}, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ {sample_id}: genre={result.get('genre', [])}, confidence={result.get('confidence', 0):.2f}")
    else:
        logger.error(f"❌ {sample_id}: 处理失败")

    return sample_id, result


# ===================== 主流程 =====================

def main():
    parser = argparse.ArgumentParser(
        description="L3 结构层：DeepSeek V4 Flash 文本标签提取",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="输入特征目录（L1 物理特征）")
    parser.add_argument("--l2-dir", type=str, default=None,
                        help="L2 语义特征目录（CLAP zero-shot）")
    parser.add_argument("--output", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--config", type=str,
                        default="configs/preannotation/preannotation_config.yaml",
                        help="配置文件路径")
    parser.add_argument("--mode", type=str, default="extract",
                        choices=["extract", "correction"],
                        help="模式：extract=标签提取，correction=疑难纠错")
    parser.add_argument("--sample-ids", type=str, default=None,
                        help="指定样本 ID 列表文件（每行一个）")
    parser.add_argument("--correction-ratio", type=float, default=0.10,
                        help="纠错模式抽样比例（默认10%）")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制处理数量")
    parser.add_argument("--dry-run", action="store_true",
                        help="试运行（不调用API，只统计）")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    l3_config = config["l3_structural"]["text_label_extraction"] if args.mode == "extract" \
        else config["l3_structural"]["error_correction"]

    # 初始化客户端
    client = DeepSeekClient(l3_config)

    # 目录
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)
    cache_dir = PROJECT_ROOT / l3_config.get("cache", {}).get("cache_dir", "data/02_preannotation/api_cache/deepseek")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载样本
    sample_ids = []
    if args.sample_ids:
        with open(args.sample_ids, "r", encoding="utf-8") as f:
            sample_ids = [line.strip() for line in f if line.strip()]
    else:
        # 从输入目录扫描
        feature_files = list(input_dir.glob("*.json"))
        sample_ids = [f.stem for f in feature_files]

    if args.limit:
        sample_ids = sample_ids[:args.limit]

    logger.info(f"待处理样本数: {len(sample_ids)}")
    logger.info(f"模式: {args.mode}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"缓存目录: {cache_dir}")

    if args.dry_run:
        logger.info("试运行模式，不调用 API")
        for sid in sample_ids:
            logger.info(f"  - {sid}")
        return

    # 加载特征
    def load_features(sample_id: str) -> Optional[Dict]:
        """加载样本特征（L1 + L2）"""
        features = {}

        # L1 特征
        l1_file = input_dir / f"{sample_id}.json"
        if l1_file.exists():
            with open(l1_file, "r", encoding="utf-8") as f:
                features["l1_physical"] = json.load(f)

        # L2 特征
        if args.l2_dir:
            l2_file = Path(args.l2_dir) / f"{sample_id}.json"
            if l2_file.exists():
                with open(l2_file, "r", encoding="utf-8") as f:
                    features["l2_semantic"] = json.load(f)

        return features if features else None

    # 并发处理
    max_workers = l3_config.get("concurrency", {}).get("max_workers", 10)
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for sample_id in sample_ids:
            features = load_features(sample_id)
            if features is None:
                logger.warning(f"跳过 {sample_id}：未找到特征文件")
                continue

            future = executor.submit(
                process_sample, sample_id, features, client, cache_dir, output_dir, args.mode
            )
            futures[future] = sample_id

        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                sid, result = future.result()
                results[sid] = result
            except Exception as e:
                logger.error(f"样本 {sample_id} 处理异常: {e}")

    # 汇总
    success_count = sum(1 for r in results.values() if r is not None)
    fail_count = len(results) - success_count

    logger.info("\n" + "=" * 60)
    logger.info("处理完成")
    logger.info("=" * 60)
    logger.info(f"  总样本数: {len(sample_ids)}")
    logger.info(f"  成功: {success_count}")
    logger.info(f"  失败: {fail_count}")
    logger.info(f"  成功率: {success_count/len(sample_ids)*100:.1f}%" if sample_ids else "  成功率: N/A")
    logger.info(f"  输出目录: {output_dir}")
    logger.info("=" * 60)

    # 保存汇总
    summary = {
        "total": len(sample_ids),
        "success": success_count,
        "fail": fail_count,
        "mode": args.mode,
        "model": client.model,
        "timestamp": datetime.now().isoformat(),
        "sample_ids": sample_ids,
    }
    with open(output_dir / "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
