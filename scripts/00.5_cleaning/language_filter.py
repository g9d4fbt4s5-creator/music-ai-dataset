"""
language_filter.py
语言过滤模块（Stage 5.1）

使用 Whisper base 模型的 detect_language() 检测音频语言，
只加载前 30 秒，轻量级，Mac CPU 可跑。

功能：
- 检测音频语言（zh/en/ja/ko/fr/de/es/it/pt/ru 等 99 种）
- 输出语言标签和置信度
- 非目标语言标记为 lang_filtered
- 结果缓存，避免重复检测

用法：
    from language_filter import LanguageFilter
    filter = LanguageFilter(allowed_languages=["zh", "en", "ja"])
    result = filter.detect(audio_path)
    if result.is_allowed:
        # 目标语言
    else:
        # 非目标语言，标记过滤
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field

import whisper
import torch

logger = logging.getLogger(__name__)


@dataclass
class LanguageDetectionResult:
    """语言检测结果"""
    audio_path: str
    language: str = ""  # 如 'zh', 'en', 'ja'
    confidence: float = 0.0  # 0.0-1.0
    is_allowed: bool = True  # 是否在允许的语言列表中
    top5_languages: List[Tuple[str, float]] = field(default_factory=list)  # top5 候选
    error: str = ""

    def to_dict(self) -> Dict:
        return {
            "audio_path": self.audio_path,
            "lang": self.language,
            "lang_confidence": round(self.confidence, 4),
            "lang_allowed": self.is_allowed,
            "lang_top5": "; ".join(f"{lang}:{prob:.3f}" for lang, prob in self.top5_languages),
            "lang_error": self.error,
        }


class LanguageFilter:
    """语言过滤器"""

    # Whisper 支持的常见语言（完整列表 99 种，这里列常用的）
    COMMON_LANGUAGES = {
        "zh": "中文",
        "en": "英语",
        "ja": "日语",
        "ko": "韩语",
        "fr": "法语",
        "de": "德语",
        "es": "西班牙语",
        "it": "意大利语",
        "pt": "葡萄牙语",
        "ru": "俄语",
        "ar": "阿拉伯语",
        "hi": "印地语",
        "th": "泰语",
        "vi": "越南语",
        "id": "印尼语",
        "ms": "马来语",
        "nl": "荷兰语",
        "pl": "波兰语",
        "tr": "土耳其语",
        "sv": "瑞典语",
    }

    def __init__(
        self,
        model_size: str = "base",
        allowed_languages: Optional[List[str]] = None,
        min_confidence: float = 0.5,
        max_seconds: int = 30,
        cache_csv: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        初始化语言过滤器

        Args:
            model_size: Whisper 模型大小（tiny/base/small/medium/large），语言检测用 base 足够
            allowed_languages: 允许的语言列表，None 表示不过滤
            min_confidence: 最低置信度，低于此值标记为不确定
            max_seconds: 最多加载多少秒音频用于检测（默认 30 秒，足够）
            cache_csv: 缓存文件路径，避免重复检测
            device: 运行设备（cpu/cuda）
        """
        self.model_size = model_size
        self.allowed_languages = allowed_languages
        self.min_confidence = min_confidence
        self.max_seconds = max_seconds
        self.cache_csv = cache_csv
        self.device = device

        # 加载模型
        logger.info(f"正在加载 Whisper {model_size} 模型（device={device}）...")
        self.model = whisper.load_model(model_size, device=device)
        logger.info(f"Whisper {model_size} 模型加载完成")

        # 加载缓存
        self.cache: Dict[str, LanguageDetectionResult] = {}
        if cache_csv and os.path.exists(cache_csv):
            self._load_cache(cache_csv)
            logger.info(f"加载语言检测缓存: {len(self.cache)} 条")

        logger.info("语言过滤器初始化完成")
        logger.info(f"  允许语言: {allowed_languages if allowed_languages else '不过滤'}")
        logger.info(f"  最低置信度: {min_confidence}")
        logger.info(f"  最大检测时长: {max_seconds}秒")

    def _load_cache(self, cache_csv: str):
        """加载缓存"""
        try:
            df = pd.read_csv(cache_csv)
            for _, row in df.iterrows():
                result = LanguageDetectionResult(
                    audio_path=row.get("audio_path", ""),
                    language=row.get("lang", ""),
                    confidence=float(row.get("lang_confidence", 0)),
                    is_allowed=bool(row.get("lang_allowed", True)),
                    error=row.get("lang_error", ""),
                )
                if result.audio_path:
                    self.cache[result.audio_path] = result
        except Exception as e:
            logger.warning(f"加载缓存失败: {e}")

    def _save_cache(self):
        """保存缓存"""
        if not self.cache_csv:
            return
        try:
            os.makedirs(os.path.dirname(self.cache_csv), exist_ok=True)
            data = [r.to_dict() for r in self.cache.values()]
            pd.DataFrame(data).to_csv(self.cache_csv, index=False, encoding="utf-8")
            logger.info(f"语言检测缓存已保存: {len(self.cache)} 条 -> {self.cache_csv}")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")

    def detect(self, audio_path: str, force: bool = False) -> LanguageDetectionResult:
        """
        检测音频语言

        Args:
            audio_path: 音频文件路径
            force: 是否强制重新检测（忽略缓存）

        Returns:
            LanguageDetectionResult: 检测结果
        """
        result = LanguageDetectionResult(audio_path=audio_path)

        # 检查缓存
        if not force and audio_path in self.cache:
            cached = self.cache[audio_path]
            # 更新 allowed 状态（允许列表可能变了）
            if self.allowed_languages:
                cached.is_allowed = cached.language in self.allowed_languages
            return cached

        try:
            # 加载音频（只加载前 max_seconds 秒）
            audio = whisper.load_audio(audio_path)

            # 检查音频长度（太短会导致 mel shape 错误）
            if len(audio) < 16000:  # 小于 1 秒
                result.error = "too_short"
                result.is_allowed = False
                logger.warning(f"  音频过短（{len(audio)/16000:.2f}秒），跳过语言检测: {audio_path}")
                self.cache[audio_path] = result
                self._save_cache()
                return result

            if len(audio) > self.max_seconds * 16000:
                audio = audio[:self.max_seconds * 16000]

            # 计算 log mel spectrogram
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

            # 检测语言
            _, probs = self.model.detect_language(mel)

            # 获取 top5
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            top5 = sorted_probs[:5]
            result.top5_languages = top5

            # 最高概率语言
            lang, prob = top5[0]
            result.language = lang
            result.confidence = float(prob)

            # 判断是否允许
            if self.allowed_languages:
                result.is_allowed = lang in self.allowed_languages
            else:
                result.is_allowed = True

            # 低置信度警告
            if result.confidence < self.min_confidence:
                result.error = f"low_confidence:{result.confidence:.3f}"
                logger.warning(f"  低置信度: {audio_path} -> {lang} ({result.confidence:.3f})")

            lang_name = self.COMMON_LANGUAGES.get(lang, lang)
            logger.info(f"  语言检测: {audio_path} -> {lang}({lang_name}) "
                       f"置信度={result.confidence:.3f} "
                       f"{'✅允许' if result.is_allowed else '❌过滤'}")

        except Exception as e:
            result.error = str(e)
            result.is_allowed = False
            logger.warning(f"  语言检测失败: {audio_path} -> {e}")

        # 写入缓存
        self.cache[audio_path] = result
        self._save_cache()

        return result

    def filter_dataframe(
        self,
        df: pd.DataFrame,
        audio_path_col: str = "audio_path",
        add_columns: bool = True,
        filter_not_allowed: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        批量检测 DataFrame 中的音频语言

        Args:
            df: 输入 DataFrame
            audio_path_col: 音频路径列名
            add_columns: 是否添加 lang/lang_confidence/lang_allowed 列
            filter_not_allowed: 是否过滤掉不允许的语言（默认只标记不删除）

        Returns:
            (filtered_df, report_df)
        """
        report_rows = []
        langs = []
        confidences = []
        allowed_list = []

        for idx, row in df.iterrows():
            audio_path = row.get(audio_path_col, "")
            if not audio_path or not os.path.exists(audio_path):
                result = LanguageDetectionResult(
                    audio_path=audio_path,
                    error="file_not_found",
                    is_allowed=False,
                )
            else:
                result = self.detect(audio_path)

            langs.append(result.language)
            confidences.append(result.confidence)
            allowed_list.append(result.is_allowed)
            report_rows.append(result.to_dict())

        report_df = pd.DataFrame(report_rows)

        if add_columns:
            df = df.copy()
            df["lang"] = langs
            df["lang_confidence"] = confidences
            df["lang_allowed"] = allowed_list

        if filter_not_allowed:
            before = len(df)
            df = df[df["lang_allowed"] == True].reset_index(drop=True)
            logger.info(f"语言过滤: {before} → {len(df)} (剔除 {before - len(df)})")

        # 统计
        if len(langs) > 0:
            lang_counts = pd.Series(langs).value_counts()
            logger.info(f"语言分布: {dict(lang_counts)}")
            allowed_count = sum(allowed_list)
            logger.info(f"允许语言: {allowed_count}/{len(langs)}")

        return df, report_df


def batch_language_detection(
    audio_paths: List[str],
    allowed_languages: Optional[List[str]] = None,
    model_size: str = "base",
    cache_csv: Optional[str] = None,
    report_csv: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[List[LanguageDetectionResult], pd.DataFrame]:
    """
    批量语言检测（便捷函数）

    Args:
        audio_paths: 音频文件路径列表
        allowed_languages: 允许的语言列表
        model_size: Whisper 模型大小
        cache_csv: 缓存文件路径
        report_csv: 报告输出路径
        device: 运行设备

    Returns:
        (results, report_df)
    """
    lang_filter = LanguageFilter(
        model_size=model_size,
        allowed_languages=allowed_languages,
        cache_csv=cache_csv,
        device=device,
    )

    results = []
    for audio_path in audio_paths:
        result = lang_filter.detect(audio_path)
        results.append(result)

    report_df = pd.DataFrame([r.to_dict() for r in results])

    if report_csv:
        os.makedirs(os.path.dirname(report_csv), exist_ok=True)
        report_df.to_csv(report_csv, index=False, encoding="utf-8")
        logger.info(f"语言检测报告已保存: {report_csv}")

    return results, report_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 5.1 语言过滤（Whisper base）")
    parser.add_argument("--file-list", type=str, help="音频文件列表路径")
    parser.add_argument("--input-dir", type=str, help="输入音频目录")
    parser.add_argument("--yamnet-results", type=str, default=None,
                        help="YAMNet结果CSV路径（用于--vocals-only过滤）")
    parser.add_argument("--vocals-only", action="store_true",
                        help="只对 has_vocals=True 的样本跑语言过滤（纯器乐跳过，节省时间）")
    parser.add_argument("--output", type=str, required=True, help="输出CSV路径")
    parser.add_argument("--model", type=str, default="base", help="Whisper模型大小")
    parser.add_argument("--allowed-languages", type=str, default=None,
                        help="允许的语言（逗号分隔，如 zh,en,ja）")
    parser.add_argument("--limit", type=int, default=None, help="只处理前N个")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # 收集音频文件
    audio_files = []
    if args.file_list:
        with open(args.file_list) as f:
            audio_files = [line.strip() for line in f if line.strip()]
    elif args.input_dir:
        audio_extensions = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
        for root, dirs, files in os.walk(args.input_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in audio_extensions:
                    audio_files.append(os.path.join(root, f))
        audio_files = sorted(audio_files)

    if args.limit:
        audio_files = audio_files[:args.limit]
        logger.info(f"限制前 {args.limit} 个音频")

    # YAMNet过滤：只对 has_vocals=True 的样本跑语言过滤
    if args.vocals_only and args.yamnet_results:
        yamnet_df = pd.read_csv(args.yamnet_results)
        vocals_track_ids = set(yamnet_df[yamnet_df["has_vocals"] == True]["track_id"])
        original_count = len(audio_files)
        audio_files = [f for f in audio_files if Path(f).stem in vocals_track_ids]
        logger.info(f"YAMNet过滤：{original_count} → {len(audio_files)}（只处理 has_vocals=True）")
        if not audio_files:
            logger.warning("没有 has_vocals=True 的样本，跳过语言过滤")
            # 输出空结果
            pd.DataFrame(columns=["track_id", "lang", "lang_confidence", "lang_allowed", "lang_top5", "lang_error"]).to_csv(args.output, index=False)
            sys.exit(0)
    elif args.vocals_only and not args.yamnet_results:
        logger.warning("--vocals-only 需要 --yamnet-results，忽略过滤")

    logger.info(f"开始语言检测：{len(audio_files)} 首，模型={args.model}")

    allowed_languages = args.allowed_languages.split(",") if args.allowed_languages else None

    # 批量检测
    lang_filter = LanguageFilter(
        model_size=args.model,
        allowed_languages=allowed_languages,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    results = []
    for i, audio_path in enumerate(audio_files):
        track_id = Path(audio_path).stem
        logger.info(f"[{i+1}/{len(audio_files)}] {track_id}")
        result = lang_filter.detect(audio_path)
        result_dict = result.to_dict()
        result_dict["track_id"] = track_id
        result_dict["audio_path"] = audio_path
        results.append(result_dict)

    # 保存
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")
    logger.info(f"语言检测完成：{len(results)} 首，结果已保存: {args.output}")

    # 统计
    if "lang" in df.columns:
        logger.info(f"语言分布:\n{df['lang'].value_counts()}")
        if "lang_confidence" in df.columns:
            logger.info(f"置信度均值: {df['lang_confidence'].mean():.2%}")
