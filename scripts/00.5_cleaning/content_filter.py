"""
content_filter.py
内容过滤模块（Stage 3 子功能）

功能：
1. 非音乐内容过滤：检测纯语音、环境音、广告片段等非音乐内容
   - 基础版：librosa 简单特征（音乐性评分）
   - 高级版：YAMNet / PANNs 音频事件检测模型（需额外安装）

2. 人声成分检测：检测人声占比，视任务决定是否保留纯器乐
   - 基础版：librosa 谐波/打击乐分离估算人声占比
   - 高级版：demucs 人声分离后精确计算能量占比（demucs 已安装）

3. 内容安全过滤：检测敏感言论、违法内容
   - 需要 ASR（Whisper）+ 关键词检测，默认关闭

用法：
    from content_filter import ContentFilter
    filter = ContentFilter(config)
    result = filter.analyze(audio_path)
    if result.is_music:
        # 是音乐
    if result.vocal_ratio > 0.5:
        # 人声占比高
"""
import os
import sys
import re
import logging
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent

logger = logging.getLogger(__name__)


@dataclass
class ContentAnalysisResult:
    """内容分析结果"""
    audio_path: str

    # 音乐性
    is_music: bool = True
    music_score: float = 0.0  # 0.0-1.0，越高越像音乐
    music_confidence: float = 0.0

    # 人声
    vocal_ratio: float = 0.0  # 0.0-1.0，人声能量占比
    is_instrumental: bool = False  # 是否纯器乐
    vocal_confidence: float = 0.0

    # 内容安全
    is_safe: bool = True
    safety_warnings: List[str] = field(default_factory=list)

    # 详细特征
    tempo: float = 0.0
    spectral_centroid: float = 0.0
    zero_crossing_rate: float = 0.0
    harmonic_energy_ratio: float = 0.0
    percussive_energy_ratio: float = 0.0
    rhythm_regularity: float = 0.0

    # 使用的方法
    method_used: str = "librosa_basic"

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "audio_path": self.audio_path,
            "is_music": self.is_music,
            "music_score": round(self.music_score, 4),
            "music_confidence": round(self.music_confidence, 4),
            "vocal_ratio": round(self.vocal_ratio, 4),
            "is_instrumental": self.is_instrumental,
            "vocal_confidence": round(self.vocal_confidence, 4),
            "is_safe": self.is_safe,
            "safety_warnings": "; ".join(self.safety_warnings),
            "tempo": round(self.tempo, 2),
            "spectral_centroid": round(self.spectral_centroid, 2),
            "zero_crossing_rate": round(self.zero_crossing_rate, 4),
            "harmonic_energy_ratio": round(self.harmonic_energy_ratio, 4),
            "percussive_energy_ratio": round(self.percussive_energy_ratio, 4),
            "rhythm_regularity": round(self.rhythm_regularity, 4),
            "method_used": self.method_used,
        }


class ContentFilter:
    """内容过滤器"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化内容过滤器

        Args:
            config: 内容过滤配置（来自 cleaning_config.yaml 的 stage3_quality.content_filter）
        """
        if config is None:
            config = {}

        # 非音乐过滤
        self.enable_non_music_filter = config.get("filter_non_music", False)
        self.min_music_score = config.get("min_music_score", 0.3)

        # 人声过滤
        self.max_vocal_ratio = config.get("max_vocal_ratio", 1.0)  # 1.0 = 不限制
        self.instrumental_only = config.get("instrumental_only", False)

        # 内容安全
        self.enable_safety_filter = config.get("enable_safety_filter", False)
        self.sensitive_keywords = config.get("sensitive_keywords", [])

        # 方法选择
        self.vocal_method = config.get("vocal_method", "librosa")  # librosa / demucs
        self.music_method = config.get("music_method", "librosa")  # librosa / yamnet / panns

        # demucs 模型（懒加载）
        self._demucs_model = None

        logger.info("内容过滤器初始化完成")
        logger.info(f"  非音乐过滤: {'开启' if self.enable_non_music_filter else '关闭'}")
        logger.info(f"  最低音乐性评分: {self.min_music_score}")
        logger.info(f"  人声占比上限: {self.max_vocal_ratio}")
        logger.info(f"  纯器乐模式: {'开启' if self.instrumental_only else '关闭'}")
        logger.info(f"  内容安全过滤: {'开启' if self.enable_safety_filter else '关闭'}")
        logger.info(f"  人声检测方法: {self.vocal_method}")
        logger.info(f"  音乐性检测方法: {self.music_method}")

    def analyze(self, audio_path: str) -> ContentAnalysisResult:
        """
        分析音频内容

        Args:
            audio_path: 音频文件路径

        Returns:
            ContentAnalysisResult: 分析结果
        """
        result = ContentAnalysisResult(audio_path=audio_path)

        try:
            # 加载音频
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            duration = len(y) / sr

            if duration < 1.0:
                logger.warning(f"音频过短 ({duration:.1f}s)，跳过滤波分析")
                return result

            # 基础特征提取
            self._extract_basic_features(y, sr, result)

            # 音乐性评分
            # 先运行规则兜底检测（YAMNet 替代方案，P0 先用规则）
            rule_non_music = self._rule_based_non_music_detection(y, sr, audio_path, result)

            if not rule_non_music:
                if self.music_method == "librosa":
                    self._score_music_librosa(y, sr, result)
                elif self.music_method == "yamnet":
                    self._score_music_yamnet(y, sr, result)
                elif self.music_method == "panns":
                    self._score_music_panns(y, sr, result)

            # 人声占比检测
            if self.vocal_method == "librosa":
                self._detect_vocal_librosa(y, sr, result)
            elif self.vocal_method == "demucs":
                self._detect_vocal_demucs(y, sr, result)

            # 内容安全过滤（需要 ASR，默认关闭）
            if self.enable_safety_filter:
                self._check_safety(audio_path, result)

            # 判定
            result.is_music = result.music_score >= self.min_music_score
            result.is_instrumental = result.vocal_ratio < 0.1

        except Exception as e:
            logger.warning(f"内容分析失败: {audio_path} -> {str(e)}")

        return result

    def _extract_basic_features(self, y: np.ndarray, sr: int, result: ContentAnalysisResult):
        """提取基础音频特征"""
        #  tempo
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            result.tempo = float(tempo) if hasattr(tempo, '__float__') else float(tempo[0]) if hasattr(tempo, '__len__') else 0.0
        except Exception:
            result.tempo = 0.0

        # 频谱质心
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            result.spectral_centroid = float(np.mean(spectral_centroids))
        except Exception:
            result.spectral_centroid = 0.0

        # 过零率
        try:
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            result.zero_crossing_rate = float(np.mean(zcr))
        except Exception:
            result.zero_crossing_rate = 0.0

        # 谐波/打击乐分离
        try:
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            harmonic_energy = np.sum(y_harmonic ** 2)
            percussive_energy = np.sum(y_percussive ** 2)
            total_energy = harmonic_energy + percussive_energy + 1e-10
            result.harmonic_energy_ratio = harmonic_energy / total_energy
            result.percussive_energy_ratio = percussive_energy / total_energy
        except Exception:
            result.harmonic_energy_ratio = 0.0
            result.percussive_energy_ratio = 0.0

    def _rule_based_non_music_detection(self, y: np.ndarray, sr: int, audio_path: str, result: ContentAnalysisResult) -> bool:
        """
        基于规则的非音乐检测（YAMNet 兜底方案，P0 先用规则）

        规则：
        1. 静音占比 > 70%（几乎全静音，可能不是音乐）
        2. 时长 < 3秒（太短，可能不是音乐）
        3. 文件名关键词（speech/interview/podcast/lecture/语音/访谈/播客等）
        4. 频谱平坦度 > 0.5（持续噪声/环境音，如空调声、电流声）

        Returns:
            bool: True 表示规则判定为非音乐，False 表示需要进一步分析
        """
        try:
            duration = len(y) / sr
            filename = Path(audio_path).name.lower()

            # 计算静音占比
            try:
                non_silent = librosa.effects.split(y, top_db=30)
                if len(non_silent) > 0:
                    non_silent_duration = sum(end - start for start, end in non_silent) / sr
                    silence_ratio = 1.0 - (non_silent_duration / duration)
                else:
                    silence_ratio = 1.0
            except Exception:
                silence_ratio = 0.0

            # 规则1: 静音占比 > 70%
            if silence_ratio > 0.7:
                result.music_score = 0.1
                result.music_confidence = 0.8
                result.method_used = "rule_based"
                logger.info(f"    规则判定非音乐: 静音占比 {silence_ratio*100:.1f}% > 70%")
                return True

            # 规则2: 时长 < 3秒
            if duration < 3.0:
                result.music_score = 0.2
                result.music_confidence = 0.7
                result.method_used = "rule_based"
                logger.info(f"    规则判定非音乐: 时长 {duration:.1f}s < 3s")
                return True

            # 规则3: 文件名关键词
            non_music_keywords = [
                "speech", "interview", "podcast", "lecture", "seminar",
                "presentation", "meeting", "conversation", "dialogue",
                "语音", "访谈", "播客", "讲座", "演讲", "会议", "对话",
                "noise", "silence", "ambient", "environment", "field_recording",
                "噪声", "静音", "环境音", "现场录音",
            ]
            for kw in non_music_keywords:
                if kw in filename:
                    result.music_score = 0.15
                    result.music_confidence = 0.75
                    result.method_used = "rule_based"
                    logger.info(f"    规则判定非音乐: 文件名含关键词 '{kw}'")
                    return True

            # 规则4: 频谱平坦度 > 0.5（持续噪声/环境音）
            try:
                S = np.abs(librosa.stft(y))
                spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(S=S)))
                if spectral_flatness > 0.5:
                    result.music_score = 0.2
                    result.music_confidence = 0.6
                    result.method_used = "rule_based"
                    logger.info(f"    规则判定非音乐: 频谱平坦度 {spectral_flatness:.3f} > 0.5")
                    return True
            except Exception:
                pass

            return False

        except Exception as e:
            logger.warning(f"    规则检测失败: {e}")
            return False

    def _score_music_librosa(self, y: np.ndarray, sr: int, result: ContentAnalysisResult):
        """
        基于 librosa 简单特征的音乐性评分

        评分依据：
        - 节奏规律性：有稳定节拍的更可能是音乐
        - 频谱复杂度：音乐的频谱变化更丰富
        - 谐波能量占比：音乐通常有较强的谐波成分
        - 动态范围：音乐通常有一定的动态变化
        """
        scores = []
        confidences = []

        # 1. 节奏规律性（基于 onset 强度的自相关）
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            # 计算 onset 强度的方差，方差低说明节奏稳定
            onset_std = np.std(onset_env)
            onset_mean = np.mean(onset_env) + 1e-10
            rhythm_cv = onset_std / onset_mean  # 变异系数
            # 变异系数在 0.3-0.8 之间通常是有规律的音乐
            if 0.2 < rhythm_cv < 1.0:
                rhythm_score = 1.0 - abs(rhythm_cv - 0.5) / 0.5
                rhythm_score = max(0.0, min(1.0, rhythm_score))
            else:
                rhythm_score = 0.2
            scores.append(rhythm_score)
            confidences.append(0.6)
            result.rhythm_regularity = rhythm_score
        except Exception:
            scores.append(0.5)
            confidences.append(0.3)

        # 2. 频谱复杂度（基于 MFCC 的方差）
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_std = np.mean(np.std(mfcc, axis=1))
            # MFCC 方差适中的更可能是音乐
            complexity_score = min(1.0, mfcc_std / 50.0)
            scores.append(complexity_score)
            confidences.append(0.5)
        except Exception:
            scores.append(0.5)
            confidences.append(0.3)

        # 3. 谐波能量占比
        harmonic_ratio = result.harmonic_energy_ratio
        # 音乐通常谐波占比在 0.4-0.8 之间
        if 0.3 < harmonic_ratio < 0.9:
            harmonic_score = 1.0 - abs(harmonic_ratio - 0.6) / 0.4
            harmonic_score = max(0.0, min(1.0, harmonic_score))
        else:
            harmonic_score = 0.3
        scores.append(harmonic_score)
        confidences.append(0.5)

        # 4. 过零率（语音的过零率通常较高且波动大）
        zcr = result.zero_crossing_rate
        # 音乐的过零率通常在 0.05-0.3 之间
        if 0.02 < zcr < 0.4:
            zcr_score = 1.0 - abs(zcr - 0.15) / 0.2
            zcr_score = max(0.0, min(1.0, zcr_score))
        else:
            zcr_score = 0.3
        scores.append(zcr_score)
        confidences.append(0.4)

        # 加权平均
        total_confidence = sum(confidences) + 1e-10
        result.music_score = sum(s * c for s, c in zip(scores, confidences)) / total_confidence
        result.music_confidence = total_confidence / len(confidences)
        result.method_used = "librosa_basic"

    def _score_music_yamnet(self, y: np.ndarray, sr: int, result: ContentAnalysisResult):
        """
        基于 YAMNet 的音乐性检测

        需要 tensorflow + tensorflow_hub
        """
        try:
            import tensorflow as tf
            import tensorflow_hub as hub

            # 加载 YAMNet 模型（懒加载）
            if not hasattr(self, '_yamnet_model'):
                self._yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

            # YAMNet 需要 16kHz 单声道
            if sr != 16000:
                y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
            else:
                y_16k = y

            # 运行推理
            scores, embeddings, spectrogram = self._yamnet_model(y_16k)
            class_scores = scores.numpy().mean(axis=0)

            # YAMNet 类别中，音乐相关的类别
            # 52: Music, 53: Musical instrument, 54: Singing, 55: Choir
            music_class_indices = [52, 53, 54, 55]
            music_score = float(sum(class_scores[i] for i in music_class_indices))

            # 语音相关类别
            # 0: Speech, 1: Child speech, 2: Conversation
            speech_class_indices = [0, 1, 2]
            speech_score = float(sum(class_scores[i] for i in speech_class_indices))

            # 音乐性评分 = 音乐得分 / (音乐得分 + 语音得分)
            total = music_score + speech_score + 1e-10
            result.music_score = music_score / total
            result.music_confidence = 0.9
            result.method_used = "yamnet"

            logger.info(f"YAMNet: music={music_score:.3f}, speech={speech_score:.3f}, score={result.music_score:.3f}")

        except ImportError:
            logger.warning("tensorflow/tensorflow_hub 未安装，回退到 librosa 基础方法")
            self._score_music_librosa(y, sr, result)
        except Exception as e:
            logger.warning(f"YAMNet 推理失败: {str(e)}，回退到 librosa 基础方法")
            self._score_music_librosa(y, sr, result)

    def _score_music_panns(self, y: np.ndarray, sr: int, result: ContentAnalysisResult):
        """
        基于 PANNs 的音乐性检测

        需要 torch + 预训练模型
        """
        try:
            import torch
            # PANNs 模型需要额外下载，这里提供接口
            # 实际使用时需要加载预训练的 Cnn14 模型
            logger.warning("PANNs 模型需要额外下载，回退到 librosa 基础方法")
            self._score_music_librosa(y, sr, result)
        except ImportError:
            logger.warning("torch 未安装，回退到 librosa 基础方法")
            self._score_music_librosa(y, sr, result)

    def _detect_vocal_librosa(self, y: np.ndarray, sr: int, result: ContentAnalysisResult):
        """
        基于 librosa 的人声占比估算

        方法：
        - 谐波/打击乐分离后，谐波部分可能包含人声
        - 人声通常在 100-3000Hz 频段
        - 用频谱通量（spectral flux）检测人声的变化
        """
        try:
            # 方法1：基于谐波能量和频段能量
            y_harmonic, y_percussive = librosa.effects.hpss(y)

            # 计算 100-3000Hz 频段的能量（人声主要频段）
            S = np.abs(librosa.stft(y_harmonic))
            freqs = librosa.fft_frequencies(sr=sr)
            vocal_band_mask = (freqs >= 100) & (freqs <= 3000)
            vocal_band_energy = np.sum(S[vocal_band_mask, :] ** 2)
            total_energy = np.sum(S ** 2) + 1e-10
            vocal_ratio_1 = vocal_band_energy / total_energy

            # 方法2：基于频谱通量（spectral flux）
            # 人声的频谱变化通常比器乐更频繁
            spectral_flux = np.mean(np.diff(S, axis=1) ** 2)
            # 归一化
            flux_normalized = min(1.0, spectral_flux / 10.0)
            vocal_ratio_2 = flux_normalized * 0.5

            # 综合估算
            vocal_ratio = 0.7 * vocal_ratio_1 + 0.3 * vocal_ratio_2
            vocal_ratio = max(0.0, min(1.0, vocal_ratio))

            result.vocal_ratio = vocal_ratio
            result.vocal_confidence = 0.5  # librosa 方法置信度中等

        except Exception as e:
            logger.warning(f"人声检测(librosa)失败: {str(e)}")
            result.vocal_ratio = 0.5
            result.vocal_confidence = 0.0

    def _detect_vocal_demucs(self, y: np.ndarray, sr: int, result: ContentAnalysisResult):
        """
        基于 demucs 的精确人声分离

        demucs 已安装，可以分离人声/鼓/贝斯/其他
        """
        try:
            import torch
            import demucs
            from demucs.pretrained import get_model
            from demucs.apply import apply_model

            # 加载模型（懒加载）
            if self._demucs_model is None:
                self._demucs_model = get_model('htdemucs')
                self._demucs_model.eval()
                if torch.cuda.is_available():
                    self._demucs_model.cuda()

            # demucs 需要 44100Hz 立体声
            if sr != 44100:
                y_44k = librosa.resample(y, orig_sr=sr, target_sr=44100)
            else:
                y_44k = y.copy()

            # 转立体声
            if y_44k.ndim == 1:
                y_stereo = np.vstack([y_44k, y_44k])
            else:
                y_stereo = y_44k

            # 转为 tensor
            wav = torch.tensor(y_stereo).float().unsqueeze(0)
            if torch.cuda.is_available():
                wav = wav.cuda()

            # 分离
            with torch.no_grad():
                sources = apply_model(self._demucs_model, wav, progress=False)[0]

            # sources: [drums, bass, other, vocals]
            vocals = sources[3].cpu().numpy()
            other = sources[2].cpu().numpy()

            # 计算人声能量占比
            vocal_energy = np.sum(vocals ** 2)
            other_energy = np.sum(other ** 2)
            total_energy = vocal_energy + other_energy + 1e-10
            vocal_ratio = vocal_energy / total_energy

            result.vocal_ratio = float(vocal_ratio)
            result.vocal_confidence = 0.9  # demucs 方法置信度高
            result.method_used = "demucs"

            logger.info(f"demucs: vocal={vocal_energy:.2f}, other={other_energy:.2f}, ratio={vocal_ratio:.3f}")

        except ImportError:
            logger.warning("demucs 未安装，回退到 librosa 方法")
            self._detect_vocal_librosa(y, sr, result)
        except Exception as e:
            logger.warning(f"人声检测(demucs)失败: {str(e)}，回退到 librosa 方法")
            self._detect_vocal_librosa(y, sr, result)

    def _check_safety(self, audio_path: str, result: ContentAnalysisResult):
        """
        内容安全检查

        需要 ASR（Whisper）+ 关键词检测
        默认关闭
        """
        try:
            import whisper

            # 加载 Whisper 模型（懒加载）
            if not hasattr(self, '_whisper_model'):
                self._whisper_model = whisper.load_model("base")

            # 转写
            transcription = self._whisper_model.transcribe(audio_path)
            text = transcription.get("text", "").lower()

            # 关键词检测
            for keyword in self.sensitive_keywords:
                if keyword.lower() in text:
                    result.safety_warnings.append(f"敏感关键词: {keyword}")
                    result.is_safe = False

            if not result.is_safe:
                logger.warning(f"内容安全警告: {result.safety_warnings}")

        except ImportError:
            logger.warning("whisper 未安装，跳过内容安全检查")
        except Exception as e:
            logger.warning(f"内容安全检查失败: {str(e)}")

    def filter(self, audio_path: str) -> Tuple[bool, ContentAnalysisResult]:
        """
        过滤音频

        Args:
            audio_path: 音频文件路径

        Returns:
            (passed, result): 是否通过过滤，分析结果
        """
        result = self.analyze(audio_path)
        passed = True

        # 非音乐过滤
        if self.enable_non_music_filter and not result.is_music:
            passed = False
            logger.info(f"非音乐内容过滤: {audio_path} (music_score={result.music_score:.3f})")

        # 人声过滤
        if self.instrumental_only and not result.is_instrumental:
            passed = False
            logger.info(f"纯器乐过滤: {audio_path} (vocal_ratio={result.vocal_ratio:.3f})")

        if result.vocal_ratio > self.max_vocal_ratio:
            passed = False
            logger.info(f"人声占比过高: {audio_path} (vocal_ratio={result.vocal_ratio:.3f})")

        # 内容安全
        if self.enable_safety_filter and not result.is_safe:
            passed = False
            logger.info(f"内容安全过滤: {audio_path}")

        return passed, result


def batch_filter(
    audio_paths: List[str],
    config: Optional[Dict] = None,
    report_csv: Optional[str] = None,
) -> Tuple[List[ContentAnalysisResult], pd.DataFrame]:
    """
    批量内容过滤

    Args:
        audio_paths: 音频文件路径列表
        config: 内容过滤配置
        report_csv: 报告输出路径

    Returns:
        (results, report_df): 分析结果列表和报告 DataFrame
    """
    content_filter = ContentFilter(config)
    results = []

    logger.info(f"开始批量内容过滤: {len(audio_paths)} 个文件")

    for i, audio_path in enumerate(audio_paths):
        logger.info(f"[{i+1}/{len(audio_paths)}] 分析: {audio_path}")
        passed, result = content_filter.filter(audio_path)
        results.append(result)

        status = "✅ 通过" if passed else "❌ 未通过"
        logger.info(f"  {status} | music:{result.music_score:.2f} | "
                    f"vocal:{result.vocal_ratio:.2f} | "
                    f"tempo:{result.tempo:.0f} | method:{result.method_used}")

    # 生成报告
    report_data = [r.to_dict() for r in results]
    report_df = pd.DataFrame(report_data)

    if report_csv:
        os.makedirs(os.path.dirname(report_csv), exist_ok=True)
        report_df.to_csv(report_csv, index=False, encoding="utf-8")
        logger.info(f"内容过滤报告已保存: {report_csv}")

    # 统计
    passed = sum(1 for r in results if r.is_music)
    logger.info(f"批量过滤完成: 音乐 {passed}/{len(results)}")

    return results, report_df


# ===================== 文本安全检测（关键词/NLP，不需要ASR） =====================

@dataclass
class TextSafetyResult:
    """文本安全检测结果"""
    text: str
    is_safe: bool = True
    matched_keywords: List[Tuple[str, str]] = field(default_factory=list)  # (category, keyword)
    matched_regex: List[Tuple[str, str]] = field(default_factory=list)  # (pattern_name, matched_text)
    risk_score: float = 0.0  # 0.0-1.0，越高越危险
    risk_level: str = "safe"  # safe / low / medium / high

    def to_dict(self) -> Dict:
        return {
            "is_safe": self.is_safe,
            "risk_score": round(self.risk_score, 4),
            "risk_level": self.risk_level,
            "matched_keywords": "; ".join(f"[{c}]{k}" for c, k in self.matched_keywords[:20]),
            "matched_regex": "; ".join(f"[{n}]{t}" for n, t in self.matched_regex[:20]),
            "total_matches": len(self.matched_keywords) + len(self.matched_regex),
        }


class TextSafetyFilter:
    """
    文本安全过滤器（关键词/NLP检测）

    对元数据文本字段（description/lyrics/notes/title）做敏感内容检测，
    不需要 ASR，直接对已有文本生效。

    检测维度：
    1. 关键词匹配：可配置的敏感关键词列表（按类别分组）
    2. 正则匹配：可配置的正则模式（如敏感联系方式、特定格式）
    3. 风险评分：根据匹配数量和类别权重计算风险分
    """

    # 默认关键词分类（可通过配置覆盖）
    DEFAULT_KEYWORD_CATEGORIES = {
        "violence": {
            "weight": 0.8,
            "keywords": ["暴力", "血腥", "殴打", "杀人", "谋杀", "恐怖", "袭击", "自残", "自杀"],
        },
        "drugs": {
            "weight": 0.9,
            "keywords": ["毒品", "吸毒", "贩毒", "海洛因", "可卡因", "大麻", "冰毒", "摇头丸"],
        },
        "political_sensitive": {
            "weight": 1.0,
            "keywords": [],  # 由用户配置，默认空
        },
        "pornography": {
            "weight": 0.7,
            "keywords": ["色情", "淫秽", "成人", "三级", "裸体"],
        },
        "gambling": {
            "weight": 0.6,
            "keywords": ["赌博", "赌场", "博彩", "下注", "赌球"],
        },
        "fraud": {
            "weight": 0.7,
            "keywords": ["诈骗", "传销", "非法集资", "骗局", "欺诈"],
        },
        "hate_speech": {
            "weight": 0.8,
            "keywords": ["歧视", "种族主义", "仇恨", "排外", "性别歧视"],
        },
    }

    # 默认正则模式
    DEFAULT_REGEX_PATTERNS = {
        "phone_number": {
            "weight": 0.3,
            "pattern": r"1[3-9]\d{9}",
        },
        "id_card": {
            "weight": 0.5,
            "pattern": r"\d{17}[\dXx]",
        },
        "bank_card": {
            "weight": 0.4,
            "pattern": r"\d{16,19}",
        },
        "url": {
            "weight": 0.2,
            "pattern": r"https?://[^\s<>\"]+",
        },
    }

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化文本安全过滤器

        Args:
            config: 文本安全配置（来自 cleaning_config.yaml 的 stage3_quality.content_filter.text_safety）
        """
        if config is None:
            config = {}

        self.enabled = config.get("enabled", False)
        self.keyword_categories = config.get("keyword_categories", self.DEFAULT_KEYWORD_CATEGORIES)
        self.regex_patterns = config.get("regex_patterns", self.DEFAULT_REGEX_PATTERNS)
        self.risk_threshold = config.get("risk_threshold", 0.5)  # 超过此阈值判定为不安全
        self.case_sensitive = config.get("case_sensitive", False)

        # 预编译正则
        self._compiled_regex = {}
        for name, pat_config in self.regex_patterns.items():
            flags = 0 if self.case_sensitive else re.IGNORECASE
            self._compiled_regex[name] = (
                re.compile(pat_config["pattern"], flags),
                pat_config.get("weight", 0.5),
            )

        logger.info("文本安全过滤器初始化完成")
        logger.info(f"  启用: {'是' if self.enabled else '否'}")
        logger.info(f"  关键词分类: {len(self.keyword_categories)} 个")
        logger.info(f"  正则模式: {len(self.regex_patterns)} 个")
        logger.info(f"  风险阈值: {self.risk_threshold}")

    def check(self, text: str) -> TextSafetyResult:
        """
        检测文本安全性

        Args:
            text: 待检测文本

        Returns:
            TextSafetyResult: 检测结果
        """
        result = TextSafetyResult(text=text)

        if not text or not isinstance(text, str) or not self.enabled:
            return result

        text_lower = text if self.case_sensitive else text.lower()

        # 1. 关键词匹配
        for category, cat_config in self.keyword_categories.items():
            weight = cat_config.get("weight", 0.5)
            keywords = cat_config.get("keywords", [])
            for kw in keywords:
                kw_lower = kw if self.case_sensitive else kw.lower()
                if kw_lower in text_lower:
                    result.matched_keywords.append((category, kw))
                    result.risk_score += weight * 0.2  # 每个关键词贡献权重*0.2

        # 2. 正则匹配
        for name, (pattern, weight) in self._compiled_regex.items():
            matches = pattern.findall(text)
            for match in matches:
                match_text = match if isinstance(match, str) else str(match)
                result.matched_regex.append((name, match_text[:50]))
                result.risk_score += weight * 0.15  # 每个正则匹配贡献权重*0.15

        # 3. 风险评分归一化
        result.risk_score = min(1.0, result.risk_score)

        # 4. 风险等级
        if result.risk_score == 0:
            result.risk_level = "safe"
        elif result.risk_score < 0.3:
            result.risk_level = "low"
        elif result.risk_score < 0.6:
            result.risk_level = "medium"
        else:
            result.risk_level = "high"

        # 5. 是否安全
        result.is_safe = result.risk_score < self.risk_threshold

        if not result.is_safe:
            logger.warning(f"文本安全警告: risk={result.risk_score:.3f}, level={result.risk_level}, "
                          f"keywords={len(result.matched_keywords)}, regex={len(result.matched_regex)}")

        return result

    def check_dataframe(
        self,
        df: pd.DataFrame,
        columns: List[str],
        add_report_columns: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        批量检测 DataFrame 中的文本列

        Args:
            df: 输入 DataFrame
            columns: 需要检测的列名列表
            add_report_columns: 是否添加报告列

        Returns:
            (checked_df, report_df)
        """
        checked_df = df.copy()
        report_rows = []

        for idx, row in df.iterrows():
            row_report = {"row_index": idx}
            max_risk = 0.0
            all_matches = 0
            unsafe_columns = []

            for col in columns:
                if col not in df.columns:
                    continue
                text = row[col]
                if pd.isna(text) or not isinstance(text, str):
                    continue

                result = self.check(text)
                all_matches += len(result.matched_keywords) + len(result.matched_regex)
                if result.risk_score > max_risk:
                    max_risk = result.risk_score
                if not result.is_safe:
                    unsafe_columns.append(col)
                    row_report[f"{col}_risk"] = result.risk_score
                    row_report[f"{col}_level"] = result.risk_level
                    row_report[f"{col}_matches"] = result.to_dict()["matched_keywords"]

            row_report["max_risk_score"] = max_risk
            row_report["total_matches"] = all_matches
            row_report["unsafe_columns"] = "; ".join(unsafe_columns)
            row_report["is_safe"] = max_risk < self.risk_threshold
            report_rows.append(row_report)

        report_df = pd.DataFrame(report_rows)

        if add_report_columns:
            checked_df["_text_safety_risk"] = report_df["max_risk_score"].values
            checked_df["_text_safety_level"] = report_df["max_risk_score"].apply(
                lambda x: "safe" if x == 0 else "low" if x < 0.3 else "medium" if x < 0.6 else "high"
            ).values
            checked_df["_text_safe"] = report_df["is_safe"].values

        unsafe_count = sum(1 for r in report_rows if not r["is_safe"])
        logger.info(f"文本安全检测完成: {len(df)} 行, 不安全 {unsafe_count} 行, 总匹配 {sum(r['total_matches'] for r in report_rows)}")

        return checked_df, report_df


def batch_text_safety_check(
    df: pd.DataFrame,
    columns: List[str],
    config: Optional[Dict] = None,
    report_csv: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """批量文本安全检测（便捷函数）"""
    safety_filter = TextSafetyFilter(config)
    checked_df, report_df = safety_filter.check_dataframe(df, columns)

    if report_csv:
        os.makedirs(os.path.dirname(report_csv), exist_ok=True)
        report_df.to_csv(report_csv, index=False, encoding="utf-8")
        logger.info(f"文本安全检测报告已保存: {report_csv}")

    return checked_df, report_df


if __name__ == "__main__":
    # 测试
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    test_audio = "/path/to/test.wav"
    if os.path.exists(test_audio):
        filter = ContentFilter()
        result = filter.analyze(test_audio)
        print(f"is_music: {result.is_music}")
        print(f"music_score: {result.music_score}")
        print(f"vocal_ratio: {result.vocal_ratio}")
        print(f"method: {result.method_used}")
    else:
        print("测试文件不存在")
        print("用法: from content_filter import ContentFilter")
