"""
quality_check.py
音频质量清洗模块（Stage 3）

功能：
- 损坏检测（ffmpeg -v error）
- 静音过滤（librosa.effects.split）
- 时长过滤
- 采样率/位深/声道检查
- 削波/过载检测
- 信噪比（SNR）估算
- 动态范围评估
- 响度归一化（pyloudnorm, ITU-R BS.1770-4）

用法：
    from quality_check import AudioQualityChecker
    checker = AudioQualityChecker(config)
    result = checker.check(audio_path)
    if result.passed:
        # 通过质量检查
    else:
        # 未通过，查看 result.reasons
"""
import os
import sys
import subprocess
import logging
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# 添加 utils 目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "utils"))

logger = logging.getLogger(__name__)


@dataclass
class QualityResult:
    """音频质量检查结果"""
    audio_path: str
    passed: bool = True
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # 基础信息
    duration: float = 0.0
    sample_rate: int = 0
    bit_depth: int = 0
    channels: int = 0
    file_size: int = 0

    # 质量指标
    silence_ratio: float = 0.0
    clipping_ratio: float = 0.0
    snr_db: float = 0.0
    dynamic_range_db: float = 0.0
    loudness_lufs: float = 0.0
    true_peak_db: float = 0.0

    # 标记
    corrupted: bool = False
    too_short: bool = False
    too_long: bool = False
    low_sample_rate: bool = False
    low_bit_depth: bool = False
    high_silence: bool = False
    high_clipping: bool = False
    low_snr: bool = False
    low_dynamic_range: bool = False
    loudness_normalized: bool = False

    def add_reason(self, reason: str):
        """添加未通过原因"""
        self.passed = False
        self.reasons.append(reason)

    def add_warning(self, warning: str):
        """添加警告（不影响通过）"""
        self.warnings.append(warning)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "audio_path": self.audio_path,
            "passed": self.passed,
            "reasons": "; ".join(self.reasons),
            "warnings": "; ".join(self.warnings),
            "duration": round(self.duration, 3),
            "sample_rate": self.sample_rate,
            "bit_depth": self.bit_depth,
            "channels": self.channels,
            "file_size": self.file_size,
            "silence_ratio": round(self.silence_ratio, 4),
            "clipping_ratio": round(self.clipping_ratio, 6),
            "snr_db": round(self.snr_db, 2),
            "dynamic_range_db": round(self.dynamic_range_db, 2),
            "loudness_lufs": round(self.loudness_lufs, 2),
            "true_peak_db": round(self.true_peak_db, 2),
            "corrupted": self.corrupted,
            "too_short": self.too_short,
            "too_long": self.too_long,
            "low_sample_rate": self.low_sample_rate,
            "low_bit_depth": self.low_bit_depth,
            "high_silence": self.high_silence,
            "high_clipping": self.high_clipping,
            "low_snr": self.low_snr,
            "low_dynamic_range": self.low_dynamic_range,
            "loudness_normalized": self.loudness_normalized,
        }


class AudioQualityChecker:
    """音频质量检查器"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化质量检查器

        Args:
            config: 质量检查配置（来自 cleaning_config.yaml 的 stage3_quality）
        """
        if config is None:
            config = {}

        # 损坏检测
        corruption_cfg = config.get("corruption_detection", {})
        self.reject_corrupted = corruption_cfg.get("reject_corrupted", True)

        # 静音过滤
        silence_cfg = config.get("silence_filter", {})
        self.silence_threshold = silence_cfg.get("silence_threshold", 0.001)
        self.max_silence_ratio = silence_cfg.get("max_silence_ratio", 0.5)

        # 音质门槛
        quality_cfg = config.get("quality_threshold", {})
        self.min_sample_rate = quality_cfg.get("min_sample_rate", 44100)
        self.min_bit_depth = quality_cfg.get("min_bit_depth", 16)
        self.min_snr = quality_cfg.get("min_snr", 15)
        self.max_clipping_ratio = quality_cfg.get("max_clipping_ratio", 0.005)
        self.min_dynamic_range = quality_cfg.get("min_dynamic_range", 4)

        # 时长过滤
        self.min_duration = quality_cfg.get("min_duration", 5)
        self.max_duration = quality_cfg.get("max_duration", 0)  # 0表示不设上限

        # 响度归一化
        loudness_cfg = config.get("loudness_normalization", {})
        self.normalize_loudness = loudness_cfg.get("enabled", True)
        self.target_lufs = loudness_cfg.get("target_lufs", -14)
        self.true_peak_max = loudness_cfg.get("true_peak_max", -1)
        self.loudness_tolerance = loudness_cfg.get("tolerance", 0.5)

        logger.info(f"音频质量检查器初始化完成")
        logger.info(f"  最小采样率: {self.min_sample_rate} Hz")
        logger.info(f"  最小位深: {self.min_bit_depth}-bit")
        logger.info(f"  最大静音占比: {self.max_silence_ratio * 100}%")
        logger.info(f"  最大削波比例: {self.max_clipping_ratio * 100}%")
        logger.info(f"  最小SNR: {self.min_snr} dB")
        logger.info(f"  最小动态范围: {self.min_dynamic_range} dB")
        logger.info(f"  目标响度: {self.target_lufs} LUFS")
        logger.info(f"  时长范围: {self.min_duration}s - {self.max_duration}s")

    def check(self, audio_path: str, output_path: Optional[str] = None) -> QualityResult:
        """
        执行完整的音频质量检查

        Args:
            audio_path: 音频文件路径
            output_path: 响度归一化后的输出路径（如果为 None 则不输出）

        Returns:
            QualityResult: 质量检查结果
        """
        result = QualityResult(audio_path=audio_path)

        # 1. 基础文件信息
        self._check_file_info(audio_path, result)
        if result.corrupted and self.reject_corrupted:
            return result

        # 2. 加载音频
        try:
            y, sr = librosa.load(audio_path, sr=None, mono=False)
            if y.ndim > 1:
                y_mono = librosa.to_mono(y)
            else:
                y_mono = y
            result.sample_rate = sr
            result.duration = len(y_mono) / sr
        except Exception as e:
            result.corrupted = True
            result.add_reason(f"音频加载失败: {str(e)}")
            return result

        # 3. 时长过滤
        self._check_duration(result)

        # 4. 采样率/位深检查
        self._check_format(result)

        # 5. 静音检测
        self._check_silence(y_mono, sr, result)

        # 6. 削波检测
        self._check_clipping(y_mono, result)

        # 7. SNR 估算
        self._estimate_snr(y_mono, sr, result)

        # 8. 动态范围评估
        self._estimate_dynamic_range(y_mono, result)

        # 9. 响度测量与归一化
        if self.normalize_loudness:
            self._measure_and_normalize_loudness(audio_path, y_mono, sr, result, output_path)

        return result

    def _check_file_info(self, audio_path: str, result: QualityResult):
        """检查文件基础信息和损坏检测"""
        path = Path(audio_path)
        if not path.exists():
            result.corrupted = True
            result.add_reason("文件不存在")
            return

        result.file_size = path.stat().st_size

        # 使用 ffmpeg 检测损坏
        try:
            cmd = [
                "ffmpeg", "-v", "error", "-i", str(path),
                "-f", "null", "-"
            ]
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            if proc.returncode != 0 or proc.stderr.strip():
                result.corrupted = True
                error_msg = proc.stderr.strip()[:200]
                result.add_reason(f"ffmpeg检测到损坏: {error_msg}")
                return
        except FileNotFoundError:
            result.add_warning("ffmpeg 未安装，跳过损坏检测")
        except subprocess.TimeoutExpired:
            result.add_warning("ffmpeg检测超时，跳过损坏检测")

        # 使用 soundfile 获取格式信息
        try:
            info = sf.info(audio_path)
            result.sample_rate = info.samplerate
            result.channels = info.channels
            # 从 subtype 推断位深
            subtype = info.subtype
            if 'PCM_16' in subtype:
                result.bit_depth = 16
            elif 'PCM_24' in subtype:
                result.bit_depth = 24
            elif 'PCM_32' in subtype:
                result.bit_depth = 32
            elif 'FLOAT' in subtype:
                result.bit_depth = 32
            else:
                result.bit_depth = 0
        except Exception as e:
            result.add_warning(f"soundfile读取格式信息失败: {str(e)}")

    def _check_duration(self, result: QualityResult):
        """检查时长"""
        if result.duration < self.min_duration:
            result.too_short = True
            result.add_reason(f"时长过短: {result.duration:.1f}s < {self.min_duration}s")
        elif self.max_duration > 0 and result.duration > self.max_duration:
            result.too_long = True
            result.add_warning(f"时长较长: {result.duration:.1f}s > {self.max_duration}s (视任务调整，不淘汰)")

    def _check_format(self, result: QualityResult):
        """检查采样率和位深"""
        if result.sample_rate > 0 and result.sample_rate < self.min_sample_rate:
            result.low_sample_rate = True
            result.add_reason(f"采样率过低: {result.sample_rate}Hz < {self.min_sample_rate}Hz")

        if result.bit_depth > 0 and result.bit_depth < self.min_bit_depth:
            result.low_bit_depth = True
            result.add_warning(f"位深较低: {result.bit_depth}-bit < {self.min_bit_depth}-bit")

    def _check_silence(self, y: np.ndarray, sr: int, result: QualityResult):
        """检测静音占比"""
        try:
            # 使用 librosa.effects.split 检测非静音段
            non_silent = librosa.effects.split(
                y, top_db=20, frame_length=2048, hop_length=512
            )

            if len(non_silent) == 0:
                result.silence_ratio = 1.0
            else:
                non_silent_duration = sum(
                    (end - start) / sr for start, end in non_silent
                )
                result.silence_ratio = 1.0 - (non_silent_duration / result.duration)

            if result.silence_ratio > self.max_silence_ratio:
                result.high_silence = True
                result.add_reason(
                    f"静音占比过高: {result.silence_ratio * 100:.1f}% > {self.max_silence_ratio * 100}%"
                )
        except Exception as e:
            result.add_warning(f"静音检测失败: {str(e)}")

    def _check_clipping(self, y: np.ndarray, result: QualityResult):
        """检测削波/过载"""
        try:
            # 检测触及 0dBFS（±1.0）的样本比例
            peak_threshold = 0.999
            clipped_samples = np.sum(np.abs(y) >= peak_threshold)
            result.clipping_ratio = clipped_samples / len(y)

            if result.clipping_ratio > self.max_clipping_ratio:
                result.high_clipping = True
                result.add_reason(
                    f"削波比例过高: {result.clipping_ratio * 100:.3f}% > {self.max_clipping_ratio * 100}%"
                )
        except Exception as e:
            result.add_warning(f"削波检测失败: {str(e)}")

    def _estimate_snr(self, y: np.ndarray, sr: int, result: QualityResult):
        """估算信噪比（SNR）"""
        try:
            # 简单的 SNR 估算：信号能量 / 噪声能量
            # 使用 librosa.effects.split 分离信号和噪声
            non_silent = librosa.effects.split(
                y, top_db=20, frame_length=2048, hop_length=512
            )

            if len(non_silent) == 0:
                result.snr_db = 0.0
                return

            # 信号段能量
            signal_energy = 0
            signal_samples = 0
            for start, end in non_silent:
                segment = y[start:end]
                signal_energy += np.sum(segment ** 2)
                signal_samples += len(segment)

            # 噪声段能量（非信号段）
            noise_mask = np.ones(len(y), dtype=bool)
            for start, end in non_silent:
                noise_mask[start:end] = False
            noise_energy = np.sum(y[noise_mask] ** 2)
            noise_samples = np.sum(noise_mask)

            if noise_samples > 0 and noise_energy > 0:
                signal_power = signal_energy / max(signal_samples, 1)
                noise_power = noise_energy / noise_samples
                result.snr_db = 10 * np.log10(signal_power / noise_power)
            else:
                result.snr_db = float('inf')

            if result.snr_db < self.min_snr:
                result.low_snr = True
                result.add_reason(f"SNR过低: {result.snr_db:.1f}dB < {self.min_snr}dB")
        except Exception as e:
            result.add_warning(f"SNR估算失败: {str(e)}")

    def _estimate_dynamic_range(self, y: np.ndarray, result: QualityResult):
        """估算动态范围"""
        try:
            # 动态范围 = 峰值 / 噪声底
            # 使用 RMS 的最大值和最小值估算
            frame_length = 2048
            hop_length = 512

            # 计算每帧的 RMS
            rms = librosa.feature.rms(
                y=y, frame_length=frame_length, hop_length=hop_length
            )[0]

            if len(rms) == 0:
                result.dynamic_range_db = 0.0
                return

            # 排除静音帧
            rms_non_silent = rms[rms > 1e-6]
            if len(rms_non_silent) == 0:
                result.dynamic_range_db = 0.0
                return

            peak_rms = np.max(rms_non_silent)
            floor_rms = np.percentile(rms_non_silent, 5)  # 5% 分位数作为噪声底

            if floor_rms > 0:
                result.dynamic_range_db = 20 * np.log10(peak_rms / floor_rms)
            else:
                result.dynamic_range_db = float('inf')

            if result.dynamic_range_db < self.min_dynamic_range:
                result.low_dynamic_range = True
                result.add_reason(
                    f"动态范围过低: {result.dynamic_range_db:.1f}dB < {self.min_dynamic_range}dB"
                )
        except Exception as e:
            result.add_warning(f"动态范围评估失败: {str(e)}")

    def _measure_and_normalize_loudness(
        self, audio_path: str, y: np.ndarray, sr: int,
        result: QualityResult, output_path: Optional[str] = None
    ):
        """测量响度并可选归一化"""
        try:
            import pyloudnorm as pyln

            # 测量响度
            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(y)
            result.loudness_lufs = loudness

            # 测量 True Peak
            # pyloudnorm 不直接提供 True Peak，使用峰值估算
            true_peak = 20 * np.log10(np.max(np.abs(y)) + 1e-10)
            result.true_peak_db = true_peak

            # 如果需要归一化且提供了输出路径
            if output_path and abs(loudness - self.target_lufs) > self.loudness_tolerance:
                # 响度归一化
                loudness_normalized_audio = pyln.normalize.loudness(
                    y, loudness, self.target_lufs
                )

                # 检查归一化后的 True Peak
                normalized_peak = 20 * np.log10(np.max(np.abs(loudness_normalized_audio)) + 1e-10)
                if normalized_peak > self.true_peak_max:
                    # 如果 True Peak 超限，应用峰值限制
                    peak_reduction = self.true_peak_max - normalized_peak
                    loudness_normalized_audio *= 10 ** (peak_reduction / 20)
                    result.add_warning(f"True Peak超限，应用峰值限制: {peak_reduction:.1f}dB")

                # 保存归一化后的音频
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                sf.write(output_path, loudness_normalized_audio, sr)
                result.loudness_normalized = True
                result.add_warning(f"响度已归一化: {loudness:.1f} → {self.target_lufs} LUFS")

        except ImportError:
            result.add_warning("pyloudnorm 未安装，跳过响度测量和归一化")
        except Exception as e:
            result.add_warning(f"响度测量失败: {str(e)}")


class AudioRepairer:
    """
    音频修复器

    可修复的问题：
    - 格式转换：mp3/ogg/m4a → wav/flac
    - 采样率转换：低采样率 → 目标采样率（上采样）
    - 位深转换：统一为 16-bit 或 24-bit
    - 声道转换：单声道 → 立体声（或反之）

    不可修复的问题（直接淘汰）：
    - 文件损坏
    - 静音占比过高
    - SNR 过低
    - 动态范围过低
    """

    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = {}

        # 目标格式
        self.target_format = config.get("target_format", "wav")
        self.target_sample_rate = config.get("target_sample_rate", 44100)
        self.target_bit_depth = config.get("target_bit_depth", 16)
        self.target_channels = config.get("target_channels", 2)

        # 允许的源格式
        self.allowed_source_formats = config.get(
            "allowed_source_formats", ["wav", "flac", "mp3", "ogg", "m4a"]
        )

        logger.info("音频修复器初始化完成")
        logger.info(f"  目标格式: {self.target_format}")
        logger.info(f"  目标采样率: {self.target_sample_rate} Hz")
        logger.info(f"  目标位深: {self.target_bit_depth}-bit")
        logger.info(f"  目标声道: {self.target_channels}")

    def needs_repair(self, audio_path: str) -> Tuple[bool, List[str]]:
        """
        检查音频是否需要修复

        Returns:
            (needs_repair, issues): 是否需要修复，问题列表
        """
        issues = []
        path = Path(audio_path)
        ext = path.suffix.lower().lstrip(".")

        # 检查格式
        if ext != self.target_format and ext in self.allowed_source_formats:
            issues.append(f"format:{ext}→{self.target_format}")

        # 检查采样率和位深（需要加载音频）
        try:
            info = sf.info(audio_path)
            if info.samplerate != self.target_sample_rate:
                issues.append(f"sample_rate:{info.samplerate}→{self.target_sample_rate}")
            if info.channels != self.target_channels:
                issues.append(f"channels:{info.channels}→{self.target_channels}")
        except Exception:
            pass

        return len(issues) > 0, issues

    def repair(self, audio_path: str, output_path: str) -> Tuple[bool, List[str]]:
        """
        修复音频

        Args:
            audio_path: 输入音频路径
            output_path: 输出音频路径

        Returns:
            (success, issues): 是否成功，修复的问题列表
        """
        needs_repair, issues = self.needs_repair(audio_path)
        if not needs_repair:
            return True, []

        logger.info(f"  修复音频: {audio_path}")
        logger.info(f"    问题: {issues}")

        try:
            # 加载音频
            y, sr = librosa.load(audio_path, sr=None, mono=False)

            # 声道处理
            if y.ndim > 1:
                if self.target_channels == 1:
                    y = librosa.to_mono(y)
                elif y.shape[0] == 1 and self.target_channels == 2:
                    # 单声道转立体声（复制通道）
                    y = np.vstack([y, y])
            elif self.target_channels == 2:
                # 单声道转立体声
                y = np.vstack([y, y])

            # 采样率转换
            if sr != self.target_sample_rate:
                if y.ndim > 1:
                    # 多通道分别重采样
                    y_resampled = []
                    for ch in range(y.shape[0]):
                        y_ch = librosa.resample(y[ch], orig_sr=sr, target_sr=self.target_sample_rate)
                        y_resampled.append(y_ch)
                    y = np.array(y_resampled)
                else:
                    y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sample_rate)
                sr = self.target_sample_rate

            # 确定位深的 subtype
            if self.target_bit_depth == 16:
                subtype = "PCM_16"
            elif self.target_bit_depth == 24:
                subtype = "PCM_24"
            elif self.target_bit_depth == 32:
                subtype = "PCM_32"
            else:
                subtype = "PCM_16"

            # 保存
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, y.T if y.ndim > 1 else y, sr, subtype=subtype)

            logger.info(f"    修复完成: {output_path}")
            return True, issues

        except Exception as e:
            logger.error(f"    修复失败: {str(e)}")
            return False, issues


def batch_check(
    audio_paths: List[str],
    config: Optional[Dict] = None,
    output_dir: Optional[str] = None,
    report_csv: Optional[str] = None,
    auto_repair: bool = True,
    repair_config: Optional[Dict] = None,
) -> Tuple[List[QualityResult], pd.DataFrame]:
    """
    批量音频质量检查（支持自动修复）

    Args:
        audio_paths: 音频文件路径列表
        config: 质量检查配置
        output_dir: 响度归一化输出目录（如果为 None 则不输出）
        report_csv: 检查报告 CSV 输出路径
        auto_repair: 是否自动修复可修复的问题（格式/采样率/位深/声道）
        repair_config: 修复配置（来自 cleaning_config.yaml 的 stage2_format）

    Returns:
        (results, report_df): 检查结果列表和报告 DataFrame
    """
    checker = AudioQualityChecker(config)
    results = []

    # 初始化修复器
    repairer = None
    repaired_dir = None
    if auto_repair:
        if repair_config is None:
            repair_config = {}
        repairer = AudioRepairer(repair_config)
        repaired_dir = PROJECT_ROOT / "data" / "00.5_cleaned" / "repaired_audio"
        repaired_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"自动修复已启用，修复后输出到: {repaired_dir}")

    logger.info(f"开始批量质量检查: {len(audio_paths)} 个文件")

    for i, audio_path in enumerate(audio_paths):
        logger.info(f"[{i+1}/{len(audio_paths)}] 检查: {audio_path}")

        # 自动修复
        check_path = audio_path
        repaired = False
        repair_issues = []
        if repairer and repaired_dir:
            needs_repair, issues = repairer.needs_repair(audio_path)
            if needs_repair:
                filename = Path(audio_path).stem + "_repaired.wav"
                repaired_path = str(repaired_dir / filename)
                success, repair_issues = repairer.repair(audio_path, repaired_path)
                if success:
                    check_path = repaired_path
                    repaired = True
                    logger.info(f"  🔧 已修复: {repair_issues}")
                else:
                    logger.warning(f"  ⚠️ 修复失败，使用原文件检查")

        # 响度归一化输出路径
        output_path = None
        if output_dir:
            filename = Path(check_path).stem + "_normalized.wav"
            output_path = str(Path(output_dir) / filename)

        # 质量检查
        result = checker.check(check_path, output_path)

        # 如果是修复后的文件，记录原始路径
        if repaired:
            result.audio_path = audio_path
            result.add_warning(f"已修复后检查: {repair_issues}")

        results.append(result)

        status = "✅ 通过" if result.passed else "❌ 未通过"
        repair_tag = " 🔧修复后" if repaired else ""
        logger.info(f"  {status}{repair_tag} | 时长:{result.duration:.1f}s | "
                    f"SR:{result.sample_rate} | 静音:{result.silence_ratio*100:.1f}% | "
                    f"削波:{result.clipping_ratio*100:.3f}% | SNR:{result.snr_db:.1f}dB | "
                    f"DR:{result.dynamic_range_db:.1f}dB | LUFS:{result.loudness_lufs:.1f}")

        if result.reasons:
            for reason in result.reasons:
                logger.info(f"    原因: {reason}")

    # 生成报告
    report_data = [r.to_dict() for r in results]
    report_df = pd.DataFrame(report_data)

    if report_csv:
        os.makedirs(os.path.dirname(report_csv), exist_ok=True)
        report_df.to_csv(report_csv, index=False, encoding="utf-8")
        logger.info(f"质量检查报告已保存: {report_csv}")

    # 统计
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    repaired_count = sum(1 for r in results if any("已修复" in w for w in r.warnings))
    logger.info(f"批量检查完成: 通过 {passed}/{len(results)}, 未通过 {failed}, 修复 {repaired_count}")

    return results, report_df


if __name__ == "__main__":
    # 测试
    import yaml

    config_path = PROJECT_ROOT / "configs" / "cleaning_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    stage3_config = config.get("stage3_quality", {})

    # 测试单个文件
    test_audio = "/path/to/test.wav"
    if os.path.exists(test_audio):
        checker = AudioQualityChecker(stage3_config)
        result = checker.check(test_audio)
        print(f"通过: {result.passed}")
        print(f"原因: {result.reasons}")
        print(f"警告: {result.warnings}")
    else:
        print("测试文件不存在，跳过单文件测试")
        print("用法: from quality_check import AudioQualityChecker, batch_check")
