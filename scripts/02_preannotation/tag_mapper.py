#!/usr/bin/env python3
"""
TagMapper — 统一标签映射类

将原始粗标签（文本提取/模型输出）映射为 V4 Label Studio 标准格式：
- 乐器 → GM128 标准编号 (GM001 格式)
- 情绪 → VAD 三元组 (valence, arousal, dominance)
- 流派 → 三级分类 + primary/secondary 拆分
- 黑名单 → 直接过滤

适配现有 label_mapping_dict.json 格式:
- instrument_gm128_map: { "piano": 0, ... }  (数字GM编号)
- emotion_vad_map: { "happy": {"valence":0.85, ...}, ... }
- genre_3level_map: { "jazz": ["爵士","传统爵士","acoustic jazz"], ... }
- blacklist_tags: ["low quality", ...]
- genre_level_bridge: (可选) 歧义标签覆盖

使用:
    from tag_mapper import TagMapper

    mapper = TagMapper("label_mapping_dict.json")

    # 一键映射
    result = mapper.map_all(["piano", "sad", "jazz", "noisy"])
    # result = {
    #   "gm128_instruments": ["GM001"],
    #   "genre_primary": "爵士",
    #   "genre_secondary": ["传统爵士", "acoustic jazz"],
    #   "vad_emotions": [{"valence":0.20, "arousal":0.30, "dominance":0.30}],
    #   "unmapped_original_tags": [],
    #   "blacklist_hit": ["noisy"]
    # }

    # 单独映射
    mapper.map_instrument("piano")      # → "GM001"
    mapper.map_genre("jazz")             # → {"primary": "爵士", "secondary": [...]}
    mapper.map_emotion("sad")            # → {"valence":0.20, ...}
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any


class TagMapper:
    """统一标签映射类，适配 V4 Label Studio 模板"""

    def __init__(self, mapping_path: str):
        """
        加载映射字典。

        Args:
            mapping_path: label_mapping_dict.json 路径
        """
        self.mapping_path = Path(mapping_path)
        if not self.mapping_path.exists():
            raise FileNotFoundError(f"映射字典不存在: {mapping_path}")

        with open(self.mapping_path, "r", encoding="utf-8") as f:
            self.mapping = json.load(f)

        # 兼容字段名（instrument_gm128_map / gm128_instrument_map）
        self.inst_map = self.mapping.get("instrument_gm128_map",
                                          self.mapping.get("gm128_instrument_map", {}))
        self.emotion_map = self.mapping.get("emotion_vad_map",
                                             self.mapping.get("vad_emotion_map", {}))
        self.genre_map = self.mapping.get("genre_3level_map", {})
        self.blacklist = set(self.mapping.get("blacklist_tags", []))
        self.bridge = self.mapping.get("genre_level_bridge", {})

        # 版本信息
        self.version = self.mapping.get("version", "v1.0")
        self.compatible_versions = self.mapping.get("compatible_preannotation_versions", [])

    def _gm_number_to_code(self, gm_num: int) -> str:
        """将 GM 数字编号转为 GM001 格式"""
        if isinstance(gm_num, str):
            return gm_num  # 已经是字符串格式
        return f"GM{gm_num + 1:03d}"  # GM编号从0开始，显示从1开始

    def map_instrument(self, raw_tag: str) -> Optional[str]:
        """
        映射乐器原始标签 → GM128 标准编号。

        Args:
            raw_tag: 原始乐器标签（如 "piano", "electric guitar"）

        Returns:
            GM编号字符串（如 "GM001"），未映射返回 None
        """
        if not raw_tag:
            return None
        key = raw_tag.lower().strip()
        if key in self.inst_map:
            return self._gm_number_to_code(self.inst_map[key])
        return None

    def map_emotion(self, raw_tag: str) -> Optional[Dict[str, float]]:
        """
        映射情绪原始标签 → VAD 三元组。

        Args:
            raw_tag: 原始情绪标签（如 "happy", "sad"）

        Returns:
            VAD字典 {"valence": float, "arousal": float, "dominance": float}
            未映射返回 None
        """
        if not raw_tag:
            return None
        key = raw_tag.lower().strip()
        if key in self.emotion_map:
            vad = self.emotion_map[key]
            # 兼容字典和列表两种格式
            if isinstance(vad, dict):
                return vad
            elif isinstance(vad, (list, tuple)) and len(vad) >= 3:
                return {"valence": vad[0], "arousal": vad[1], "dominance": vad[2]}
        return None

    def map_genre(self, raw_tag: str) -> Dict[str, Any]:
        """
        映射流派原始标签 → V4 primary + secondary。

        规则:
        1. 查 genre_3level_map 得到三级列表
        2. 查 genre_level_bridge 歧义覆盖（如有）
        3. 默认: primary = level3[0], secondary = level3[1:]

        Args:
            raw_tag: 原始流派标签（如 "jazz", "bebop"）

        Returns:
            {"primary": str, "secondary": list, "full_path": list, "unmapped": str|None}
        """
        if not raw_tag:
            return {"primary": None, "secondary": [], "full_path": [], "unmapped": raw_tag}

        key = raw_tag.lower().strip()
        level3 = self.genre_map.get(key)

        if not level3:
            return {"primary": None, "secondary": [], "full_path": [], "unmapped": raw_tag}

        # 歧义覆盖
        if key in self.bridge:
            b = self.bridge[key]
            return {
                "primary": b.get("primary", level3[0] if level3 else raw_tag),
                "secondary": [b.get("secondary", raw_tag)],
                "full_path": level3,
                "unmapped": None,
            }

        # 默认规则: primary = 一级, secondary = 二级+三级
        primary = level3[0] if level3 else raw_tag
        secondary = level3[1:] if len(level3) > 1 else []

        return {
            "primary": primary,
            "secondary": secondary,
            "full_path": level3,
            "unmapped": None,
        }

    def is_blacklisted(self, raw_tag: str) -> bool:
        """检查标签是否在黑名单中"""
        if not raw_tag:
            return False
        return raw_tag.lower().strip() in self.blacklist

    def map_all(self, raw_tags: List[str]) -> Dict[str, Any]:
        """
        一键映射所有维度的原始标签。

        Args:
            raw_tags: 原始标签列表（如 ["piano", "sad", "jazz", "noisy"]）

        Returns:
            {
                "raw_tags": 原始标签列表,
                "gm128_instruments": ["GM001", ...],
                "genre_primary": "爵士" | None,
                "genre_secondary": ["传统爵士", ...],
                "vad_emotions": [{"valence":..., "arousal":..., "dominance":...}, ...],
                "unmapped_original_tags": ["未映射的标签", ...],
                "blacklist_hit": ["命中黑名单的标签", ...],
                "mapping_version": "v2.0"
            }
        """
        result = {
            "raw_tags": list(raw_tags),
            "gm128_instruments": [],
            "genre_primary": None,
            "genre_secondary": [],
            "vad_emotions": [],
            "unmapped_original_tags": [],
            "blacklist_hit": [],
            "mapping_version": self.version,
        }

        seen_instruments = set()
        seen_genres = set()
        seen_emotions = set()

        for tag in raw_tags:
            if not tag:
                continue

            # 黑名单优先
            if self.is_blacklisted(tag):
                result["blacklist_hit"].append(tag)
                continue

            # 乐器
            inst = self.map_instrument(tag)
            if inst and inst not in seen_instruments:
                result["gm128_instruments"].append(inst)
                seen_instruments.add(inst)
                continue

            # 流派
            genre = self.map_genre(tag)
            if genre["primary"] and tag.lower() not in seen_genres:
                result["genre_primary"] = genre["primary"]
                for sec in genre["secondary"]:
                    if sec not in result["genre_secondary"]:
                        result["genre_secondary"].append(sec)
                seen_genres.add(tag.lower())
                continue

            # 情绪
            emotion = self.map_emotion(tag)
            if emotion and tag.lower() not in seen_emotions:
                result["vad_emotions"].append(emotion)
                seen_emotions.add(tag.lower())
                continue

            # 未映射
            if tag not in result["unmapped_original_tags"]:
                result["unmapped_original_tags"].append(tag)

        return result

    def get_unmapped_stats(self, all_raw_tags: List[List[str]]) -> Dict[str, int]:
        """
        统计所有未映射标签的出现频次。

        Args:
            all_raw_tags: 多条样本的原始标签列表

        Returns:
            {"unmapped_tag": 出现次数, ...} 按频次降序
        """
        from collections import Counter
        counter = Counter()

        for tags in all_raw_tags:
            result = self.map_all(tags)
            for tag in result["unmapped_original_tags"]:
                counter[tag] += 1

        return dict(counter.most_common())

    def to_v4_predictions(self, mapped_result: Dict, audio_id: str,
                           duration: float = 0.0) -> List[Dict]:
        """
        将映射结果转为 V4 Label Studio predictions 格式。

        Args:
            mapped_result: map_all() 的输出
            audio_id: 音频ID
            duration: 音频时长（秒）

        Returns:
            Label Studio predictions result 列表
        """
        predictions = []

        # 流派主标签
        if mapped_result.get("genre_primary"):
            predictions.append({
                "id": f"genre_p_{audio_id}",
                "type": "choices",
                "from_name": "genre_primary",
                "to_name": "audio_source",
                "value": {"choices": [mapped_result["genre_primary"]]},
            })

        # 流派次标签
        if mapped_result.get("genre_secondary"):
            predictions.append({
                "id": f"genre_s_{audio_id}",
                "type": "choices",
                "from_name": "genre_secondary",
                "to_name": "audio_source",
                "value": {"choices": mapped_result["genre_secondary"][:3]},
            })

        # 乐器（整首区间）
        if mapped_result.get("gm128_instruments"):
            predictions.append({
                "id": f"inst_{audio_id}",
                "type": "labels",
                "from_name": "instruments",
                "to_name": "audio_source",
                "value": {
                    "start": 0,
                    "end": duration or 180,
                    "labels": mapped_result["gm128_instruments"][:5],
                },
            })

        # 情绪（转为V4情绪标签）
        if mapped_result.get("vad_emotions"):
            # VAD → V4情绪标签的简单映射
            mood_labels = []
            for vad in mapped_result["vad_emotions"]:
                v = vad.get("valence", 0.5)
                a = vad.get("arousal", 0.5)
                if v > 0.6 and a > 0.6:
                    mood_labels.append("欢快活泼 Joyful")
                elif v > 0.6 and a < 0.4:
                    mood_labels.append("温柔舒缓 Calm")
                elif v < 0.4 and a > 0.6:
                    mood_labels.append("激昂热血 Intense")
                elif v < 0.4 and a < 0.4:
                    mood_labels.append("忧郁伤感 Melancholic")
                else:
                    mood_labels.append("神秘 Mysterious")
            if mood_labels:
                predictions.append({
                    "id": f"mood_{audio_id}",
                    "type": "choices",
                    "from_name": "mood",
                    "to_name": "audio_source",
                    "value": {"choices": list(set(mood_labels))[:3]},
                })

        return predictions


# ========== 使用示例 ==========

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python tag_mapper.py <mapping_dict_path> [tag1 tag2 ...]")
        print("示例: python tag_mapper.py label_mapping_dict.json piano sad jazz noisy")
        sys.exit(1)

    mapping_path = sys.argv[1]
    tags = sys.argv[2:] if len(sys.argv) > 2 else ["piano", "sad", "jazz", "noisy"]

    mapper = TagMapper(mapping_path)
    print(f"映射字典版本: {mapper.version}")
    print(f"乐器映射数: {len(mapper.inst_map)}")
    print(f"情绪映射数: {len(mapper.emotion_map)}")
    print(f"流派映射数: {len(mapper.genre_map)}")
    print(f"黑名单数: {len(mapper.blacklist)}")
    print(f"歧义桥接数: {len(mapper.bridge)}")
    print()

    result = mapper.map_all(tags)
    print(f"输入标签: {tags}")
    print(f"映射结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
