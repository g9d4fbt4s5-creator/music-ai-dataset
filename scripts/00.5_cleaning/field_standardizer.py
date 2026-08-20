"""
field_standardizer.py
字段标准化模块（Stage 1 核心功能）

复用已有的 label_mapping_dict.json（v2.0，80个标签）：
- instrument_gm128_map: 乐器 → GM128 标准 MIDI 程序号（33个乐器）
- emotion_vad_map: 情绪 → VAD 三维度（效价/唤醒度/支配度）（16个情绪）
- genre_3level_map: 流派 → 三级分类（15个流派）
- scene_tags: 场景标签（8个）
- quality_tags: 音质标签（4个）
- blacklist_tags: 黑名单标签（4个）

功能：
- 乐器名称标准化（各种别名 → GM128 编号 + 标准名称）
- 情绪标签标准化（各种别名 → VAD 三维度值）
- 流派标签标准化（各种别名 → 三级分类）
- 黑名单检测（命中黑名单标签则标记）

用法：
    from field_standardizer import FieldStandardizer
    standardizer = FieldStandardizer()

    # 乐器标准化
    result = standardizer.standardize_instrument("e piano")
    # -> {"standard_name": "electric piano", "gm128_id": 4, "matched": True}

    # 情绪标准化
    result = standardizer.standardize_emotion("joyful")
    # -> {"standard_name": "happy", "vad": {"valence": 0.85, ...}, "matched": True}

    # 流派标准化
    result = standardizer.standardize_genre("rap")
    # -> {"standard_name": "hip hop", "level1": "嘻哈", "level2": "说唱", "level3": "rap", "matched": True}
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 默认映射字典路径
DEFAULT_MAPPING_PATH = PROJECT_ROOT / "data" / "02_preannotation" / "label_mapping" / "label_mapping_dict.json"

logger = logging.getLogger(__name__)


class FieldStandardizer:
    """字段标准化器"""

    def __init__(self, mapping_path: Optional[str] = None):
        """
        初始化字段标准化器

        Args:
            mapping_path: 映射字典 JSON 路径，默认使用项目中的 label_mapping_dict.json
        """
        if mapping_path is None:
            mapping_path = DEFAULT_MAPPING_PATH

        self.mapping_path = Path(mapping_path)
        self.mapping_data = self._load_mapping()

        # 提取各映射表
        self.instrument_map = self.mapping_data.get("instrument_gm128_map", {})
        self.emotion_map = self.mapping_data.get("emotion_vad_map", {})
        self.genre_map = self.mapping_data.get("genre_3level_map", {})
        self.scene_tags = self.mapping_data.get("scene_tags", [])
        self.quality_tags = self.mapping_data.get("quality_tags", [])
        self.blacklist_tags = self.mapping_data.get("blacklist_tags", [])

        # 构建别名索引（小写，去空格）用于模糊匹配
        self.instrument_alias_index = self._build_alias_index(self.instrument_map.keys())
        self.emotion_alias_index = self._build_alias_index(self.emotion_map.keys())
        self.genre_alias_index = self._build_alias_index(self.genre_map.keys())

        logger.info(f"字段标准化器初始化完成")
        logger.info(f"  映射字典: {self.mapping_path}")
        logger.info(f"  版本: {self.mapping_data.get('version', 'unknown')}")
        logger.info(f"  乐器标签: {len(self.instrument_map)} 个")
        logger.info(f"  情绪标签: {len(self.emotion_map)} 个")
        logger.info(f"  流派标签: {len(self.genre_map)} 个")
        logger.info(f"  场景标签: {len(self.scene_tags)} 个")
        logger.info(f"  音质标签: {len(self.quality_tags)} 个")
        logger.info(f"  黑名单标签: {len(self.blacklist_tags)} 个")

    def _load_mapping(self) -> Dict:
        """加载映射字典"""
        if not self.mapping_path.exists():
            logger.error(f"映射字典不存在: {self.mapping_path}")
            raise FileNotFoundError(f"Mapping file not found: {self.mapping_path}")

        with open(self.mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    def _build_alias_index(self, keys: List[str]) -> Dict[str, str]:
        """
        构建别名索引（小写，去空格，去连字符）用于模糊匹配

        Args:
            keys: 标准标签列表

        Returns:
            {normalized_alias: standard_key} 字典
        """
        index = {}
        for key in keys:
            # 标准化：小写，去空格，去连字符，去下划线
            normalized = self._normalize(key)
            index[normalized] = key

            # 常见别名
            aliases = self._generate_aliases(key)
            for alias in aliases:
                index[self._normalize(alias)] = key

        return index

    def _normalize(self, text: str) -> str:
        """标准化文本：小写，去空格，去连字符，去下划线，去&符号"""
        return text.lower().replace(" ", "").replace("-", "").replace("_", "").replace("&", "").strip()

    def _generate_aliases(self, key: str) -> List[str]:
        """生成常见别名"""
        aliases = []
        key_lower = key.lower()

        # 乐器常见别名
        instrument_aliases = {
            "piano": ["pianoforte", "grand piano", "acoustic piano"],
            "electric piano": ["e piano", "epiano", "electric grand", "rhodes", "wurlitzer"],
            "guitar": ["gtr", "guit"],
            "acoustic guitar": ["acoustic gtr", "folk guitar", "steel guitar"],
            "electric guitar": ["electric gtr", "e guitar", "eguitar", "lead guitar", "rhythm guitar"],
            "bass": ["bass guitar", "bass guitar"],
            "electric bass": ["e bass", "ebass", "bass guitar", "electric bass guitar"],
            "drums": ["drum", "drum set", "drum kit", "percussion drums"],
            "percussion": ["perc", "percussive"],
            "violin": ["fiddle", "violinist"],
            "cello": ["violoncello", "cellist"],
            "flute": ["flautist", "flutist"],
            "saxophone": ["sax", "saxophonist"],
            "trumpet": ["trumpeter", "trumpet player"],
            "vocals": ["vocal", "voice", "singing", "singer", "lead vocal", "lead vocals"],
            "male vocals": ["male vocal", "male voice", "male singer", "man vocal"],
            "female vocals": ["female vocal", "female voice", "female singer", "woman vocal"],
            "choir": ["chorus", "choral", "choir vocals"],
            "organ": ["pipe organ", "church organ", "hammond organ", "hammond"],
            "synth lead": ["synth lead", "synthesizer lead", "lead synth"],
            "synth pad": ["synth pad", "synthesizer pad", "pad synth"],
            "strings": ["string section", "string ensemble", "orchestral strings"],
            "harp": ["harpist"],
            "clarinet": ["clarinetist"],
            "oboe": ["oboist"],
            "trombone": ["trombonist"],
            "tuba": ["tubist"],
            "french horn": ["horn", "frenchhorn", "hornist"],
            "harpsichord": ["harpsichordist", "cembalo"],
            "viola": ["violist"],
            "contrabass": ["double bass", "upright bass", "standup bass"],
            "whistle": ["whistler", "whistling"],
        }

        # 情绪常见别名
        emotion_aliases = {
            "happy": ["joyful", "joy", "cheerful", "glad", "delighted", "pleased", "happiness"],
            "sad": ["sorrowful", "sorrow", "unhappy", "depressed", "melancholy", "gloomy", "sadness"],
            "calm": ["peaceful", "serene", "tranquil", "quiet", "still", "relaxed"],
            "energetic": ["energetic", "energy", "lively", "vigorous", "dynamic", "active"],
            "relaxing": ["relaxed", "chill", "chilled", "mellow", "easygoing"],
            "dark": ["darkness", "gloomy", "ominous", "shadowy"],
            "uplifting": ["uplift", "inspiring", "inspirational", "motivational", "elevating"],
            "tense": ["tension", "anxious", "anxiety", "nervous", "strained", "stressful"],
            "mellow": ["soft", "gentle", "smooth", "laidback"],
            "angry": ["anger", "rage", "furious", "irritated", "mad", "hostile"],
            "fearful": ["fear", "afraid", "scared", "frightened", "terrified", "horror"],
            "surprised": ["surprise", "shocked", "astonished", "amazed", "startled"],
            "nostalgic": ["nostalgia", "wistful", "remembrance", "sentimental"],
            "romantic": ["romance", "love", "loving", "tender", "passionate", "amorous"],
            "mysterious": ["mystery", "enigmatic", "cryptic", "puzzling", "uncanny"],
            "playful": ["play", "fun", "lighthearted", "whimsical", "mischievous", "jolly"],
        }

        # 流派常见别名
        genre_aliases = {
            "pop": ["popular", "pop music", "mainstream pop"],
            "rock": ["rock music", "rock n roll", "rock and roll"],
            "electronic": ["electro", "edm", "electronic music", "dance electronic"],
            "jazz": ["jazz music", "jazzy"],
            "classical": ["classical music", "art music", "orchestral"],
            "hip hop": ["hiphop", "hip-hop", "rap", "rapping", "mc"],
            "R&B": ["rnb", "r&b music", "rhythm and blues", "rhythm & blues"],
            "folk": ["folk music", "folky", "traditional folk"],
            "country": ["country music", "country western", "countrywestern"],
            "blues": ["blues music", "blue"],
            "soul": ["soul music", "southern soul"],
            "funk": ["funk music", "funky"],
            "reggae": ["reggae music", "dub reggae"],
            "ambient": ["ambient music", "atmospheric", "drone music"],
            "experimental": ["experimental music", "avant garde", "avantgarde", "noise music"],
        }

        # 合并所有别名（key转小写）
        all_aliases = {}
        for k, v in instrument_aliases.items():
            all_aliases[k.lower()] = v
        for k, v in emotion_aliases.items():
            all_aliases[k.lower()] = v
        for k, v in genre_aliases.items():
            all_aliases[k.lower()] = v

        if key_lower in all_aliases:
            aliases.extend(all_aliases[key_lower])

        return aliases

    def _fuzzy_match(self, input_text: str, alias_index: Dict[str, str]) -> Optional[str]:
        """
        模糊匹配

        Args:
            input_text: 输入文本
            alias_index: 别名索引

        Returns:
            匹配到的标准标签，未匹配返回 None
        """
        normalized = self._normalize(input_text)

        # 精确匹配
        if normalized in alias_index:
            return alias_index[normalized]

        # 包含匹配（输入包含标准标签，或标准标签包含输入）
        for alias, standard in alias_index.items():
            if normalized in alias or alias in normalized:
                return standard

        return None

    def standardize_instrument(self, input_name: str) -> Dict:
        """
        标准化乐器名称

        Args:
            input_name: 输入乐器名称（各种别名）

        Returns:
            {
                "input": 原始输入,
                "standard_name": 标准名称,
                "gm128_id": GM128 标准 MIDI 程序号,
                "matched": 是否匹配成功
            }
        """
        result = {
            "input": input_name,
            "standard_name": None,
            "gm128_id": None,
            "matched": False,
        }

        if not input_name or not isinstance(input_name, str):
            return result

        matched = self._fuzzy_match(input_name, self.instrument_alias_index)

        if matched:
            result["standard_name"] = matched
            result["gm128_id"] = self.instrument_map[matched]
            result["matched"] = True

        return result

    def standardize_emotion(self, input_name: str) -> Dict:
        """
        标准化情绪标签

        Args:
            input_name: 输入情绪名称（各种别名）

        Returns:
            {
                "input": 原始输入,
                "standard_name": 标准名称,
                "vad": {"valence": 效价, "arousal": 唤醒度, "dominance": 支配度},
                "matched": 是否匹配成功
            }
        """
        result = {
            "input": input_name,
            "standard_name": None,
            "vad": None,
            "matched": False,
        }

        if not input_name or not isinstance(input_name, str):
            return result

        matched = self._fuzzy_match(input_name, self.emotion_alias_index)

        if matched:
            result["standard_name"] = matched
            result["vad"] = self.emotion_map[matched]
            result["matched"] = True

        return result

    def standardize_genre(self, input_name: str) -> Dict:
        """
        标准化流派标签

        Args:
            input_name: 输入流派名称（各种别名）

        Returns:
            {
                "input": 原始输入,
                "standard_name": 标准名称（英文）,
                "level1": 一级分类（中文）,
                "level2": 二级分类（中文）,
                "level3": 三级分类（英文/别名）,
                "matched": 是否匹配成功
            }
        """
        result = {
            "input": input_name,
            "standard_name": None,
            "level1": None,
            "level2": None,
            "level3": None,
            "matched": False,
        }

        if not input_name or not isinstance(input_name, str):
            return result

        matched = self._fuzzy_match(input_name, self.genre_alias_index)

        if matched:
            result["standard_name"] = matched
            levels = self.genre_map[matched]
            if isinstance(levels, list) and len(levels) >= 3:
                result["level1"] = levels[0]
                result["level2"] = levels[1]
                result["level3"] = levels[2]
            elif isinstance(levels, list) and len(levels) == 2:
                result["level1"] = levels[0]
                result["level2"] = levels[1]
            elif isinstance(levels, list) and len(levels) == 1:
                result["level1"] = levels[0]
            result["matched"] = True

        return result

    def is_blacklisted(self, tag: str) -> bool:
        """
        检查标签是否在黑名单中

        Args:
            tag: 待检查标签

        Returns:
            True 如果在黑名单中
        """
        if not tag or not isinstance(tag, str):
            return False

        normalized = self._normalize(tag)
        for blacklist_tag in self.blacklist_tags:
            if normalized == self._normalize(blacklist_tag):
                return True
        return False

    def standardize_row(self, row: Dict, instrument_col: str = "instrument",
                        emotion_col: str = "emotion", genre_col: str = "genre") -> Dict:
        """
        标准化一行数据的所有标签字段

        Args:
            row: 数据行字典
            instrument_col: 乐器列名
            emotion_col: 情绪列名
            genre_col: 流派列名

        Returns:
            标准化后的行字典（新增 _standardized 后缀的列）
        """
        result = row.copy()

        # 乐器标准化
        if instrument_col in row and row[instrument_col]:
            inst_result = self.standardize_instrument(row[instrument_col])
            result[f"{instrument_col}_standard"] = inst_result["standard_name"]
            result[f"{instrument_col}_gm128_id"] = inst_result["gm128_id"]
            result[f"{instrument_col}_matched"] = inst_result["matched"]

        # 情绪标准化
        if emotion_col in row and row[emotion_col]:
            emo_result = self.standardize_emotion(row[emotion_col])
            result[f"{emotion_col}_standard"] = emo_result["standard_name"]
            if emo_result["vad"]:
                result[f"{emotion_col}_valence"] = emo_result["vad"]["valence"]
                result[f"{emotion_col}_arousal"] = emo_result["vad"]["arousal"]
                result[f"{emotion_col}_dominance"] = emo_result["vad"]["dominance"]
            result[f"{emotion_col}_matched"] = emo_result["matched"]

        # 流派标准化
        if genre_col in row and row[genre_col]:
            genre_result = self.standardize_genre(row[genre_col])
            result[f"{genre_col}_standard"] = genre_result["standard_name"]
            result[f"{genre_col}_level1"] = genre_result["level1"]
            result[f"{genre_col}_level2"] = genre_result["level2"]
            result[f"{genre_col}_level3"] = genre_result["level3"]
            result[f"{genre_col}_matched"] = genre_result["matched"]

        return result


if __name__ == "__main__":
    # 测试
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    standardizer = FieldStandardizer()

    print("\n=== 乐器标准化测试 ===")
    test_instruments = ["e piano", "rhodes", "gtr", "sax", "vocal", "unknown instrument"]
    for inst in test_instruments:
        result = standardizer.standardize_instrument(inst)
        std_name = result['standard_name'] or "N/A"
        gm128 = result['gm128_id'] if result['gm128_id'] is not None else "N/A"
        print(f"  {inst:20s} -> {std_name:20s} (GM128: {gm128}, matched: {result['matched']})")

    print("\n=== 情绪标准化测试 ===")
    test_emotions = ["joyful", "depressed", "chill", "rage", "love", "unknown emotion"]
    for emo in test_emotions:
        result = standardizer.standardize_emotion(emo)
        std_name = result['standard_name'] or "N/A"
        vad = result["vad"]
        vad_str = f"V:{vad['valence']:.2f} A:{vad['arousal']:.2f} D:{vad['dominance']:.2f}" if vad else "N/A"
        print(f"  {emo:20s} -> {std_name:15s} ({vad_str}, matched: {result['matched']})")

    print("\n=== 流派标准化测试 ===")
    test_genres = ["rap", "edm", "rnb", "rock n roll", "avant garde", "unknown genre"]
    for genre in test_genres:
        result = standardizer.standardize_genre(genre)
        std_name = result['standard_name'] or "N/A"
        levels = f"{result['level1']}/{result['level2']}/{result['level3']}" if result['level1'] else "N/A"
        print(f"  {genre:20s} -> {std_name:15s} ({levels}, matched: {result['matched']})")

    print("\n=== 黑名单测试 ===")
    test_blacklist = ["low quality", "noisy", "good music", "bad audio"]
    for tag in test_blacklist:
        is_black = standardizer.is_blacklisted(tag)
        print(f"  {tag:20s} -> blacklisted: {is_black}")
