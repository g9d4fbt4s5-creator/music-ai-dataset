"""
pii_remover.py
PII（个人身份信息）正则移除模块（Stage 5 辅助清洗）

功能：
- 手机号、身份证、邮箱、银行卡号、固定电话
- IP地址、URL、MAC地址
- 中文地址（简单匹配）
- 姓名（需配合姓氏词典，可选）

用法：
    from pii_remover import PIIRemover
    remover = PIIRemover()
    cleaned_text = remover.clean(text)
    # 或批量处理 DataFrame 列
    cleaned_df = remover.clean_dataframe(df, columns=['description', 'lyrics'])
"""
import os
import re
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PIIRemovalResult:
    """PII 移除结果"""
    original_text: str
    cleaned_text: str
    removed_count: int = 0
    removed_types: Dict[str, int] = field(default_factory=dict)
    removed_examples: List[Tuple[str, str]] = field(default_factory=list)  # (type, matched_text)

    def to_dict(self) -> Dict:
        return {
            "original_length": len(self.original_text),
            "cleaned_length": len(self.cleaned_text),
            "removed_count": self.removed_count,
            "removed_types": "; ".join(f"{k}:{v}" for k, v in self.removed_types.items()),
            "removed_examples": "; ".join(f"[{t}]{m}" for t, m in self.removed_examples[:10]),
        }


class PIIRemover:
    """PII 正则移除器"""

    # 中文姓氏（常见姓氏，用于姓名匹配）
    CHINESE_SURNAMES = set(
        "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
        "戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐"
        "费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄"
        "和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁"
        "杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍"
        "虞万支柯昝管卢莫经房裘缪干解应宗丁宣贲邓郁单杭洪包诸左石崔吉钮龚"
        "程嵇邢滑裴陆荣翁荀羊於惠甄曲家封芮羿储靳汲邴糜松井段富巫乌焦巴弓"
        "牧隗山谷车侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘钭厉戎祖武符刘景詹束龙"
        "叶幸司韶郜黎蓟薄印宿白怀蒲邰从鄂索咸籍赖卓蔺屠蒙池乔阴郁胥能苍双"
        "闻莘党翟谭贡劳逄姬申扶堵冉宰郦雍却璩桑桂濮牛寿通边扈燕冀郏浦尚农"
        "温别庄晏柴瞿阎充慕连茹习宦艾鱼容向古易慎戈廖庾终暨居衡步都耿满弘"
        "匡国文寇广禄阙东欧殳沃利蔚越夔隆师巩厍聂晁勾敖融冷訾辛阚那简饶空"
        "曾毋沙乜养鞠须丰巢关蒯相查后荆红游竺权逯盖益桓公"
    )

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化 PII 移除器

        Args:
            config: PII 移除配置（来自 cleaning_config.yaml 的 stage5_auxiliary.pii_removal）
        """
        if config is None:
            config = {}

        self.enabled_types = config.get("enabled_types", [
            "phone", "id_card", "email", "bank_card", "landline",
            "ip_address", "url", "mac_address", "chinese_address"
        ])
        self.enable_name_detection = config.get("enable_name_detection", False)  # 姓名检测默认关闭（误报率高）
        self.replacement = config.get("replacement", "[REDACTED]")
        self.keep_examples = config.get("keep_examples", 10)  # 保留多少个移除样例

        # 编译正则
        self._patterns = self._compile_patterns()

        logger.info("PII 移除器初始化完成")
        logger.info(f"  启用类型: {', '.join(self.enabled_types)}")
        logger.info(f"  姓名检测: {'开启' if self.enable_name_detection else '关闭'}")
        logger.info(f"  替换符: {self.replacement}")

    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """编译所有 PII 正则模式"""
        patterns = {}

        # 手机号（中国大陆）
        if "phone" in self.enabled_types:
            patterns["phone"] = re.compile(
                r'(?<!\d)1[3-9]\d{9}(?!\d)'
            )

        # 身份证号（18位，最后一位可为X）
        if "id_card" in self.enabled_types:
            patterns["id_card"] = re.compile(
                r'(?<!\d)\d{17}[\dXx](?!\d)'
            )

        # 邮箱
        if "email" in self.enabled_types:
            patterns["email"] = re.compile(
                r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
            )

        # 银行卡号（16-19位数字）
        if "bank_card" in self.enabled_types:
            patterns["bank_card"] = re.compile(
                r'(?<!\d)\d{16,19}(?!\d)'
            )

        # 固定电话（中国大陆）
        if "landline" in self.enabled_types:
            patterns["landline"] = re.compile(
                r'(?<!\d)0\d{2,3}[-\s]?\d{7,8}(?!\d)'
            )

        # IP 地址（IPv4）
        if "ip_address" in self.enabled_types:
            patterns["ip_address"] = re.compile(
                r'(?<!\d)(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?!\d)'
            )

        # URL
        if "url" in self.enabled_types:
            patterns["url"] = re.compile(
                r'https?://[^\s<>"{}|\\^`\[\]]+'
            )

        # MAC 地址
        if "mac_address" in self.enabled_types:
            patterns["mac_address"] = re.compile(
                r'(?i)(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}'
            )

        # 中文地址（简单匹配：数字+路/街/号/室/栋/单元/巷/弄）
        if "chinese_address" in self.enabled_types:
            patterns["chinese_address"] = re.compile(
                r'\d+[\u4e00-\u9fa5]{0,10}(?:路|街|号|室|栋|单元|巷|弄|大道|大道号)'
                r'(?:\d+[\u4e00-\u9fa5]{0,5})?'
            )

        return patterns

    def clean(self, text: str) -> PIIRemovalResult:
        """
        清理文本中的 PII

        Args:
            text: 原始文本

        Returns:
            PIIRemovalResult: 清理结果
        """
        if not text or not isinstance(text, str):
            return PIIRemovalResult(
                original_text=str(text) if text else "",
                cleaned_text=str(text) if text else "",
            )

        original_text = text
        cleaned_text = text
        removed_count = 0
        removed_types: Dict[str, int] = {}
        removed_examples: List[Tuple[str, str]] = []

        # 逐类替换
        for pii_type, pattern in self._patterns.items():
            matches = pattern.findall(cleaned_text)
            if matches:
                count = len(matches)
                removed_count += count
                removed_types[pii_type] = removed_types.get(pii_type, 0) + count

                # 记录样例
                for match in matches[:3]:
                    if len(removed_examples) < self.keep_examples:
                        removed_examples.append((pii_type, match[:50]))

                # 替换
                cleaned_text = pattern.sub(self.replacement, cleaned_text)

        # 姓名检测（可选，误报率高）
        if self.enable_name_detection:
            name_matches = self._detect_chinese_names(cleaned_text)
            if name_matches:
                count = len(name_matches)
                removed_count += count
                removed_types["chinese_name"] = removed_types.get("chinese_name", 0) + count
                for match in name_matches[:3]:
                    if len(removed_examples) < self.keep_examples:
                        removed_examples.append(("chinese_name", match))
                # 替换姓名（用分词边界，简单替换）
                for name in set(name_matches):
                    cleaned_text = cleaned_text.replace(name, self.replacement)

        return PIIRemovalResult(
            original_text=original_text,
            cleaned_text=cleaned_text,
            removed_count=removed_count,
            removed_types=removed_types,
            removed_examples=removed_examples,
        )

    def _detect_chinese_names(self, text: str) -> List[str]:
        """
        简单的中文姓名检测

        规则：姓氏 + 1-2个汉字，且不在常见词中
        注意：误报率较高，默认关闭
        """
        names = []
        # 匹配：姓氏 + 1-2个汉字
        pattern = re.compile(
            r'([' + ''.join(re.escape(s) for s in self.CHINESE_SURNAMES) + r'])'
            r'[\u4e00-\u9fa5]{1,2}'
            r'(?![\u4e00-\u9fa5])'
        )
        for match in pattern.finditer(text):
            name = match.group()
            # 排除常见非姓名词（简单黑名单）
            non_name_words = {
                "国王", "王子", "公主", "皇后", "皇帝", "将军", "大人", "先生",
                "小姐", "女士", "老师", "同学", "朋友", "家人", "大家", "人们",
            }
            if name not in non_name_words and len(name) >= 2:
                names.append(name)
        return names

    def clean_dataframe(
        self,
        df: pd.DataFrame,
        columns: List[str],
        add_report_columns: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        批量清理 DataFrame 中的 PII

        Args:
            df: 输入 DataFrame
            columns: 需要清理的列名列表
            add_report_columns: 是否添加报告列（_pii_removed_count, _pii_removed_types）

        Returns:
            (cleaned_df, report_df): 清理后的 DataFrame 和报告 DataFrame
        """
        cleaned_df = df.copy()
        report_rows = []

        for idx, row in df.iterrows():
            row_report = {"row_index": idx}
            total_removed = 0
            all_types: Dict[str, int] = {}

            for col in columns:
                if col not in df.columns:
                    continue
                original_val = row[col]
                if pd.isna(original_val) or not isinstance(original_val, str):
                    continue

                result = self.clean(original_val)
                cleaned_df.at[idx, col] = result.cleaned_text

                if result.removed_count > 0:
                    total_removed += result.removed_count
                    for pii_type, count in result.removed_types.items():
                        all_types[pii_type] = all_types.get(pii_type, 0) + count
                    row_report[f"{col}_removed"] = result.removed_count
                    row_report[f"{col}_types"] = "; ".join(
                        f"{k}:{v}" for k, v in result.removed_types.items()
                    )

            row_report["total_removed"] = total_removed
            row_report["all_types"] = "; ".join(f"{k}:{v}" for k, v in all_types.items())
            report_rows.append(row_report)

            if total_removed > 0:
                logger.info(f"  行 {idx}: 移除 {total_removed} 个 PII ({row_report['all_types']})")

        report_df = pd.DataFrame(report_rows)

        # 添加汇总列
        if add_report_columns:
            cleaned_df["_pii_removed_count"] = report_df["total_removed"].values
            cleaned_df["_pii_removed_types"] = report_df["all_types"].values

        logger.info(f"PII 移除完成: {len(df)} 行, 共移除 {sum(r['total_removed'] for r in report_rows)} 个 PII")

        return cleaned_df, report_df


def batch_pii_removal(
    df: pd.DataFrame,
    columns: List[str],
    config: Optional[Dict] = None,
    report_csv: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    批量 PII 移除（便捷函数）

    Args:
        df: 输入 DataFrame
        columns: 需要清理的列名列表
        config: PII 移除配置
        report_csv: 报告输出路径

    Returns:
        (cleaned_df, report_df)
    """
    remover = PIIRemover(config)
    cleaned_df, report_df = remover.clean_dataframe(df, columns)

    if report_csv:
        os.makedirs(os.path.dirname(report_csv), exist_ok=True)
        report_df.to_csv(report_csv, index=False, encoding="utf-8")
        logger.info(f"PII 移除报告已保存: {report_csv}")

    return cleaned_df, report_df


if __name__ == "__main__":
    # 测试
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    test_text = """
    联系人：张三，手机号 13812345678，邮箱 zhangsan@example.com
    身份证号：110101199001011234
    银行卡：6222021234567890123
    固定电话：010-12345678
    IP地址：192.168.1.100
    网址：https://example.com/path?query=1
    MAC地址：AA:BB:CC:DD:EE:FF
    地址：北京市朝阳区建国路88号SOHO现代城A座1203室
    """

    remover = PIIRemover()
    result = remover.clean(test_text)

    print("=== 原始文本 ===")
    print(test_text)
    print("\n=== 清理后 ===")
    print(result.cleaned_text)
    print(f"\n移除数量: {result.removed_count}")
    print(f"移除类型: {result.removed_types}")
    print(f"移除样例: {result.removed_examples}")
