#!/usr/bin/env python3
"""
将genre统一标注从单选改为多标签并存。
每首的genres字段为列表，每个标签含label/source/source_priority/weight/confidence。
P0文本LLM weight=1.0，P1 Qwen weight=0.8。
标签之间不裁决，全保留。
How to count两首：文本LLM Ambient标为低置信推断不采信(weight=0, rejected=True)，Qwen Post-Rock为准。
"""
import json, glob, os
from collections import Counter

PROJECT_ROOT = "/Users/m.jian/music_corpus_project"

# 加载数据
final = json.load(open(f"{PROJECT_ROOT}/data/02_preannotation/genre_unified_final.json"))
text_llm = json.load(open(f"{PROJECT_ROOT}/data/02_preannotation/genre_text_llm_annotations.json"))

qwen_data = {}
for f in glob.glob(f"{PROJECT_ROOT}/data/02_preannotation/l4_deepseek/*_text_labels.json"):
    d = json.load(open(f))
    aid = f.split('/')[-1].replace('_text_labels.json', '')
    qwen_data[aid] = d

# 同时从l3_structural取Qwen精标（黄金集5首），优先级高于l4_deepseek里的deepseek旧标签
qwen_golden = {}
for f in glob.glob(f"{PROJECT_ROOT}/data/02_preannotation/l3_structural/*_l3_qwen.json"):
    d = json.load(open(f))
    aid = f.split('/')[-1].replace('_l3_qwen.json', '')
    ann = d.get('annotation', d)
    qwen_golden[aid] = ann

# 合并：l3_structural的Qwen精标覆盖l4_deepseek里的deepseek旧标签
for aid, ann in qwen_golden.items():
    if 'qwen' in str(ann.get('source', '')).lower() or aid in qwen_data:
        qwen_data[aid] = ann
        print(f"  用l3_structural Qwen精标覆盖: {aid} -> {ann.get('genre', '?')}")

# How to count两首的特殊处理：文本LLM Ambient低置信不采信
HOW_TO_COUNT_IDS = [
    "2996876392774F67887DA90C3E",  # at the same time
    "AA089CC166EF4DE8AA78014AD9",  # windmill
]

def build_genres(aid, v):
    """构建多标签列表"""
    genres = []
    sp = v.get('source_priority', '')
    
    # P0: 文本LLM标签
    if sp == 'P0' and aid in text_llm:
        tl = text_llm[aid]
        primary = tl.get('primary_genre', '')
        conf = tl.get('confidence', 0.9)
        if primary:
            # How to count两首：文本LLM Ambient低置信不采信
            if aid in HOW_TO_COUNT_IDS and primary.lower() == 'ambient':
                genres.append({
                    "label": primary,
                    "source": "text_llm_search",
                    "source_priority": "P0",
                    "weight": 0.0,
                    "confidence": conf,
                    "rejected": True,
                    "reject_reason": "低置信文件名推断，不采信；以Qwen听音频结果为准",
                    "note": "保留原推断记录供人看"
                })
            else:
                genres.append({
                    "label": primary,
                    "source": "text_llm_search",
                    "source_priority": "P0",
                    "weight": 1.0,
                    "confidence": conf
                })
        # sub_genres也作为候选标签（权重稍低）
        for sg in tl.get('sub_genres', []):
            if sg and sg != primary:
                genres.append({
                    "label": sg,
                    "source": "text_llm_search",
                    "source_priority": "P0",
                    "weight": 0.7,
                    "confidence": conf * 0.8,
                    "tag_type": "subgenre"
                })
    
    # P1: Qwen标签
    if sp == 'P1' and aid in qwen_data:
        qd = qwen_data[aid]
        qgenre = qd.get('genre', '')
        qsub = qd.get('subgenre', '')
        qconf = qd.get('confidence', 0.8)
        if qgenre:
            genres.append({
                "label": qgenre,
                "source": "qwen_omni",
                "source_priority": "P1",
                "weight": 0.8,
                "confidence": qconf
            })
        if qsub:
            for s in qsub.split('/'):
                s = s.strip()
                if s and s != qgenre:
                    genres.append({
                        "label": s,
                        "source": "qwen_omni",
                        "source_priority": "P1",
                        "weight": 0.6,
                        "confidence": qconf * 0.8,
                        "tag_type": "subgenre"
                    })
    
    # 冲突样本：同时有文本LLM和Qwen标签，都保留
    if v.get('resolution') in ['category1_text_llm_primary_qwen_subgenre_merged',
                                  'category2_text_llm_primary_qwen_as_subgenre',
                                  'pending_human_review']:
        # 文本LLM主标签
        if aid in text_llm:
            tl = text_llm[aid]
            primary = tl.get('primary_genre', '')
            conf = tl.get('confidence', 0.9)
            if primary and not any(g['label'] == primary and g['source'] == 'text_llm_search' for g in genres):
                if aid in HOW_TO_COUNT_IDS and primary.lower() == 'ambient':
                    genres.append({
                        "label": primary,
                        "source": "text_llm_search",
                        "source_priority": "P0",
                        "weight": 0.0,
                        "confidence": conf,
                        "rejected": True,
                        "reject_reason": "低置信文件名推断，不采信；以Qwen听音频结果为准",
                        "note": "保留原推断记录供人看"
                    })
                else:
                    genres.append({
                        "label": primary,
                        "source": "text_llm_search",
                        "source_priority": "P0",
                        "weight": 1.0,
                        "confidence": conf
                    })
        # Qwen主标签
        if aid in qwen_data:
            qd = qwen_data[aid]
            qgenre = qd.get('genre', '')
            qconf = qd.get('confidence', 0.8)
            if qgenre and not any(g['label'] == qgenre and g['source'] == 'qwen_omni' for g in genres):
                genres.append({
                    "label": qgenre,
                    "source": "qwen_omni",
                    "source_priority": "P1",
                    "weight": 0.8,
                    "confidence": qconf
                })
    
    # 去重（同source同label只保留一个）
    seen = set()
    unique = []
    for g in genres:
        key = (g['source'], g['label'])
        if key not in seen:
            seen.add(key)
            unique.append(g)
    
    return unique


# 更新所有样本
print("转换为多标签结构...")
multi_label_count = 0
single_label_count = 0

for aid, v in final.items():
    genres = build_genres(aid, v)
    
    # 最高权重标签作为primary（兼容字段）
    accepted = [g for g in genres if not g.get('rejected', False)]
    if accepted:
        primary = max(accepted, key=lambda g: g['weight'])['label']
    else:
        primary = ''
    
    v['genres'] = genres
    v['primary_genre'] = primary  # 兼容字段，最高权重
    v['multi_label'] = True
    v['label_selection'] = "human_review_pending"  # 待Label Studio人工审核选定
    
    if len(accepted) > 1:
        multi_label_count += 1
    else:
        single_label_count += 1

# 保存
with open(f"{PROJECT_ROOT}/data/02_preannotation/genre_unified_final.json", 'w') as f:
    json.dump(final, f, ensure_ascii=False, indent=2)

print(f"  多标签样本: {multi_label_count}")
print(f"  单标签样本: {single_label_count}")
print(f"  总计: {len(final)}")

# 更新l4_unified产物
print("\n更新l4_unified产物...")
output_dir = f"{PROJECT_ROOT}/data/02_preannotation/l4_unified"
for aid, v in final.items():
    tag_file = f"{output_dir}/{aid}_unified_tags.json"
    if os.path.exists(tag_file):
        tag = json.load(open(tag_file))
    else:
        tag = {}
    
    tag['audio_id'] = aid
    tag['genres'] = v['genres']
    tag['primary_genre'] = v['primary_genre']  # 兼容字段
    tag['sub_genres'] = v.get('sub_genres', [])  # 兼容字段
    tag['source'] = v.get('source', '')
    tag['source_priority'] = v.get('source_priority', '')
    tag['confidence'] = v.get('confidence', 0)
    tag['title'] = v.get('title', '')
    tag['artist'] = v.get('artist', '')
    tag['reference'] = v.get('reference', '')
    tag['knn_propagated'] = False
    tag['l4_version'] = 'unified_v3_multi_label'
    tag['multi_label'] = True
    tag['label_selection'] = 'human_review_pending'
    tag['deprecated_knn_note'] = 'KNN传播已从L4移除，一致率0%，详见archive/l4_knn_legacy/DEPRECATED.md'
    
    # 保留裁决信息
    if 'resolution' in v:
        tag['resolution'] = v['resolution']
        tag['resolution_note'] = v['resolution_note']
    if 'candidate_labels' in v:
        tag['candidate_labels'] = v['candidate_labels']
    if 'reannotated' in v:
        tag['reannotated'] = v['reannotated']
        tag['reannotated_from'] = v.get('reannotated_from', '')
    if 'segmented' in v:
        tag['segmented'] = v['segmented']
    if v.get('source_priority') == 'EXCLUDED':
        tag['excluded'] = True
        tag['exclude_reason'] = 'ACE Studio generated (demucs_vocals)'
    
    with open(tag_file, 'w') as f:
        json.dump(tag, f, ensure_ascii=False, indent=2)

print(f"  l4_unified/ 已更新（{len(os.listdir(output_dir))}个文件）")

# 验证How to count两首
print("\n=== 验证How to count两首（Ambient低置信不采信）===")
for aid in HOW_TO_COUNT_IDS:
    v = final[aid]
    print(f"\n{v['title']}:")
    for g in v['genres']:
        status = "❌ 不采信" if g.get('rejected') else "✅ 保留"
        print(f"  {status} {g['label']} (source={g['source']}, weight={g['weight']}, conf={g['confidence']})")
        if g.get('reject_reason'):
            print(f"    原因: {g['reject_reason']}")

# 统计标签来源分布
print("\n=== 标签来源分布（所有genres条目）===")
source_dist = Counter()
for v in final.values():
    for g in v['genres']:
        source_dist[g['source']] += 1
for src, cnt in source_dist.most_common():
    print(f"  {src}: {cnt}个标签")

# 多标签样本示例
print("\n=== 多标签样本示例（Love Me Tender）===")
lmt = final.get("146D04DCF3BB4D56B68F9DD25C", {})
for g in lmt.get('genres', []):
    print(f"  {g['label']} (source={g['source']}, weight={g['weight']}, conf={g['confidence']})")
