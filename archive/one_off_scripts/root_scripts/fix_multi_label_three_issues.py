#!/usr/bin/env python3
"""
修复多标签结构的三个问题：
1. 把用户对4首真冲突+Love Me Tender的裁决写进genres
2. 删除primary_genre字段（多标签并存，不变相单选）
3. 已裁决的样本标记locked，只有未裁决的真冲突才human_review_pending
"""
import json, os

PROJECT_ROOT = "/Users/m.jian/music_corpus_project"
final = json.load(open(f"{PROJECT_ROOT}/data/02_preannotation/genre_unified_final.json"))

# === 用户裁决 ===
# Love Me Tender: Classical Crossover更贴切，两个都对
# How to count ×2: Post-Rock，Ambient不采信(rejected)
# Welcome t7: Dream House和Trance都保留，兼容不冲突
# Ouverture: Italo House和Trance都保留，兼容不冲突

USER_RULINGS = {
    "146D04DCF3BB4D56B68F9DD25C": {  # Love Me Tender
        "ruling": "用户认定Classical Crossover更贴切，Traditional Pop也对，两者并存",
        "labels": [
            {"label": "Classical Crossover", "source": "text_llm_search", "source_priority": "P0",
             "weight": 1.0, "confidence": 0.95, "user_ruling": "用户认定更贴切"},
            {"label": "Traditional Pop", "source": "qwen_omni", "source_priority": "P1",
             "weight": 0.8, "confidence": 0.92, "user_ruling": "用户认定也对"},
        ]
    },
    "2996876392774F67887DA90C3E": {  # How to count at the same time
        "ruling": "用户认定Post-Rock，Ambient低置信文件名推断不采信",
        "labels": [
            {"label": "Ambient", "source": "text_llm_search", "source_priority": "P0",
             "weight": 0.0, "confidence": 0.6, "rejected": True,
             "reject_reason": "低置信文件名推断，不采信；用户认定以Qwen Post-Rock为准"},
            {"label": "Post-Rock", "source": "qwen_omni", "source_priority": "P1",
             "weight": 0.8, "confidence": 0.88, "user_ruling": "用户认定"},
            {"label": "Experimental", "source": "text_llm_search", "source_priority": "P0",
             "weight": 0.7, "confidence": 0.48, "tag_type": "subgenre"},
            {"label": "Drone", "source": "text_llm_search", "source_priority": "P0",
             "weight": 0.7, "confidence": 0.48, "tag_type": "subgenre"},
        ]
    },
    "AA089CC166EF4DE8AA78014AD9": {  # How to count windmill
        "ruling": "用户认定Post-Rock，Ambient低置信文件名推断不采信",
        "labels": [
            {"label": "Ambient", "source": "text_llm_search", "source_priority": "P0",
             "weight": 0.0, "confidence": 0.6, "rejected": True,
             "reject_reason": "低置信文件名推断，不采信；用户认定以Qwen Post-Rock为准"},
            {"label": "Post-Rock", "source": "qwen_omni", "source_priority": "P1",
             "weight": 0.8, "confidence": 0.9, "user_ruling": "用户认定"},
            {"label": "Experimental", "source": "text_llm_search", "source_priority": "P0",
             "weight": 0.7, "confidence": 0.48, "tag_type": "subgenre"},
            {"label": "Drone", "source": "text_llm_search", "source_priority": "P0",
             "weight": 0.7, "confidence": 0.48, "tag_type": "subgenre"},
        ]
    },
    "818C48C49AE547DCBC6D3D9B43": {  # Welcome To Paradise track 7
        "ruling": "用户认定Dream House和Trance兼容不冲突，两者并存",
        "labels": [
            {"label": "Dream House", "source": "text_llm_search", "source_priority": "P0",
             "weight": 1.0, "confidence": 0.85, "user_ruling": "用户认定兼容不冲突"},
            {"label": "Trance", "source": "qwen_omni", "source_priority": "P1",
             "weight": 0.8, "confidence": 0.85, "user_ruling": "用户认定兼容不冲突"},
            {"label": "Italo House", "source": "text_llm_search", "source_priority": "P0",
             "weight": 0.7, "confidence": 0.8, "tag_type": "subgenre"},
            {"label": "Deep House", "source": "text_llm_search", "source_priority": "P0",
             "weight": 0.7, "confidence": 0.8, "tag_type": "subgenre"},
            {"label": "Progressive House", "source": "text_llm_search", "source_priority": "P0",
             "weight": 0.7, "confidence": 0.8, "tag_type": "subgenre"},
        ]
    },
    "DC92B30BB51D42488F02AA203F": {  # Ouverture (Don Carlos)
        "ruling": "用户认定Italo House和Trance兼容不冲突，两者并存",
        "labels": [
            {"label": "Italo House", "source": "text_llm_search", "source_priority": "P0",
             "weight": 1.0, "confidence": 0.85, "user_ruling": "用户认定兼容不冲突"},
            {"label": "Trance", "source": "qwen_omni", "source_priority": "P1",
             "weight": 0.8, "confidence": 0.85, "user_ruling": "用户认定兼容不冲突"},
        ]
    },
}

# === 应用裁决 ===
print("应用用户裁决（5首）...")
for aid, ruling in USER_RULINGS.items():
    if aid in final:
        final[aid]['genres'] = ruling['labels']
        final[aid]['user_ruling'] = ruling['ruling']
        final[aid]['label_selection'] = 'locked'
        final[aid]['ruled_by'] = 'user'
        print(f"  ✅ {final[aid]['title']}: {ruling['ruling'][:40]}...")

# === 删除primary_genre，统一label_selection ===
print("\n删除primary_genre字段，统一label_selection...")
pending_count = 0
locked_count = 0

for aid, v in final.items():
    # 删除primary_genre（多标签并存，不变相单选）
    if 'primary_genre' in v:
        del v['primary_genre']
    
    # label_selection：已裁决的locked，只有未裁决的真冲突才pending
    # 用户已经裁决了所有真冲突（5首），所以全部locked
    if v.get('label_selection') == 'human_review_pending' and aid not in USER_RULINGS:
        # 检查是否真的有冲突（多个来源标签不一致且未裁决）
        accepted = [g for g in v.get('genres', []) if not g.get('rejected', False)]
        sources = set(g['source'] for g in accepted)
        if len(sources) > 1:
            # 多来源但用户未明确裁决 → 保持pending（但这类应该很少）
            pending_count += 1
        else:
            v['label_selection'] = 'locked'
            locked_count += 1
    else:
        if v.get('label_selection') != 'locked':
            v['label_selection'] = 'locked'
        locked_count += 1

print(f"  locked: {locked_count}")
print(f"  human_review_pending: {pending_count}")

# === 保存 ===
with open(f"{PROJECT_ROOT}/data/02_preannotation/genre_unified_final.json", 'w') as f:
    json.dump(final, f, ensure_ascii=False, indent=2)
print(f"\ngenre_unified_final.json 已更新")

# === 更新l4_unified产物 ===
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
    # 不设primary_genre
    if 'primary_genre' in tag:
        del tag['primary_genre']
    tag['sub_genres'] = v.get('sub_genres', [])
    tag['source'] = v.get('source', '')
    tag['source_priority'] = v.get('source_priority', '')
    tag['confidence'] = v.get('confidence', 0)
    tag['title'] = v.get('title', '')
    tag['artist'] = v.get('artist', '')
    tag['reference'] = v.get('reference', '')
    tag['knn_propagated'] = False
    tag['l4_version'] = 'unified_v3_multi_label_no_primary'
    tag['multi_label'] = True
    tag['label_selection'] = v.get('label_selection', 'locked')
    tag['deprecated_knn_note'] = 'KNN传播已从L4移除，一致率0%，详见archive/l4_knn_legacy/DEPRECATED.md'
    
    if 'user_ruling' in v:
        tag['user_ruling'] = v['user_ruling']
        tag['ruled_by'] = 'user'
    if 'resolution' in v:
        tag['resolution'] = v['resolution']
        tag['resolution_note'] = v['resolution_note']
    if 'reannotated' in v:
        tag['reannotated'] = v['reannotated']
    if 'segmented' in v:
        tag['segmented'] = v['segmented']
    if v.get('source_priority') == 'EXCLUDED':
        tag['excluded'] = True
        tag['exclude_reason'] = 'ACE Studio generated (demucs_vocals)'
    
    with open(tag_file, 'w') as f:
        json.dump(tag, f, ensure_ascii=False, indent=2)

print(f"  l4_unified/ 已更新（{len(os.listdir(output_dir))}个文件）")

# === 验证 ===
print("\n" + "=" * 60)
print("验证（WorkBuddy三件事）")
print("=" * 60)

# ① 用户裁决有没有写进去
print("\n① 用户裁决写入验证:")
for aid, ruling in USER_RULINGS.items():
    v = final[aid]
    has_user_ruling = any('user_ruling' in g for g in v['genres'])
    print(f"  {'✅' if has_user_ruling else '❌'} {v['title']}: user_ruling={'有' if has_user_ruling else '无'}")

# ② 有没有primary_genre
print("\n② primary_genre字段验证:")
has_primary = sum(1 for v in final.values() if 'primary_genre' in v)
print(f"  {'❌' if has_primary > 0 else '✅'} primary_genre残留: {has_primary}个（应为0）")

# ③ 84首是不是全标pending
print("\n③ label_sel验证:")
pending = sum(1 for v in final.values() if v.get('label_selection') == 'human_review_pending')
locked = sum(1 for v in final.values() if v.get('label_selection') == 'locked')
print(f"  human_review_pending: {pending}首")
print(f"  locked: {locked}首")
print(f"  {'✅' if pending < 84 else '❌'} 不是84首全pending")

# Love Me Tender示例
print("\n=== Love Me Tender 多标签示例（无primary_genre）===")
lmt = final["146D04DCF3BB4D56B68F9DD25C"]
print(f"  label_selection: {lmt['label_selection']}")
print(f"  user_ruling: {lmt.get('user_ruling', '无')}")
for g in lmt['genres']:
    ruling = g.get('user_ruling', '')
    print(f"  - {g['label']} (source={g['source']}, weight={g['weight']}) {ruling}")
