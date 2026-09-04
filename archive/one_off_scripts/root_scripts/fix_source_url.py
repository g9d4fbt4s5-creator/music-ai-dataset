#!/usr/bin/env python3
"""为58首文本LLM标注补充真实source_url"""
import json
from collections import Counter

text_llm = json.load(open('data/02_preannotation/genre_text_llm_annotations.json'))

# source_url映射：audio_id -> {source_url, source_type}
source_url_map = {
    # Carpenters
    "8DDE7F1A99A0441C9A1564A6D7": ("https://en.wikipedia.org/wiki/Yesterday_Once_More_(song)", "wikipedia"),
    "3C6F3F3C4839423DB83F7A55CD": ("https://en.wikipedia.org/wiki/(They_Long_to_Be)_Close_to_You", "wikipedia"),
    "A7E56197254D451CB98808C9BD": ("https://en.wikipedia.org/wiki/I_Just_Fall_in_Love_Again", "wikipedia"),
    # La La Land
    "1EDDB5EFB16E4A36AFFFAC2132": ("https://en.wikipedia.org/wiki/Another_Day_of_Sun", "wikipedia"),
    "364CBB655E7D4A90B83A060A39": ("https://en.wikipedia.org/wiki/City_of_Stars", "wikipedia"),
    # Hiromi
    "5CF227BBD8FB42BDA25C4755A1": ("https://www.discogs.com/search?q=Hiromi+Deamer", "discogs_search"),
    "78250AD45D2842A3AA18C9071D": ("https://www.discogs.com/search?q=Hiromi+Seeker", "discogs_search"),
    "D0CC3C06BCC643C4AD0737115C": ("https://www.discogs.com/search?q=Hiromi+Warrior", "discogs_search"),
    # Nelly
    "A1AD6E948D2149EC940866F9D3": ("https://en.wikipedia.org/wiki/Dilemma_(Nelly_song)", "wikipedia"),
    # Kiana Lede
    "F7E5F56ECCCC479589A26A2A10": ("https://en.wikipedia.org/wiki/Ex_(Kiana_Led%C3%A9_song)", "wikipedia"),
    # Silk Sonic
    "4AD0681C6347477D8B1E849422": ("https://en.wikipedia.org/wiki/Leave_the_Door_Open_(Silk_Sonic_song)", "wikipedia"),
    # Jobim
    "33EA926BB2D64917AF8A166779": ("https://en.wikipedia.org/wiki/Wave_(Ant%C3%B4nio_Carlos_Jobim_album)", "wikipedia"),
    "A0F9B5560A194DE5A25A22BD62": ("https://www.discogs.com/search?q=Antonio+Carlos+Jobim+The+Red+Blouse", "discogs_search"),
    "D2098BC896D6450A8E3744528E": ("https://www.discogs.com/search?q=Antonio+Carlos+Jobim+Look+To+The+Sky", "discogs_search"),
    "3D71C975F9D24E4CB5B77D85FA": ("https://en.wikipedia.org/wiki/Triste_(song)", "wikipedia"),
    # Bocelli
    "146D04DCF3BB4D56B68F9DD25C": ("https://en.wikipedia.org/wiki/Love_Me_Tender_(song)#Andrea_Bocelli_version", "wikipedia"),
    # Don Carlos
    "DC92B30BB51D42488F02AA203F": ("https://www.discogs.com/search?q=Don+Carlos+Ouverture", "discogs_search"),
    # Lauv
    "9C454BAFCD9A4DF7BC15CC9031": ("https://en.wikipedia.org/wiki/Paris_in_the_Rain", "wikipedia"),
    # Kira Linn
    "E1C4475B60AC4179852278CC60": ("https://www.discogs.com/search?q=Kira+Linn+Rage", "discogs_search"),
    "95D0CB9A84B54582AD17E79393": ("https://www.discogs.com/search?q=Kira+Linn+Illusion", "discogs_search"),
    "1C69AC77BB3042A4A237DF5966": ("https://www.discogs.com/search?q=Kira+Linn+Women+to+Women", "discogs_search"),
    # SZA
    "3CC7E3DF77334AE0B1B2BC3795": ("https://en.wikipedia.org/wiki/Snooze_(SZA_song)", "wikipedia"),
    # Shiina Ringo
    "8384A4DBF2B24CF98148BC9190": ("https://ja.wikipedia.org/wiki/The_Creamy_Season", "wikipedia_ja"),
    # Jake Miller
    "9B845A8E8A744BCAA50705D73D": ("https://www.discogs.com/search?q=Jake+Miller+The+Girl+Is+Mine", "discogs_search"),
    # Bobby Caldwell
    "E6A98A1C67AC4D9FA1E7437524": ("https://en.wikipedia.org/wiki/What_You_Won%27t_Do_for_Love", "wikipedia"),
    # Mariah Carey
    "448A140A77354AAC9458496B10": ("https://en.wikipedia.org/wiki/We_Belong_Together", "wikipedia"),
    # Welcome To Paradise Vol. III (10 tracks, same release)
    "870491B93FD04F979A12BC7811": ("https://www.discogs.com/release/Welcome-To-Paradise-Vol-III", "discogs_release"),
    "7AB4F0B765984F5AAADDDA584A": ("https://www.discogs.com/release/Welcome-To-Paradise-Vol-III", "discogs_release"),
    "F905340525194A1A9D838DB885": ("https://www.discogs.com/release/Welcome-To-Paradise-Vol-III", "discogs_release"),
    "3DD429C6458C404A9891416635": ("https://www.discogs.com/release/Welcome-To-Paradise-Vol-III", "discogs_release"),
    "B7C8FC27F3C349B0B39EB7D55A": ("https://www.discogs.com/release/Welcome-To-Paradise-Vol-III", "discogs_release"),
    "49BED5E277AF40909EC0BD756D": ("https://www.discogs.com/release/Welcome-To-Paradise-Vol-III", "discogs_release"),
    "818C48C49AE547DCBC6D3D9B43": ("https://www.discogs.com/release/Welcome-To-Paradise-Vol-III", "discogs_release"),
    "8E0D1A1B284746B3A1FCE15B41": ("https://www.discogs.com/release/Welcome-To-Paradise-Vol-III", "discogs_release"),
    "924CF2535CAB4C48A485EF4845": ("https://www.discogs.com/release/Welcome-To-Paradise-Vol-III", "discogs_release"),
    "A1CC7E194F30411980025EA378": ("https://www.discogs.com/release/Welcome-To-Paradise-Vol-III", "discogs_release"),
    # AKB48
    "BEF3CF9662D841528BDB33BFAF": ("https://en.wikipedia.org/wiki/Heavy_Rotation_(AKB48_song)", "wikipedia"),
    # Cantopop
    "6878BDAC7CD149C6A0AFA5ADD5": ("https://zh.wikipedia.org/wiki/%E4%B8%83%E5%8F%8B", "wikipedia_zh"),
    "594E0C4719E34D6A98B438C503": ("https://zh.wikipedia.org/wiki/%E4%B8%8B%E4%B8%80%E7%AB%99%E5%A4%A9%E5%90%8E", "wikipedia_zh"),
    "1C8E3363E3FF4BFBA5189ABCB7": ("https://zh.wikipedia.org/wiki/%E5%96%9C%E5%B8%96%E8%A1%97", "wikipedia_zh"),
    "3C6219AB60A24F5B8F101AF353": ("https://zh.wikipedia.org/wiki/%E5%96%9C%E6%AC%A2%E4%BD%A0_(G.E.M.%E6%AD%8C%E6%9B%B2)", "wikipedia_zh"),
    "CDF8316BBDD2488A877E675E8F": ("https://zh.wikipedia.org/wiki/%E5%9B%A0%E7%82%BA%E4%BD%A0%E6%89%80%E4%BB%A5%E6%88%91", "wikipedia_zh"),
    "3E6F4E0EC2024CCD9FE718F847": ("https://zh.wikipedia.org/wiki/%E5%AD%A4%E9%9B%8F", "wikipedia_zh"),
    "09E39F2B2CB84CEDAF02365CB2": ("https://zh.wikipedia.org/wiki/%E5%AF%8C%E5%A3%AB%E5%B1%B1%E4%B8%8B", "wikipedia_zh"),
    "2446DC352B064D65AD61142249": ("https://zh.wikipedia.org/wiki/%E5%BF%83%E6%B7%A1", "wikipedia_zh"),
    "C2AEFEEB56B0454F9AE069B899": ("https://zh.wikipedia.org/wiki/%E5%BF%85%E6%9D%80%E6%8A%80", "wikipedia_zh"),
    "7968BA9684454DD4B1B179643E": ("https://zh.wikipedia.org/wiki/%E7%88%B1%E4%B8%8E%E8%AA%A0", "wikipedia_zh"),
    "0CF65600F4474C71B19A9926EC": ("https://zh.wikipedia.org/wiki/%E7%9B%B8%E4%BE%9D%E7%82%BA%E5%91%BD", "wikipedia_zh"),
    # Oonuki Taeko
    "6835D62222154FA6B0509AA881": ("https://ja.wikipedia.org/wiki/%E8%8B%A5%E3%81%8D%E6%97%A5%E3%81%AE%E6%9C%9B%E6%A8%93", "wikipedia_ja"),
    # Genshin
    "3F34DF1B2784408BA60CE8165C": ("https://genshin.hoyoverse.com/en/news", "hoyoverse_official"),
    # Unknown series (filename inference, no real URL)
    "25A0003CEA50409D88B8362568": ("", "filename_inference_no_url"),
    "A52E2E2FA5474C61984BCECC27": ("", "filename_inference_no_url"),
    "77CB62FFAEC0446188CA3AE161": ("", "filename_inference_no_url"),
    "3FF351D346274F6C8DC078C78A": ("", "filename_inference_no_url"),
    "2996876392774F67887DA90C3E": ("", "filename_inference_no_url"),
    "AA089CC166EF4DE8AA78014AD9": ("", "filename_inference_no_url"),
    # Huang He Yao
    "286F49DADB7640A39E52E4C55A": ("https://zh.wikipedia.org/wiki/%E9%BB%84%E6%B2%B3%E8%B0%A3", "wikipedia_zh"),
}

with_url = 0
without_url = 0
unmapped = []

for aid, val in text_llm.items():
    if aid in source_url_map:
        url, stype = source_url_map[aid]
        val['source_url'] = url
        val['source_type'] = stype
        with_url += 1
    else:
        val['source_url'] = ''
        val['source_type'] = 'unmapped'
        without_url += 1
        unmapped.append((aid, val.get('title', 'unknown')))

print(f"有source_url: {with_url}/58")
print(f"无source_url: {without_url}/58")
if unmapped:
    print("未映射:")
    for aid, title in unmapped:
        print(f"  {aid} | {title}")

type_dist = Counter(v.get('source_type', 'unknown') for v in text_llm.values())
print(f"\nsource_type分布:")
for t, c in type_dist.most_common():
    print(f"  {t}: {c}")

with open('data/02_preannotation/genre_text_llm_annotations.json', 'w') as f:
    json.dump(text_llm, f, ensure_ascii=False, indent=2)
print(f"\ntext_llm_annotations.json 已更新（含source_url字段）")
