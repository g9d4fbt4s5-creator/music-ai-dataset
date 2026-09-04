"""
【实验特征链暂停 · T3 终审计 2026-09-04 用户拍板】
MERT/CLAP 嵌入聚类可视化属 KNN/聚类实验链（KNN 一致率 0% 已证伪、声学相似≠风格相似）。
作为诊断资产保留原地、不归档不删除，但不再服务当前 L4 生产；写报告/复盘时仍可运行。

MERT vs CLAP 嵌入聚类可视化（t-SNE 2D + plotly）
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent.parent.parent
REPORT = ROOT / "data/00.5_cleaned/reports/v20260824_083208"
OUT = REPORT / "embedding_tsne_comparison.html"
COLORS = ["#e6194B","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#42d4f4","#f032e6","#bfef45","#469990","#dcbeff","#9A6324","#800000","#aaffc3"]

def load_emb(d):
    return {f.stem.replace("_mert_embedding","").replace("_clap_embedding",""): np.load(f)
            for f in sorted(d.glob("*.npy"))}

def load_labels(d):
    out = {}
    for f in sorted(d.glob("*.json")):
        x = json.load(open(f, encoding="utf-8"))
        aid = x.get("audio_id", f.stem.replace("_semantic",""))
        gc = x.get("genre_candidates", [])
        mc = x.get("mood_candidates", [])
        out[aid] = {
            "genre": gc[0]["label"] if gc and isinstance(gc[0], dict) else (gc[0] if gc else "unknown"),
            "mood": mc[0]["label"] if mc and isinstance(mc[0], dict) else (mc[0] if mc else "unknown"),
            "vocal": x.get("vocal_presence","unknown"),
        }
    return out

def main():
    mert = load_emb(REPORT/"l2_mert_embedding")
    clap = load_emb(REPORT/"l2_clap_embedding")
    labels = load_labels(ROOT/"data/02_preannotation/l2_semantic")
    qc = pd.read_csv(REPORT/"qc_gate_report_v2.csv")
    qc_map = {r["audio_id"]: {"dur":r["duration_sec"],"branch":r["final_branch"]} for _,r in qc.iterrows()}

    ids = sorted(set(mert)&set(clap)&set(labels))
    print(f"共同样本: {len(ids)}")
    Xm = np.array([mert[i] for i in ids])
    Xc = np.array([clap[i] for i in ids])
    genres = [labels[i]["genre"] for i in ids]
    moods = [labels[i]["mood"] for i in ids]
    vocals = [labels[i]["vocal"] for i in ids]
    durs = [qc_map.get(i,{}).get("dur",0) for i in ids]
    branches = [qc_map.get(i,{}).get("branch","?") for i in ids]
    print(f"流派: {pd.Series(genres).value_counts().to_dict()}")
    print(f"情绪: {pd.Series(moods).value_counts().to_dict()}")

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5,len(ids)-1), max_iter=1000, learning_rate="auto")
    Xm2 = tsne.fit_transform(Xm)
    Xc2 = tsne.fit_transform(Xc)

    def cmap(cats):
        u = sorted(set(cats))
        return {c: COLORS[i%len(COLORS)] for i,c in enumerate(u)}

    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        "MERT (按人声)","CLAP (按人声)","MERT (按情绪)","CLAP (按情绪)"],
        horizontal_spacing=0.08, vertical_spacing=0.12)

    def scatter(row,col,X2,cats,cm,name):
        for cat in sorted(set(cats)):
            idx = [j for j,c in enumerate(cats) if c==cat]
            fig.add_trace(go.Scatter(
                x=[X2[j,0] for j in idx], y=[X2[j,1] for j in idx],
                mode="markers", name=cat,
                marker=dict(size=12, color=cm[cat], line=dict(width=1,color="#333")),
                text=[f"ID: {ids[j][:20]}<br>流派: {genres[j]}<br>情绪: {moods[j]}<br>人声: {vocals[j]}<br>时长: {durs[j]:.0f}s<br>QC: {branches[j]}" for j in idx],
                hovertemplate="%{text}<extra></extra>",
                showlegend=(row==1 and col==1)), row=row, col=col)

    scatter(1,1,Xm2,vocals,cmap(vocals),"mv")
    scatter(1,2,Xc2,vocals,cmap(vocals),"cv")
    scatter(2,1,Xm2,moods,cmap(moods),"mm")
    scatter(2,2,Xc2,moods,cmap(moods),"cm")

    fig.update_layout(title=dict(text=f"MERT vs CLAP 嵌入聚类对比 (t-SNE 2D, n={len(ids)})", x=0.5, font=dict(size=18)),
        height=900, width=1400, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    for r in [1,2]:
        for c in [1,2]:
            fig.update_xaxes(title_text="t-SNE 1", row=r, col=c, gridcolor="#eee")
            fig.update_yaxes(title_text="t-SNE 2", row=r, col=c, gridcolor="#eee")

    fig.write_html(str(OUT), include_plotlyjs="cdn")
    print(f"\n✅ 已保存: {OUT} ({OUT.stat().st_size/1024:.0f}KB)")

    print("\n=== 轮廓系数 (越高聚类越好) ===")
    if len(set(genres))>1:
        print(f"  流派: MERT={silhouette_score(Xm,genres):.3f}, CLAP={silhouette_score(Xc,genres):.3f}")
    if len(set(moods))>1:
        print(f"  情绪: MERT={silhouette_score(Xm,moods):.3f}, CLAP={silhouette_score(Xc,moods):.3f}")
    if len(set(vocals))>1:
        print(f"  人声: MERT={silhouette_score(Xm,vocals):.3f}, CLAP={silhouette_score(Xc,vocals):.3f}")
    print("  解读: >0.5良好, 0.2-0.5弱聚类, <0.2无明显聚类")

if __name__=="__main__":
    main()
