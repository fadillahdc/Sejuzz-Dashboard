"""Grafik Plotly. Satu rona biru untuk besaran, oranye hanya untuk pembanding/negatif.
Warna tidak bergantung pada tema: latar mengikuti tema Streamlit (theme="streamlit")."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config
from data_utils import fmt_pct

TINGGI_BARIS = 26


FONT = dict(family="Source Sans Pro, sans-serif", size=13, color="#17243a")
SKALA_BIRU = [[0, "#bcd6f5"], [1, config.BIRU]]


def _gaya(fig):
    """Latar transparan, font seragam, tooltip bersih (berlaku untuk semua grafik)."""
    fig.update_layout(font=FONT, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      hoverlabel=dict(bgcolor="white", font_size=13, bordercolor="#dde6f2"))
    fig.update_xaxes(gridcolor="#e3eaf4", linecolor="#c9d6e8")
    fig.update_yaxes(gridcolor="#e3eaf4", linecolor="#c9d6e8")
    return fig


def _pasang(fig, judul=None, tinggi=None):
    fig.update_layout(title=dict(text=judul, x=0, xanchor="left", font=dict(size=15)) if judul else None,
                      margin=dict(l=10, r=30, t=50 if judul else 10, b=10), height=tinggi, showlegend=False,
                      hoverlabel=dict(font_size=12))
    fig.update_xaxes(showgrid=True, gridwidth=0.6, zeroline=False)
    fig.update_yaxes(showgrid=False)
    return _gaya(fig)


def bar_yield(ring, judul="Yield rata-rata per jenis"):
    s = ring.sort_values("yield_rata2")
    fig = go.Figure(go.Bar(
        x=s["yield_rata2"] * 100, y=s.index, orientation="h",
        marker=dict(color=s["yield_rata2"], colorscale=SKALA_BIRU, cmin=s["yield_rata2"].min() - 0.15, cmax=s["yield_rata2"].max()),
        text=[fmt_pct(v) for v in s["yield_rata2"]], textposition="outside", cliponaxis=False,
        customdata=np.stack([s["n"], s["yield_sd"] * 100, s["yield_p10"] * 100], axis=-1),
        hovertemplate="<b>%{y}</b><br>Rata-rata: %{x:.1f}%<br>Simpangan baku: %{customdata[1]:.1f} poin<br>"
                      "P10: %{customdata[2]:.1f}%<br>Sampel: %{customdata[0]:.0f}<extra></extra>"))
    fig.update_xaxes(range=[0, 112], title="bagian layak / berat utuh (%)")
    return _pasang(fig, judul, TINGGI_BARIS * len(s) + 80)


def bar_cv(ring, judul="Ketidakstabilan yield (koefisien variasi)"):
    s = ring.sort_values("yield_cv")
    fig = go.Figure(go.Bar(
        x=s["yield_cv"] * 100, y=s.index, orientation="h",
        marker=dict(color=s["yield_cv"], colorscale=[[0, "#fbd9c8"], [1, config.ORANYE]]),
        text=[fmt_pct(v) for v in s["yield_cv"]], textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>CV: %{x:.1f}%<extra></extra>"))
    fig.update_xaxes(range=[0, max(s["yield_cv"].max() * 100 * 1.2, 5)], title="sd / rata-rata (%), makin besar makin sulit diprediksi")
    return _pasang(fig, judul, TINGGI_BARIS * len(s) + 80)


def box_yield(df, urutan, judul="Sebaran yield per jenis"):
    fig = go.Figure()
    for j in urutan:
        fig.add_trace(go.Box(y=df.loc[df["jenis_buah"] == j, "yield_pct"] * 100, name=j, marker_color=config.BIRU,
                             line=dict(width=1.5), boxpoints="outliers", hovertemplate="%{y:.1f}%<extra>" + j + "</extra>"))
    fig.update_yaxes(title="yield (%)", showgrid=True)
    fig.update_xaxes(showgrid=False, tickangle=-45)
    fig.update_layout(title=dict(text=judul, x=0, font=dict(size=15)), margin=dict(l=10, r=10, t=50, b=10), height=420, showlegend=False)
    return _gaya(fig)


def hist_yield(df_satu, nama):
    fig = px.histogram(df_satu, x=df_satu["yield_pct"] * 100, nbins=20, color_discrete_sequence=[config.BIRU])
    fig.update_traces(marker_line_width=1, marker_line_color="rgba(255,255,255,0.8)",
                      hovertemplate="Yield %{x:.1f}%<br>Jumlah sampel: %{y}<extra></extra>")
    fig.update_xaxes(title="yield (%)"); fig.update_yaxes(title="jumlah sampel", showgrid=True)
    return _pasang(fig, f"Sebaran yield {nama}", 320)


def bar_sentimen(df):
    hitung = df["label"].value_counts().reindex(config.LABELS, fill_value=0)
    total = max(int(hitung.sum()), 1)
    fig = go.Figure(go.Bar(
        x=list(hitung.index), y=hitung.values, marker_color=[config.WARNA[l] for l in hitung.index],
        text=[f"{v} ({v / total * 100:.0f}%)" for v in hitung.values], textposition="outside", cliponaxis=False,
        hovertemplate="%{x}: %{y} komentar<extra></extra>", width=0.55))
    fig.update_yaxes(title="jumlah komentar", showgrid=True, range=[0, max(hitung.max() * 1.18, 1)])
    fig.update_xaxes(showgrid=False)
    fig.update_layout(title=dict(text="Distribusi sentimen", x=0, font=dict(size=15)), margin=dict(l=10, r=10, t=50, b=10),
                      height=340, showlegend=False)
    return _gaya(fig)


def hist_bobot(df):
    fig = go.Figure()
    for l in config.LABELS:
        d = df[df["label"] == l]
        fig.add_trace(go.Histogram(x=d["bobot"], name=l, marker_color=config.WARNA[l], xbins=dict(size=0.25), opacity=0.9,
                                   hovertemplate="Bobot %{x}<br>%{y} komentar<extra>" + l + "</extra>"))
    fig.update_layout(barmode="stack", title=dict(text="Sebaran bobot skor", x=0, font=dict(size=15)),
                      margin=dict(l=10, r=10, t=50, b=10), height=340, bargap=0.05,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"))
    fig.add_vline(x=config.BATAS_NEGATIF, line_width=1.2, line_dash="dot", line_color="#8f8e88")
    fig.update_xaxes(title="bobot skor (di bawah 0 = Negatif)", showgrid=False); fig.update_yaxes(title="jumlah komentar", showgrid=True)
    return _gaya(fig)


def heat_confusion(cm):
    z = cm.values
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"Prediksi {c}" for c in cm.columns], y=[f"Asli {c}" for c in cm.index], colorscale=[[0, "#eaf2fc"], [1, "#2a78d6"]],
        showscale=False, text=z, texttemplate="%{text}", textfont=dict(size=15),
        hovertemplate="%{y}, %{x}: %{z} komentar<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False); fig.update_xaxes(showgrid=False, side="top")
    fig.update_layout(title=dict(text="Confusion matrix", x=0, font=dict(size=15)), margin=dict(l=10, r=10, t=80, b=10), height=330)
    return _gaya(fig)


def scatter_aktual_prediksi(uji):
    """Data uji: berat layak sebenarnya vs prediksi. Titik di garis putus-putus = prediksi tepat."""
    mx = float(max(uji["berat_bagian_layak_g"].max(), uji["prediksi"].max())) * 1.03
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color="#8f8e88", dash="dot", width=1.2), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=uji["berat_bagian_layak_g"], y=uji["prediksi"], mode="markers", text=uji["jenis_buah"],
                             marker=dict(color=config.BIRU, size=7, opacity=0.65, line=dict(width=0.5, color="white")),
                             hovertemplate="<b>%{text}</b><br>Sebenarnya: %{x:.0f} g<br>Prediksi: %{y:.0f} g<extra></extra>"))
    fig.update_xaxes(title="berat bagian layak sebenarnya (g)", showgrid=True)
    fig.update_yaxes(title="prediksi (g)", showgrid=True)
    return _pasang(fig, "Prediksi vs sebenarnya (data uji)", 400)


def bar_mae_jenis(per_jenis):
    s = per_jenis.sort_values("MAE_persen")
    fig = go.Figure(go.Bar(
        x=s["MAE_persen"], y=s.index, orientation="h", marker=dict(color=s["MAE_persen"], colorscale=[[0, "#c9dcf3"], [1, config.ORANYE]]),
        text=[f"{v:.1f}%".replace(".", ",") for v in s["MAE_persen"]], textposition="outside", cliponaxis=False,
        customdata=np.stack([s["MAE_g"], s["n_uji"]], axis=-1),
        hovertemplate="<b>%{y}</b><br>MAE: %{customdata[0]:.1f} g<br>Data uji: %{customdata[1]:.0f}<extra></extra>"))
    fig.update_xaxes(range=[0, max(s["MAE_persen"].max() * 1.2, 5)], title="galat rata-rata terhadap rata-rata berat layak (%)")
    return _pasang(fig, "Galat prediksi per jenis (data uji)", TINGGI_BARIS * len(s) + 80)
