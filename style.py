"""Tampilan: CSS kustom dan komponen kecil (hero, chip). Warna mengikuti tema di .streamlit/config.toml."""
import re

import streamlit as st

CSS = """
<style>
.block-container {padding-top: 4.2rem; padding-bottom: 3rem; max-width: 1280px;}
h1, h2, h3 {letter-spacing: -0.01em;}
h2, h3 {font-weight: 650;}
.block-container h2 {font-size: 1.45rem;}
.block-container h3 {font-size: 1.2rem;}

/* Kartu metrik */
[data-testid="stMetric"] {
  background: #ffffff; border: 1px solid #dde6f2; border-radius: 16px;
  padding: 14px 18px; box-shadow: 0 1px 3px rgba(23,36,58,.06);
}
[data-testid="stMetricLabel"] {color: #5b6b82;}
[data-testid="stMetricValue"] {font-weight: 700; color: #17243a; font-size: 1.75rem;}
[data-testid="stMetricValue"] > div, [data-testid="stMetricLabel"] p {overflow: visible !important; text-overflow: clip !important; white-space: normal !important;}

/* Kontainer berbingkai */
[data-testid="stVerticalBlockBorderWrapper"] {background: #ffffff; border-radius: 16px;}

/* Sidebar */
[data-testid="stSidebar"] {background: linear-gradient(180deg, #ffffff 0%, #eaf1fb 100%); border-right: 1px solid #dde6f2;}
[data-testid="stSidebar"] [role="radiogroup"] {gap: 4px;}
[data-testid="stSidebar"] [role="radiogroup"] label {
  padding: 9px 12px; border-radius: 12px; width: 100%; transition: background .15s;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover:not([data-selected="true"]) {background: #dbe8f9;}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {background: #1f6fd1 !important;}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] * {color: #ffffff !important; font-weight: 600;}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child {display: none;}

/* Tab */
button[data-baseweb="tab"] {font-weight: 600;}

/* Hero */
.hero {
  background: linear-gradient(120deg, #1f6fd1 0%, #3fa2f0 55%, #6cc3e8 100%);
  color: #fff; border-radius: 22px; padding: 22px 28px; margin-bottom: 16px;
  box-shadow: 0 8px 24px rgba(31,111,209,.22);
}
.hero h1 {color: #fff; margin: 0 0 4px 0; font-size: 1.9rem; padding: 0;}
.hero p {margin: 0; opacity: .94; font-size: 1.02rem;}
.chip {
  display: inline-block; padding: 3px 11px; margin: 10px 8px 0 0; border-radius: 999px;
  font-size: .78rem; font-weight: 600; background: rgba(255,255,255,.22); color: #fff; border: 1px solid rgba(255,255,255,.35);
}
.chip.warn {background: #fff3cd; color: #7a5200; border-color: #f2d98a;}
.sidebar-brand {font-size: 1.15rem; font-weight: 700; color: #17243a; margin-bottom: 2px;}
.sidebar-sub {color: #5b6b82; font-size: .85rem; margin-bottom: 14px;}
.kartu-temuan {
  background: #fff; border: 1px solid #dde6f2; border-left: 5px solid #1f6fd1; border-radius: 14px;
  padding: 14px 18px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(23,36,58,.05);
}
.kartu-temuan b {color: #17243a;}
/* Kartu KPI (Ringkasan) */
.kpi {background: #fff; border: 1px solid #dde6f2; border-radius: 16px; padding: 14px 18px; height: 100%;
  box-shadow: 0 1px 3px rgba(23,36,58,.06);}
.kpi .l {color: #5b6b82; font-size: .85rem;}
.kpi .v {color: #17243a; font-weight: 700; font-size: 1.9rem; line-height: 1.2; margin-top: 2px;}
.kpi .s {color: #1f6fd1; font-weight: 600; font-size: .9rem; margin-top: 2px;}
.kpi.warn .s {color: #d4541f;}
/* Hasil besar (Kalkulator) */
.hasil {background: linear-gradient(135deg, #eaf3ff 0%, #ffffff 100%); border: 1px solid #cfe0f6; border-radius: 16px; padding: 18px 22px; margin: 6px 0 10px 0;}
.hasil .l {color: #5b6b82; font-size: .9rem;}
.hasil .v {color: #1f6fd1; font-weight: 750; font-size: 2.6rem; line-height: 1.1;}
.hasil .v small {font-size: 1.1rem; color: #5b6b82; font-weight: 600; margin-left: 6px;}
.hasil .d {color: #33445c; font-size: .92rem; margin-top: 6px;}
/* Catatan tipis */
.catatan {background: #fff8e1; border: 1px solid #f2d98a; color: #7a5200; border-radius: 12px; padding: 8px 14px; font-size: .9rem; margin: -4px 0 14px 0;}
</style>
"""


def pasang_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(judul, subjudul="", chips=(), peringatan=(), auto=True):
    """Header halaman. chips = label netral, peringatan = label kuning (mis. SIMULASI)."""
    if auto and not peringatan:
        df = st.session_state.get("df_buah")
        if df is not None and "SIMULASI" in set(df["sumber_data"].astype(str)):
            peringatan = ["DATA SIMULASI"]
    c = "".join(f'<span class="chip">{x}</span>' for x in chips)
    c += "".join(f'<span class="chip warn">{x}</span>' for x in peringatan)
    st.markdown(f'<div class="hero"><h1>{judul}</h1><p>{subjudul}</p>{c}</div>', unsafe_allow_html=True)


def _md(teks):
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", str(teks))


def kartu(teks):
    st.markdown(f'<div class="kartu-temuan">{_md(teks)}</div>', unsafe_allow_html=True)


def kpi(label, nilai, sub="", peringatan=False):
    st.markdown(f'<div class="kpi{" warn" if peringatan else ""}"><div class="l">{label}</div><div class="v">{nilai}</div>'
                f'<div class="s">{sub or "&nbsp;"}</div></div>', unsafe_allow_html=True)


def hasil_besar(label, nilai, satuan="", detail=""):
    st.markdown(f'<div class="hasil"><div class="l">{label}</div><div class="v">{nilai}<small>{satuan}</small></div>'
                f'<div class="d">{detail}</div></div>', unsafe_allow_html=True)


def catatan(teks):
    st.markdown(f'<div class="catatan">{_md(teks)}</div>', unsafe_allow_html=True)
