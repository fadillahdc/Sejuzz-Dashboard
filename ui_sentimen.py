"""Halaman sentimen pengunjung (bobot skor + label)."""
import pandas as pd
import streamlit as st

import charts
import config
import style
from data_utils import fmt_pct, metrik_klasifikasi
from sentiment_client import SentimentAPIError, SentimentClient, _PETA_LABEL, get_client


@st.cache_data(show_spinner="Menghitung skor sentimen...")
def _skor(teks, mode, url, api_key, timeout, model_path):
    client = SentimentClient(mode=mode, url=url, api_key=api_key, timeout=timeout, model_path=model_path)
    return pd.DataFrame(client.predict(list(teks)))


def skor_komentar(daftar_teks):
    """Beri bobot dan label pada daftar komentar lewat klien aktif (mock atau API)."""
    s = config.sentiment_settings()
    hasil = _skor(tuple(daftar_teks), s["mode"], s["url"], s["api_key"], s["timeout"], s["model_path"])
    out = pd.DataFrame({"komentar": list(daftar_teks), "bobot": hasil["bobot"].values, "label": hasil["label"].values})
    tidak_terbaca = out["label"].isna()          # API mengembalikan sentimen null bila tidak ada kata yang bisa dianalisis
    out = out[~tidak_terbaca].reset_index(drop=True)
    out.attrs["dilewati"] = int(tidak_terbaca.sum())
    return out


def ambil_sentimen():
    """Komentar contoh (dummy) yang sudah diberi skor. Dipakai halaman Ringkasan."""
    teks = pd.read_csv(config.KOMENTAR_CONTOH)["komentar"].astype(str).tolist()
    return skor_komentar(teks)


def _fmt_bobot(b):
    return f"{b:.2f}".replace(".", ",")


def _baris(r):
    return f"- {r.komentar}. **Bobot {_fmt_bobot(r.bobot)}** · {r.label}"


def _info_mode():
    s = config.sentiment_settings()
    if not config.sentimen_nyata(s["mode"]):
        st.warning("**MODE CONTOH (MOCK).** Skor dihitung dengan kamus kata sederhana, bukan model tim, dan komentarnya dummy. "
                   "Tampilan ini untuk menguji dasbor. Sambungkan API model di tab **Koneksi API**.")
    else:
        st.success("Memakai model tim " + ("lewat API (mode api)." if s["mode"] == "api" else "dari berkas lokal (mode joblib)."))


def _kpi(df):
    k = st.columns(len(config.LABELS) + 2)
    k[0].metric("Total komentar", f"{len(df):,}".replace(",", "."))
    for col, l in zip(k[1:-1], config.LABELS):
        col.metric(l, fmt_pct((df["label"] == l).mean(), 0), help=f"{int((df['label'] == l).sum())} komentar")
    k[-1].metric("Rata-rata bobot", _fmt_bobot(df["bobot"].mean()))


def _tab_ringkasan(df):
    _kpi(df)
    a, b = st.columns(2)
    a.plotly_chart(charts.bar_sentimen(df), width="stretch")
    b.plotly_chart(charts.hist_bobot(df), width="stretch")


def _tab_komentar(df):
    st.markdown("**Contoh komentar terklasifikasi**")
    a, b = st.columns(2)
    with a:
        st.markdown("Paling positif")
        for r in df.sort_values("bobot", ascending=False).head(5).itertuples():
            st.markdown(_baris(r))
    with b:
        st.markdown("Paling negatif")
        for r in df.sort_values("bobot").head(5).itertuples():
            st.markdown(_baris(r))
    st.divider()
    c1, c2, c3 = st.columns([2, 3, 2])
    pilih = c1.multiselect("Sentimen", config.LABELS, default=list(config.LABELS))
    cari = c2.text_input("Cari kata dalam komentar")
    urut = c3.selectbox("Urutkan", ["Bobot tertinggi", "Bobot terendah", "Urutan awal"])
    t = df[df["label"].isin(pilih)]
    if cari:
        t = t[t["komentar"].str.contains(cari, case=False, na=False, regex=False)]
    if urut == "Bobot tertinggi":
        t = t.sort_values("bobot", ascending=False)
    elif urut == "Bobot terendah":
        t = t.sort_values("bobot")
    st.caption(f"{len(t):,} komentar".replace(",", "."))
    st.dataframe(t.rename(columns={"komentar": "Kalimat", "bobot": "Bobot", "label": "Sentimen"}), hide_index=True, width="stretch",
                 column_config={"Bobot": st.column_config.NumberColumn(format="%.2f")})
    st.download_button("Unduh hasil (CSV)", t.to_csv(index=False).encode("utf-8"), "hasil_sentimen.csv", "text/csv")


def _tab_coba():
    st.caption("Ketik satu komentar untuk melihat bobot dan labelnya.")
    teks = st.text_area("Komentar", placeholder="Contoh: jusnya segar tapi lama banget", height=100)
    if st.button("Analisis", type="primary"):
        if not teks.strip():
            st.info("Tulis komentar dulu.")
            return
        try:
            h = get_client().predict([teks])[0]
        except SentimentAPIError as e:
            st.error(str(e))
            return
        if h["label"] is None:
            st.warning(h.get("peringatan") or "Model tidak bisa membaca kalimat ini (tidak ada kata yang dikenali).")
            return
        st.markdown(f"### {teks.strip()}\n**Bobot {_fmt_bobot(h['bobot'])}** · **{h['label']}**")
        if h.get("yakin") is False:
            st.caption("Model ragu pada kalimat ini (skor mendekati 0).")


def _norm_eval(df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    kurang = [c for c in ["komentar", "label_asli", "label_prediksi"] if c not in df.columns]
    if kurang:
        return None, f"Kolom tidak ada: {', '.join(kurang)}"
    for c in ["label_asli", "label_prediksi"]:
        df[c] = df[c].astype(str).str.strip().str.lower().map(_PETA_LABEL)
    if df[["label_asli", "label_prediksi"]].isna().any().any():
        return None, "Ada label yang tidak dikenali. Gunakan Positif atau Negatif."
    return df, None


def _tab_evaluasi():
    up = st.file_uploader("Unggah hasil uji (CSV: komentar, label_asli, label_prediksi, opsional bobot)", type="csv", key="evalup")
    if up is not None:
        raw, sumber, asli_tim = pd.read_csv(up), f"file unggahan ({up.name})", True
    elif config.EVALUASI_MODEL.exists():
        raw, sumber, asli_tim = pd.read_csv(config.EVALUASI_MODEL), "data/evaluasi_model.csv (hasil uji model tim)", True
    else:
        raw, sumber, asli_tim = pd.read_csv(config.EVALUASI_CONTOH), "data/evaluasi_contoh.csv", False
    if not asli_tim:
        st.warning("**EVALUASI CONTOH.** Ini menilai skor MOCK pada komentar dummy, bukan akurasi model tim. "
                   "Untuk menampilkan evaluasi model tim, unggah hasil ujinya di atas atau simpan sebagai data/evaluasi_model.csv.")
    st.caption(f"Sumber: {sumber}")
    df, err = _norm_eval(raw)
    if err:
        st.error(err)
        return
    m = metrik_klasifikasi(df["label_asli"], df["label_prediksi"], list(config.LABELS))
    a, b, c = st.columns(3)
    a.metric("Accuracy", fmt_pct(m["accuracy"]))
    b.metric("Macro F1", fmt_pct(m["macro_f1"]))
    c.metric("Jumlah data uji", f"{m['n']:,}".replace(",", "."))
    st.caption("Data uji harus data yang tidak dipakai saat melatih model.")
    k, h = st.columns([3, 2])
    t = m["per_kelas"].copy()
    for col in ["precision", "recall", "f1"]:
        t[col] = (t[col] * 100).round(1)
    k.markdown("**Per kelas**")
    k.dataframe(t.rename(columns={"precision": "Precision (%)", "recall": "Recall (%)", "f1": "F1 (%)", "kelas": "Kelas", "jumlah_data": "Jumlah data"}),
                hide_index=True, width="stretch")
    h.plotly_chart(charts.heat_confusion(m["confusion"]), width="stretch")
    salah = df[df["label_asli"] != df["label_prediksi"]]
    with st.expander(f"Komentar yang salah diklasifikasi ({len(salah)})"):
        st.dataframe(salah.head(100), hide_index=True, width="stretch")


def _tab_koneksi():
    s = config.sentiment_settings()
    st.markdown(f"**Mode saat ini:** `{s['mode']}`  \n**URL API:** `{s['url'] or '(belum diisi)'}`  \n"
                f"**API key:** {'terisi' if s['api_key'] else 'belum diisi'}  \n**Model lokal:** `{s['model_path']}`")
    a, b = st.columns(2)
    if a.button("Tes koneksi", width="stretch"):
        try:
            h = get_client().tes_koneksi()
            if h["label"] is None:
                st.warning("Terhubung, tapi kalimat tes tidak terbaca model.")
            else:
                st.success(f"Berhasil. Contoh hasil: bobot {_fmt_bobot(h['bobot'])}, {h['label']}.")
        except SentimentAPIError as e:
            st.error(str(e))
    if b.button("Cek info model", width="stretch"):
        try:
            st.json(get_client().info())
        except SentimentAPIError as e:
            st.error(str(e))
    st.markdown("**Mode model lokal (tanpa API key)**")
    st.code('# otomatis aktif bila model/model_sentimen.joblib ada dan scikit-learn terpasang.\n# Paksa lewat secrets.toml:\nSENTIMENT_MODE = "joblib"', language="toml")
    st.markdown("**Cara menyambungkan ke Sejuzz AI Service**")
    st.code('# .streamlit/secrets.toml   (atau variabel lingkungan dengan nama sama)\n'
            'SENTIMENT_MODE = "api"\nSENTIMENT_API_URL = "https://sejuzz-capstone-production-362bebe7.up.railway.app"\n'
            'SENTIMENT_API_KEY = ""   # kosongkan bila tidak diperlukan atau tidak diizinkan', language="toml")
    st.markdown("**Kontrak API yang dipakai**")
    st.code('POST /api/v1/sentimen/batch      header: X-API-Key\n{"daftar_teks": ["jusnya segar", "lama banget"]}\n\n'
            'Jawaban:\n{"jumlah": 2, "jumlah_positif": 1, "jumlah_negatif": 1, "jumlah_tidak_terbaca": 0,\n'
            ' "hasil": [{"teks": "jusnya segar", "sentimen": "positif", "skor": 1.2, "yakin": true, "peringatan": null}, ...]}\n\n'
            'Maks 100 teks per permintaan (dasbor memecah otomatis). sentimen null = tidak terbaca, dilewati dasbor.', language="json")


def halaman_sentimen():
    _mode = config.sentiment_settings()["mode"]
    style.hero("Sentimen pengunjung", "Apa kata pengunjung tentang jus dan layanan cafe.", auto=False,
               chips=[{"api": "Model tim via API", "joblib": "Model tim (joblib)"}[_mode]] if config.sentimen_nyata(_mode) else [],
               peringatan=[] if config.sentimen_nyata(_mode) else ["MODE CONTOH (MOCK)"])
    _info_mode()
    sumber = st.radio("Sumber komentar", ["Data contoh (dummy)", "Unggah CSV komentar"], horizontal=True)
    df = None
    try:
        if sumber.startswith("Data contoh"):
            df = ambil_sentimen()
            st.caption("Komentar contoh adalah dummy buatan (bukan dari Instagram atau orang sungguhan).")
        else:
            up = st.file_uploader("CSV dengan kolom `komentar`", type="csv", key="komup")
            if up is not None:
                raw = pd.read_csv(up)
                raw.columns = [c.strip().lower() for c in raw.columns]
                if "komentar" not in raw.columns:
                    st.error("Kolom `komentar` tidak ditemukan.")
                else:
                    teks = raw["komentar"].dropna().astype(str).str.strip()
                    teks = teks[teks != ""].tolist()[:config.MAX_BARIS_UNGGAH]
                    df = skor_komentar(teks) if teks else None
    except SentimentAPIError as e:
        st.error(f"Gagal menghitung sentimen: {e}")
    if df is not None and df.attrs.get("dilewati"):
        st.warning(f"{df.attrs['dilewati']} komentar tidak bisa dibaca model (tidak ada kata yang dikenali) dan dilewati.")
    if df is not None and df.empty:
        st.error("Tidak ada komentar yang bisa dianalisis.")
        df = None
    t1, t2, t3, t4, t5 = st.tabs(["Ringkasan", "Komentar", "Coba kalimat", "Evaluasi model", "Koneksi API"])
    with t1:
        if df is not None:
            _tab_ringkasan(df)
        else:
            st.info("Belum ada komentar untuk ditampilkan.")
    with t2:
        if df is not None:
            _tab_komentar(df)
        else:
            st.info("Belum ada komentar untuk ditampilkan.")
    with t3:
        _tab_coba()
    with t4:
        _tab_evaluasi()
    with t5:
        _tab_koneksi()
