"""Halaman data buah: Ringkasan, Yield per jenis, Kalkulator produksi, Data & pembaruan."""
import pandas as pd
import streamlit as st

import charts
import config
import style
from data_utils import (KOLOM_TEMPLATE, estimasi_cup, fmt_num, fmt_pct, kebutuhan_buah, pilih_yield, ringkasan_per_jenis,
                        validasi_dan_siapkan)
from sentiment_client import SentimentAPIError


@st.cache_data
def _bawaan():
    df, _, _ = validasi_dan_siapkan(pd.read_csv(config.DATASET_BUAH))
    return df


@st.cache_data
def _ring(df):
    return ringkasan_per_jenis(df)


def dataset_aktif():
    if "df_buah" not in st.session_state:
        st.session_state["df_buah"] = _bawaan()
        st.session_state["nama_dataset"] = "Data bawaan (dataset_buah_bersih.csv)"
    return st.session_state["df_buah"]


def banner_data():
    """Catatan tipis di bawah header. Simulasi sudah ditandai chip di header; di sini hanya kasus asal data tidak diketahui."""
    sumber = set(dataset_aktif()["sumber_data"].astype(str).unique())
    if "TIDAK DIKETAHUI" in sumber:
        style.catatan("**Asal data tidak diketahui** (kolom sumber_data kosong). Isi SIMULASI atau NYATA agar label di dasbor benar.")


def halaman_ringkasan():
    df = dataset_aktif(); ring = _ring(df)
    style.hero("Ringkasan produksi buah", "Seberapa banyak bagian buah yang benar-benar jadi jus, dan jenis mana yang paling bisa diandalkan.",
               chips=[f"{len(df):,} sampel".replace(",", "."), f"{len(ring)} jenis"])
    banner_data()
    t, r, c = ring["yield_rata2"].idxmax(), ring["yield_rata2"].idxmin(), ring["yield_cv"].idxmax()
    k = st.columns(5)
    with k[0]: style.kpi("Total sampel", f"{len(df):,}".replace(",", "."), "timbangan tercatat")
    with k[1]: style.kpi("Jenis tercatat", len(ring), f"{ring['kelompok'].nunique()} kelompok")
    with k[2]: style.kpi("Yield tertinggi", fmt_pct(ring.loc[t, "yield_rata2"]), t)
    with k[3]: style.kpi("Yield terendah", fmt_pct(ring.loc[r, "yield_rata2"]), r, peringatan=True)
    with k[4]: style.kpi("Paling tidak stabil", fmt_pct(ring.loc[c, "yield_cv"]), f"{c} (koef. variasi)", peringatan=True)
    st.write("")
    a, b = st.columns(2)
    a.plotly_chart(charts.bar_yield(ring), width="stretch")
    b.plotly_chart(charts.bar_cv(ring), width="stretch")
    st.subheader("Sentimen pengunjung")
    try:
        from ui_sentimen import ambil_sentimen
        s = ambil_sentimen()
        q = st.columns(len(config.LABELS) + 1)
        for col, l in zip(q, config.LABELS):
            col.metric(l, fmt_pct((s["label"] == l).mean(), 0))
        q[-1].metric("Rata-rata bobot", f"{s['bobot'].mean():.2f}".replace(".", ","))
        if not config.sentimen_nyata(config.sentiment_settings()["mode"]):
            st.caption("Data contoh (dummy) dengan skor MOCK. Detail di halaman Sentimen pengunjung.")
    except SentimentAPIError as e:
        st.info(f"Sentimen belum tersedia: {e}")


def halaman_yield():
    df = dataset_aktif(); ring = _ring(df)
    style.hero("Yield per jenis", "Bandingkan hasil jus antarjenis buah dan lihat sebarannya.")
    banner_data()
    st.caption("Yield = berat bagian layak / berat buah utuh. P10 = yield yang hanya dilampaui 10% sampel terburuk (angka konservatif).")
    c1, c2 = st.columns(2)
    kel_opsi = sorted(ring["kelompok"].unique())
    kel = c1.multiselect("Kelompok", kel_opsi, default=kel_opsi)
    j_opsi = sorted(ring[ring["kelompok"].isin(kel)].index)
    jenis = c2.multiselect("Jenis", j_opsi, default=j_opsi)
    if not jenis:
        st.info("Pilih minimal satu jenis.")
        return
    rf = ring.loc[jenis]; dff = df[df["jenis_buah"].isin(jenis)]
    t1, t2, t3 = st.tabs(["Peringkat", "Sebaran", "Tabel"])
    with t1:
        a, b = st.columns(2)
        a.plotly_chart(charts.bar_yield(rf), width="stretch")
        b.plotly_chart(charts.bar_cv(rf), width="stretch")
    with t2:
        st.plotly_chart(charts.box_yield(dff, list(rf.sort_values("yield_rata2").index)), width="stretch")
        pilih = st.selectbox("Lihat sebaran satu jenis", jenis)
        st.plotly_chart(charts.hist_yield(dff[dff["jenis_buah"] == pilih], pilih), width="stretch")
    with t3:
        tabel = rf.reset_index()
        for col in ["yield_rata2", "yield_sd", "yield_p10", "yield_p90", "yield_cv", "ci95_poin"]:
            tabel[col] = (tabel[col] * 100).round(2)
        tabel = tabel.rename(columns={"jenis_buah": "Jenis", "kelompok": "Kelompok", "n": "Sampel", "rata2_berat_utuh_g": "Rata-rata berat utuh (g)",
                                      "yield_rata2": "Yield rata-rata (%)", "yield_sd": "SD (poin)", "yield_p10": "P10 (%)",
                                      "yield_p90": "P90 (%)", "yield_cv": "CV (%)", "ci95_poin": "Galat 95% (poin)"})
        st.dataframe(tabel, hide_index=True, width="stretch")
        st.download_button("Unduh tabel (CSV)", tabel.to_csv(index=False).encode("utf-8"), "ringkasan_yield.csv", "text/csv")


def halaman_kalkulator():
    df = dataset_aktif(); ring = _ring(df)
    style.hero("Kalkulator produksi", "Hitung cup yang bisa dibuat, atau buah yang harus dibeli untuk target tertentu.")
    banner_data()
    if "SIMULASI" in set(df["sumber_data"].astype(str)):
        style.catatan("**Angka di bawah memakai data simulasi.** Jangan dipakai untuk membeli stok sebelum diganti data timbangan nyata.")
    with st.container(border=True):
        c = st.columns(4)
        jenis = c[0].selectbox("Jenis buah", list(ring.index))
        takaran = c[1].number_input("Takaran per cup (g bagian layak)", min_value=1, value=config.TAKARAN_DEFAULT_G, step=10)
        mode = c[2].selectbox("Yield yang dipakai", ["Rata-rata", "Konservatif (P10)", "Manual"])
        kehilangan = c[3].number_input("Kehilangan tambahan (%)", min_value=0.0, max_value=50.0, value=0.0, step=1.0, format="%.0f",
                                       help="Blender, saringan, tumpah. Isi setelah diukur.") / 100
    baris = ring.loc[jenis]
    manual = None
    if mode == "Manual":
        manual = st.slider("Yield manual (%)", 1.0, 100.0, float(round(baris["yield_rata2"] * 100, 1)), 0.5) / 100
    y = pilih_yield(baris, mode, manual)
    st.caption(f"Yield dipakai: **{fmt_pct(y)}**. Dari {int(baris['n'])} sampel: rata-rata {fmt_pct(baris['yield_rata2'])}, "
               f"P10 {fmt_pct(baris['yield_p10'])}. Takaran awal {config.TAKARAN_DEFAULT_G} g adalah asumsi, ganti dengan resep cafe.")
    if baris["n"] < 5:
        st.warning("Kurang dari 5 sampel untuk jenis ini: estimasi kurang andal.")
    kiri, kanan = st.columns(2)
    with kiri, st.container(border=True):
        st.subheader("Berapa cup dari buah yang ada?")
        utuh = st.number_input("Berat buah utuh (g)", min_value=1.0, step=100.0,
                               value=float(max(100, round(baris["rata2_berat_utuh_g"] * 10, -2))), format="%.0f",
                               help="Nilai awal = sekitar 10 buah berukuran rata-rata. Ganti dengan berat buah yang tersedia.")
        layak, efektif, cup = estimasi_cup(utuh, y, takaran, kehilangan)
        style.hasil_besar("Estimasi cup yang bisa dibuat", f"{cup}", "cup",
                          f"Bagian layak <b>{fmt_num(layak, 0)} g</b> &rarr; setelah kehilangan <b>{fmt_num(efektif, 0)} g</b>")
    with kanan, st.container(border=True):
        st.subheader("Berapa buah untuk target cup?")
        target = st.number_input("Target jumlah cup", min_value=1, value=config.TARGET_CUP_DEFAULT, step=10)
        if kehilangan >= 1 or y <= 0:
            st.error("Parameter tidak valid.")
            return
        layak2, utuh_g, jumlah = kebutuhan_buah(target, y, takaran, kehilangan, float(baris["rata2_berat_utuh_g"]))
        style.hasil_besar("Buah yang perlu disiapkan (perkiraan)", f"{jumlah}", "buah",
                          f"Buah utuh <b>{fmt_num(utuh_g / 1000, 1)} kg</b>, bagian layak <b>{fmt_num(layak2 / 1000, 1)} kg</b>. "
                          "Berdasarkan rata-rata berat buah pada data.")
    st.subheader("Bandingkan skenario untuk target cup")
    baris_tabel = []
    for nama in ["Rata-rata", "Konservatif (P10)"]:
        yy = pilih_yield(baris, nama)
        _, ug, jm = kebutuhan_buah(target, yy, takaran, kehilangan, float(baris["rata2_berat_utuh_g"]))
        baris_tabel.append({"Skenario": nama, "Yield": f"{fmt_pct(yy)}", "Buah utuh (kg)": round(ug / 1000, 1), "Jumlah buah (perkiraan)": jm})
    st.dataframe(pd.DataFrame(baris_tabel), hide_index=True, width="stretch",
                 column_config={"Buah utuh (kg)": st.column_config.NumberColumn(format="%.1f")})


def halaman_data():
    df = dataset_aktif()
    style.hero("Data & pembaruan", "Unggah data timbangan nyata, unduh template, dan lihat arti tiap kolom.")
    banner_data()
    if st.session_state.get("pesan_data"):
        st.success(st.session_state.pop("pesan_data"))
    t1, t2, t3 = st.tabs(["Dataset aktif", "Unggah data baru", "Kamus data"])
    with t1:
        st.caption(f"Sedang dipakai: {st.session_state.get('nama_dataset', '')}")
        c = st.columns(3)
        c[0].metric("Baris", f"{len(df):,}".replace(",", "."))
        c[1].metric("Jenis", df["jenis_buah"].nunique())
        c[2].metric("Sumber data", ", ".join(sorted(df["sumber_data"].astype(str).unique())))
        a, b, _ = st.columns([1, 1, 2])
        a.download_button("Unduh template CSV", ("," .join(KOLOM_TEMPLATE) + "\n").encode("utf-8"), "template_data_buah.csv", "text/csv", width="stretch")
        b.download_button("Unduh dataset aktif", df.to_csv(index=False).encode("utf-8"), "dataset_aktif.csv", "text/csv", width="stretch")
        st.markdown("**Contoh 15 baris pertama**")
        st.dataframe(df.head(15), hide_index=True, width="stretch")
    with t2:
        st.caption("Unggah CSV hasil timbangan. Kolom wajib: jenis_buah, berat_buah_utuh_g, berat_bagian_layak_g. "
                   "Disarankan juga: sample_id, berat_sisa_g, sumber_data (SIMULASI atau NYATA). Unduh templatenya di tab Dataset aktif.")
        up = st.file_uploader("File CSV", type="csv", key="dataup")
        if up is not None:
            try:
                raw = pd.read_csv(up)
            except Exception as e:  # noqa: BLE001
                st.error(f"File tidak bisa dibaca: {e}")
                raw = None
            if raw is not None:
                if len(raw) > 200000:
                    st.error("File terlalu besar (maksimal 200.000 baris).")
                else:
                    baru, pesan, ok = validasi_dan_siapkan(raw)
                    for tingkat, teks in pesan:
                        getattr(st, tingkat)(teks)
                    if ok:
                        st.dataframe(baru.head(20), hide_index=True, width="stretch")
                        if st.button("Pakai data ini", type="primary"):
                            st.session_state["df_buah"] = baru
                            st.session_state["nama_dataset"] = f"Unggahan: {up.name}"
                            st.session_state["pesan_data"] = f"Dasbor sekarang memakai {up.name} ({len(baru):,} baris).".replace(",", ".")
                            st.rerun()
        if st.session_state.get("nama_dataset", "").startswith("Unggahan") and st.button("Kembalikan ke data bawaan"):
            st.session_state["df_buah"] = _bawaan()
            st.session_state["nama_dataset"] = "Data bawaan (dataset_buah_bersih.csv)"
            st.session_state["pesan_data"] = "Kembali ke data bawaan."
            st.rerun()
    with t3:
        from ui_temuan import tampilkan_kamus
        tampilkan_kamus()
