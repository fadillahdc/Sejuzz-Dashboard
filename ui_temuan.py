"""Halaman Temuan & rekomendasi: kesimpulan otomatis dari data aktif, plus laporan yang bisa diunduh."""
import pandas as pd
import streamlit as st

import config
import style
from data_utils import fmt_num, fmt_pct, kebutuhan_buah


def tabel_rekomendasi(ring, takaran_g, target_cup):
    """Yield acuan per jenis: konservatif (P10) bila CV di atas median, selain itu rata-rata."""
    batas = ring["yield_cv"].median()
    baris = []
    for jenis, r in ring.iterrows():
        pakai_p10 = r["yield_cv"] > batas
        y = r["yield_p10"] if pakai_p10 else r["yield_rata2"]
        _, utuh_g, jumlah = kebutuhan_buah(target_cup, y, takaran_g, 0.0, float(r["rata2_berat_utuh_g"]))
        _, utuh_rata, _ = kebutuhan_buah(target_cup, r["yield_rata2"], takaran_g, 0.0, float(r["rata2_berat_utuh_g"]))
        baris.append({"Jenis": jenis, "Acuan": "Konservatif (P10)" if pakai_p10 else "Rata-rata", "Yield acuan (%)": round(y * 100, 1),
                      "Buah utuh (kg)": round(utuh_g / 1000, 1), "Jumlah buah (perkiraan)": jumlah,
                      "Tambahan vs rata-rata (%)": round((utuh_g / utuh_rata - 1) * 100, 1)})
    return pd.DataFrame(baris).sort_values("Yield acuan (%)", ascending=False)


def susun_temuan(df, ring):
    """Kumpulkan angka kunci sebagai daftar kalimat (markdown tebal untuk angka)."""
    urut = ring.sort_values("yield_rata2", ascending=False)
    tinggi, rendah = urut.head(3), urut.tail(3).iloc[::-1]
    cv = ring.sort_values("yield_cv", ascending=False).head(3)
    sebut = lambda s, kol: ", ".join(f"{j} ({fmt_pct(v)})" for j, v in s[kol].items())
    hasil = [
        f"**Yield tertinggi:** {sebut(tinggi, 'yield_rata2')}. Hampir seluruh bagian buah bisa dijadikan jus.",
        f"**Yield terendah:** {sebut(rendah, 'yield_rata2')}. Untuk jenis ini, buah yang dibeli jauh lebih banyak dari volume jus yang jadi.",
        f"**Paling tidak stabil (CV tertinggi):** {sebut(cv, 'yield_cv')}. Hasil antarbuah berbeda jauh, sehingga estimasi stok lebih berisiko.",
    ]
    if ring["kelompok"].nunique() > 1:
        g = df.groupby("kelompok").agg(y=("yield_pct", "mean"), j=("jenis_buah", "nunique"))
        hasil.append("**Perbandingan kelompok:** " + ", ".join(f"{k} {fmt_pct(r.y)} ({int(r.j)} jenis)" for k, r in g.iterrows())
                     + " (rata-rata sampel). Kelompok dengan sedikit jenis kurang layak dibandingkan.")
    prem = (ring["yield_rata2"] / ring["yield_p10"] - 1).sort_values(ascending=False).head(3)
    hasil.append("**Selisih rata-rata vs P10 terbesar:** " + ", ".join(f"{j} (+{fmt_pct(v, 0)} buah)" for j, v in prem.items())
                 + ". Angka ini adalah tambahan buah bila ingin aman pada hari hasil buruk.")
    return hasil


def susun_kualitas(df):
    n = len(df)
    pesan = [f"**Jumlah sampel:** {n:,} dari {df['jenis_buah'].nunique()} jenis.".replace(",", ".")]
    for kol, nama in [("flag_outlier_yield", "outlier yield"), ("flag_outlier_berat", "outlier berat utuh"), ("flag_duplikat_nilai", "baris bernilai kembar")]:
        if kol in df.columns:
            pesan.append(f"**{nama.capitalize()}:** {int(df[kol].sum())} baris ({fmt_pct(df[kol].mean())}).")
    if "selisih_g" in df.columns:
        sel = df["selisih_g"].abs()
        pesan.append(f"**Konsistensi timbangan:** selisih utuh - layak - sisa rata-rata {fmt_num(sel.mean())} g, maksimum {fmt_num(sel.max(), 0)} g "
                     f"({fmt_pct((sel / df['berat_buah_utuh_g']).max())} dari berat utuh). Selisih terbesar ada pada buah besar.")
    pesan.append("**Sumber data:** " + ", ".join(sorted(df["sumber_data"].astype(str).unique())) + ".")
    return pesan


def laporan_markdown(df, ring, takaran_g, target_cup):
    sim = "SIMULASI" in set(df["sumber_data"].astype(str))
    rek = tabel_rekomendasi(ring, takaran_g, target_cup)
    baris = ["# Laporan Analisis Yield Buah", ""]
    if sim:
        baris += ["> **PERINGATAN:** data yang dianalisis adalah **SIMULASI**, bukan hasil timbangan nyata. "
                  "Angka di bawah menunjukkan cara kerja analisis, bukan kondisi sebenarnya di cafe.", ""]
    baris += ["## 1. Pertanyaan bisnis",
              "Berapa buah yang harus disiapkan agar target jumlah cup jus tercapai, dan jenis mana yang hasilnya paling tidak pasti?", "",
              "## 2. Data dan pengolahan",
              "- Satuan analisis: satu sampel buah (berat utuh, berat bagian layak, berat sisa).",
              "- Yield = berat bagian layak / berat utuh.",
              "- Baris tidak valid dibuang (kosong, bukan angka, berat tidak masuk akal, layak lebih besar dari utuh).",
              "- Penanda kualitas: outlier yield dan berat (batas 1,5 x IQR per jenis) dan nilai kembar.",
              "- Ringkasan per jenis: rata-rata, simpangan baku, P10, P90, koefisien variasi (CV), galat 95%.", "",
              "## 3. Pemeriksaan ketepatan data"] + [f"- {x}" for x in susun_kualitas(df)]
    baris += ["", "## 4. Temuan"] + [f"- {x}" for x in susun_temuan(df, ring)]
    baris += ["", f"## 5. Rekomendasi (target {target_cup} cup, takaran {takaran_g} g)",
              "Jenis dengan CV di atas median memakai yield konservatif (P10), sisanya rata-rata.", "",
              _md_tabel(rek), "",
              "## 6. Model prediksi (regresi berat bagian layak)"] + _bagian_model(df) + ["",
              "## 7. Keterbatasan",
              "- Takaran per cup adalah asumsi, ganti dengan resep cafe.",
              "- Yield bisa berubah menurut musim, tingkat kematangan, dan cara mengupas.",
              "- Jumlah buah memakai rata-rata berat buah pada data, bukan berat buah yang benar-benar dibeli.",
              "- Rekomendasi baru layak dipakai setelah data nyata (disarankan minimal 30 sampel per jenis) menggantikan data simulasi."]
    return "\n".join(baris) + "\n"


def _bagian_model(df):
    import model_utils as mu
    if df.groupby("jenis_buah").size().min() < 10:
        return ["Model tidak dilatih: ada jenis dengan kurang dari 10 sampel."]
    r = mu.evaluasi(df)
    t = r["tabel"].round(2)
    b = r["bagian"]
    isi = [f"Pembagian data per jenis {int(mu.PORSI_LATIH * 100)}:{int(round((1 - mu.PORSI_LATIH) * 100))} ({int((b['bagian'] == 'latih').sum())} latih, "
           f"{int((b['bagian'] == 'uji').sum())} uji), seed {mu.SEED}. Baris bernilai kembar selalu berada di sisi yang sama (tanpa kebocoran). "
           f"Model dipilih dengan validasi silang {mu.LIPATAN} lipatan pada data latih, data uji hanya untuk laporan.", "", _md_tabel(t), "",
           f"Model terpilih: **{r['terbaik']}**."]
    if r["terbaik"] == list(mu.MODEL)[0]:
        isi.append("Model yang lebih rumit tidak menurunkan galat, artinya yield hampir tidak bergantung pada ukuran buah pada data ini.")
    return isi


def _md_tabel(d):
    kepala = "| " + " | ".join(map(str, d.columns)) + " |"
    garis = "|" + "---|" * len(d.columns)
    isi = ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([kepala, garis] + isi)


def halaman_temuan():
    from ui_buah import dataset_aktif, _ring
    df = dataset_aktif(); ring = _ring(df)
    sim = "SIMULASI" in set(df["sumber_data"].astype(str))
    style.hero("Temuan & rekomendasi", "Kesimpulan otomatis dari data yang sedang aktif.",
               chips=[f"{len(df):,} sampel".replace(",", "."), f"{len(ring)} jenis"])
    c = st.columns(2)
    takaran = c[0].number_input("Takaran per cup (g)", 1, value=config.TAKARAN_DEFAULT_G, step=10, key="tk_temuan")
    target = c[1].number_input("Target cup", 1, value=config.TARGET_CUP_DEFAULT, step=10, key="tg_temuan")

    t1, t2, t3 = st.tabs(["Temuan utama", "Rekomendasi stok", "Kualitas data"])
    with t1:
        kiri, kanan = st.columns(2)
        for i, x in enumerate(susun_temuan(df, ring)):
            with (kiri if i % 2 == 0 else kanan):
                style.kartu(x)
    with t2:
        st.caption("Jenis dengan variasi yield di atas median memakai yield konservatif (P10), sisanya rata-rata.")
        rek = tabel_rekomendasi(ring, takaran, target)
        st.dataframe(rek, hide_index=True, width="stretch", column_config={
            "Yield acuan (%)": st.column_config.ProgressColumn("Yield acuan (%)", min_value=0, max_value=100, format="%.1f"),
            "Buah utuh (kg)": st.column_config.NumberColumn(format="%.1f"),
            "Tambahan vs rata-rata (%)": st.column_config.NumberColumn(format="%.1f")})
    with t3:
        kiri, kanan = st.columns(2)
        for i, x in enumerate(susun_kualitas(df)):
            with (kiri if i % 2 == 0 else kanan):
                style.kartu(x)
    st.download_button("Unduh laporan analisis (Markdown)", laporan_markdown(df, ring, takaran, target).encode("utf-8"),
                       "laporan_analisis.md", "text/markdown", type="primary")


def tampilkan_kamus():
    kamus = pd.read_csv(config.DATA_DIR / "kamus_data.csv")
    st.dataframe(kamus.rename(columns={"kolom": "Kolom", "tipe": "Tipe", "satuan": "Satuan", "deskripsi": "Deskripsi", "wajib_unggah": "Saat unggah"}),
                 hide_index=True, width="stretch")
    st.download_button("Unduh kamus data (CSV)", kamus.to_csv(index=False).encode("utf-8"), "kamus_data.csv", "text/csv")
