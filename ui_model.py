"""Halaman Model prediksi: bandingkan model, hasil uji, dan coba prediksi."""
import streamlit as st

import charts
import config
import model_utils as mu
import style
from data_utils import fmt_num, fmt_pct


@st.cache_data(show_spinner="Melatih dan menguji model...")
def _evaluasi(df):
    return mu.evaluasi(df)


def halaman_model():
    from ui_buah import dataset_aktif
    df = dataset_aktif()
    sim = "SIMULASI" in set(df["sumber_data"].astype(str))
    style.hero("Model prediksi", "Memprediksi berat bagian layak dari jenis dan berat buah utuh.",
               chips=["Regresi", "Data uji terpisah"], peringatan=["DATA SIMULASI: metrik belum mewakili kondisi cafe"] if sim else [])
    if df.groupby("jenis_buah").size().min() < 10:
        st.warning("Ada jenis dengan kurang dari 10 sampel. Model tidak dilatih supaya hasilnya tidak menyesatkan.")
        return
    r = _evaluasi(df)
    b = r["bagian"]
    t1, t2, t3, t4 = st.tabs(["Perbandingan model", "Hasil uji", "Coba prediksi", "Metode"])
    with t1:
        k = st.columns(3)
        k[0].metric("Data latih", f"{int((b['bagian'] == 'latih').sum()):,}".replace(",", "."))
        k[1].metric("Data uji", f"{int((b['bagian'] == 'uji').sum()):,}".replace(",", "."))
        k[2].metric("Model terpilih", r["terbaik"].split(":")[0].split(" (")[0])
        tabel = r["tabel"].copy()
        tabel.insert(1, "Dipilih", ["Terpilih" if m == r["terbaik"] else "" for m in tabel["Model"]])
        st.dataframe(tabel, hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="%.3f" if c.startswith("R2") else "%.1f") for c in tabel.columns if c not in ("Model", "Dipilih")})
        base = r["tabel"].iloc[0]["RMSE (g) uji"]
        terbaik_rmse = r["tabel"].set_index("Model").loc[r["terbaik"], "RMSE (g) uji"]
        if r["terbaik"] == list(mu.MODEL)[0]:
            st.info("Model sederhana (yield rata-rata per jenis) sudah cukup: regresi linier dan fitur kuadrat tidak menurunkan galat. "
                    "Artinya yield hampir tidak bergantung pada ukuran buah, jadi model yang lebih rumit tidak membantu.")
        else:
            st.success(f"Model terpilih menurunkan RMSE uji dari {fmt_num(base)} g (baseline) menjadi {fmt_num(terbaik_rmse)} g.")
        st.caption("Pemilihan model memakai RMSE validasi silang pada data latih. Angka uji hanya dilaporkan, tidak dipakai memilih. "
                   "R2 tinggi terutama karena berat antarjenis sangat berbeda, jadi lebih bermakna melihat MAE dan MAPE.")
    with t2:
        a, c = st.columns(2)
        a.plotly_chart(charts.scatter_aktual_prediksi(r["uji"]), width="stretch")
        c.plotly_chart(charts.bar_mae_jenis(r["per_jenis"]), width="stretch")
    with t3:
        ring_utuh = df.groupby("jenis_buah")["berat_buah_utuh_g"].mean()
        with st.container(border=True):
            c = st.columns(3)
            jenis = c[0].selectbox("Jenis", list(ring_utuh.index), key="pm_jenis")
            utuh = c[1].number_input("Berat buah utuh (g)", min_value=1.0, value=float(round(ring_utuh[jenis])), step=50.0, key=f"pm_utuh_{jenis}")
            takaran = c[2].number_input("Takaran per cup (g)", min_value=1, value=config.TAKARAN_DEFAULT_G, step=10, key="pm_takaran")
        rata = float(mu.prediksi(r["model"], jenis, utuh))
        kons = float(mu.prediksi(r["model"], jenis, utuh, konservatif=True))
        m = st.columns(4)
        m[0].metric("Prediksi bagian layak", f"{fmt_num(rata, 0)} g")
        m[1].metric("Perkiraan konservatif", f"{fmt_num(kons, 0)} g", help="Prediksi ditambah sisa (residu) P10 data latih: hari hasil buruk.")
        m[2].metric("Yield perkiraan", fmt_pct(rata / utuh))
        m[3].metric("Cup (konservatif)", int(kons // takaran))
        st.caption("Untuk integrasi ke aplikasi web, parameter model diekspor oleh `python3 tools/latih_model.py` ke `data/model_parameter.json`.")
    with t4:
        st.markdown(f"""
**Target:** berat bagian layak (g). **Fitur:** jenis buah dan berat buah utuh (g).

**Pembagian data.** Per jenis, {int(mu.PORSI_LATIH * 100)}% latih dan {int((1 - mu.PORSI_LATIH) * 100)}% uji, acak dengan seed {mu.SEED}.
Baris bernilai kembar (jenis, utuh, layak, sisa sama) dikelompokkan dan selalu berada di sisi yang sama, supaya tidak ada salinan baris
yang bocor dari data latih ke data uji.

**Model yang dibandingkan.**
1. *Baseline:* berat utuh x yield rata-rata jenis itu.
2. *Regresi linier per jenis:* layak = a + b x utuh.
3. *Regresi + fitur kuadrat:* menambah suku utuh kuadrat (feature engineering) untuk menangkap hubungan melengkung.

**Evaluasi.** MAE, RMSE, MAPE, dan R2 pada data uji. Pilihan model berdasarkan validasi silang {mu.LIPATAN} lipatan pada data latih.

**Keterbatasan.** Data bawaan simulasi, sehingga hubungan yield dengan ukuran buah mengikuti cara data dibuat, bukan kenyataan.
Ganti dengan data timbangan nyata lalu lihat apakah model yang lebih rumit mulai unggul.
""")
