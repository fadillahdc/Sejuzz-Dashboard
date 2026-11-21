"""Sejuzz Dashboard: yield buah, kalkulator, temuan, dan sentimen pengunjung.
Jalankan:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="Sejuzz Dashboard", page_icon=":material/local_cafe:", layout="wide")

import config  # noqa: E402
import style  # noqa: E402
import ui_buah  # noqa: E402
import ui_model  # noqa: E402
import ui_sentimen  # noqa: E402
import ui_temuan  # noqa: E402

style.pasang_css()

HALAMAN = {
    "Ringkasan": ui_buah.halaman_ringkasan,
    "Yield per jenis": ui_buah.halaman_yield,
    "Kalkulator produksi": ui_buah.halaman_kalkulator,
    "Model prediksi": ui_model.halaman_model,
    "Temuan & rekomendasi": ui_temuan.halaman_temuan,
    "Sentimen pengunjung": ui_sentimen.halaman_sentimen,
    "Data & pembaruan": ui_buah.halaman_data,
}

df = ui_buah.dataset_aktif()
with st.sidebar:
    st.markdown('<div class="sidebar-brand">Sejuzz Dashboard</div><div class="sidebar-sub">Yield, stok, dan sentimen mini cafe jus</div>',
                unsafe_allow_html=True)
    pilih = st.radio("Halaman", list(HALAMAN), label_visibility="collapsed")
    st.divider()
    with st.expander("Pengaturan sentimen"):
        dasar = config.sentiment_settings_dasar()
        st.session_state.setdefault("mode_sesi", dasar["mode"])
        st.radio("Sumber skor", ["joblib", "api", "mock"], key="mode_sesi",
                 format_func=lambda m: {"joblib": "Model lokal (joblib)", "api": "API Sejuzz", "mock": "Contoh (MOCK)"}[m])
        if st.session_state["mode_sesi"] == "api":
            st.session_state.setdefault("url_sesi", dasar["url"] or config.DEFAULT_API_URL)
            st.text_input("Alamat API", key="url_sesi")
            st.text_input("API key (kosongkan bila tidak perlu)", type="password", key="api_key_sesi",
                          help="Hanya disimpan selama sesi ini dan tidak ditulis ke file.")
            st.caption("Lalu klik **Tes koneksi** di Sentimen pengunjung > Koneksi API.")
    st.caption(f"**Data buah:** {st.session_state.get('nama_dataset', '')}")
    mode = config.sentiment_settings()["mode"]
    st.caption("**Sentimen:** " + {"api": "API model aktif", "joblib": "model lokal (joblib)"}.get(mode, "mode contoh (MOCK)"))

HALAMAN[pilih]()
