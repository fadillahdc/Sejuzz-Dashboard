"""Konfigurasi dasbor. Pengaturan yang mungkin berubah dikumpulkan di sini."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATASET_BUAH = DATA_DIR / "dataset_buah_bersih.csv"
KOMENTAR_CONTOH = DATA_DIR / "komentar_contoh.csv"      # komentar dummy (kolom: komentar)
EVALUASI_CONTOH = DATA_DIR / "evaluasi_contoh.csv"      # evaluasi dummy (bukan model tim)
EVALUASI_MODEL = DATA_DIR / "evaluasi_model.csv"        # taruh hasil uji model tim di sini

TAKARAN_DEFAULT_G = 200      # ASUMSI dari contoh brief; ganti dengan takaran resep cafe
TARGET_CUP_DEFAULT = 100
MAX_BARIS_UNGGAH = 5000

LABELS = ("Positif", "Negatif")   # model tim memakai 2 kelas, tanpa Netral
BIRU, ORANYE, ABU = "#2a78d6", "#eb6834", "#8f8e88"
WARNA = {"Positif": BIRU, "Negatif": ORANYE}

# Aturan label model tim: bobot di bawah 0 = Negatif, selebihnya Positif.
# Dipakai oleh skor MOCK dan garis batas di grafik. Dengan API asli, label berasal dari model tim.
BATAS_NEGATIF = 0.0


def get_setting(name, default=None):
    """Baca dari variabel lingkungan, lalu .streamlit/secrets.toml."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default


MODEL_SENTIMEN = BASE_DIR / "model" / "model_sentimen.joblib"


def _ada_sklearn():
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


def sentimen_nyata(mode):
    """True bila skor berasal dari model tim (api atau joblib), bukan MOCK."""
    return mode in ("api", "joblib")


DEFAULT_API_URL = "https://sejuzz-capstone-production-362bebe7.up.railway.app"


def sentiment_settings_dasar():
    """Pengaturan dari variabel lingkungan, secrets.toml, atau bawaan (tanpa pilihan sesi di sidebar)."""
    mode = str(get_setting("SENTIMENT_MODE", "") or "").lower()
    if not mode:   # tanpa pengaturan: pakai model lokal bila berkasnya ada dan scikit-learn terpasang
        mode = "joblib" if MODEL_SENTIMEN.exists() and _ada_sklearn() else "mock"
    return {
        "mode": mode,                                   # "mock", "api", atau "joblib"
        "model_path": str(get_setting("SENTIMENT_MODEL_PATH", str(MODEL_SENTIMEN))),
        "url": get_setting("SENTIMENT_API_URL", DEFAULT_API_URL),
        "api_key": get_setting("SENTIMENT_API_KEY", ""),
        "timeout": float(get_setting("SENTIMENT_API_TIMEOUT", 30)),
    }



def sentiment_settings():
    """Pengaturan efektif: dasar + pilihan yang dibuat di sidebar untuk sesi ini (mode, URL, API key)."""
    s = sentiment_settings_dasar()
    try:
        import streamlit as st
        ss = st.session_state
        for kunci, kolom in (("mode_sesi", "mode"), ("url_sesi", "url"), ("api_key_sesi", "api_key")):
            if ss.get(kunci):
                s[kolom] = ss[kunci].strip() if isinstance(ss[kunci], str) else ss[kunci]
    except Exception:  # noqa: BLE001
        pass
    return s