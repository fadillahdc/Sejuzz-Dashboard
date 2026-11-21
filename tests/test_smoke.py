"""Uji asap: setiap halaman harus tampil tanpa error. Jalankan dari folder dashboard:  python -m pytest tests -q"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
HALAMAN = ["Ringkasan", "Yield per jenis", "Kalkulator produksi", "Model prediksi", "Temuan & rekomendasi", "Sentimen pengunjung", "Data & pembaruan"]


@pytest.mark.parametrize("halaman", HALAMAN)
def test_halaman_tampil(halaman):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value(halaman).run()
    assert not at.exception, [e.value for e in at.exception]


def test_pipeline_mentah_sama_dengan_bersih():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import pandas as pd
    from tools.bangun_dataset import bangun, KOLOM
    import config
    baru, _ = bangun()
    lama = pd.read_csv(config.DATASET_BUAH)
    assert len(baru) == len(lama)
    for c in KOLOM:
        assert (baru[c].astype(str).values == lama[c].astype(str).values).all(), c


def test_split_tanpa_kebocoran_dan_model_jalan():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import pandas as pd
    import config
    import model_utils as mu
    df = pd.read_csv(config.DATASET_BUAH)
    b = mu.bagi_data(df)
    kunci = b.groupby(["jenis_buah", "berat_buah_utuh_g", "berat_bagian_layak_g", "berat_sisa_g"]).bagian.nunique()
    assert (kunci == 1).all()                       # baris kembar tidak terpisah antara latih dan uji
    assert set(b["bagian"]) == {"latih", "uji"}
    r = mu.evaluasi(df)
    assert r["tabel"]["RMSE (g) uji"].notna().all()
    assert (mu.prediksi(r["model"], "Alpukat", 250) <= 250)


# ---------- Sambungan ke Sejuzz AI Service (API palsu mengikuti openapi.json) ----------
class _Resp:
    def __init__(self, kode, data):
        self.status_code, self._d, self.text = kode, data, str(data)

    def json(self):
        return self._d

    def raise_for_status(self):
        import requests
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


def _api_palsu(kunci="rahasia", panggilan=None):
    def fake(metode, url, headers=None, timeout=None, json=None, **kw):
        if panggilan is not None:
            panggilan.append((metode, url, dict(headers or {}), json))
        if (headers or {}).get("X-API-Key") != kunci:
            return _Resp(401, {"detail": "API key tidak valid"})
        if url.endswith("/api/v1/sentimen/info"):
            return _Resp(200, {"model": "contoh", "sklearn": "1.5.0"})
        assert url.endswith("/api/v1/sentimen/batch")
        assert len(json["daftar_teks"]) <= 100 and all(len(t) <= 1000 for t in json["daftar_teks"])
        hasil = []
        for t in json["daftar_teks"]:
            if t.startswith("???"):
                hasil.append({"teks": t, "sentimen": None, "skor": None, "yakin": None, "peringatan": "tidak ada kata dikenali"})
            elif any(k in t.lower() for k in ("lama", "mahal", "jelek")):
                hasil.append({"teks": t, "sentimen": "negatif", "skor": 1.3, "yakin": True, "peringatan": None})
            else:
                hasil.append({"teks": t, "sentimen": "positif", "skor": 0.3, "yakin": False, "peringatan": None})
        return _Resp(200, {"jumlah": len(hasil), "jumlah_positif": 0, "jumlah_negatif": 0, "jumlah_tidak_terbaca": 0, "hasil": hasil})
    return fake


def test_klien_api_sesuai_kontrak(monkeypatch):
    import requests
    from sentiment_client import SentimentClient, SentimentAPIError
    log = []
    monkeypatch.setattr(requests, "request", _api_palsu(panggilan=log))
    c = SentimentClient("api", "sejuzz-capstone-production-362bebe7.up.railway.app/docs", "rahasia")
    teks = ["jusnya segar", "lama banget", "???"] + ["enak"] * 150 + ["x" * 2000]
    h = c.predict(teks)
    assert len(h) == len(teks)
    assert h[0]["label"] == "Positif" and h[0]["bobot"] > 0
    assert h[1]["label"] == "Negatif" and h[1]["bobot"] < 0 and h[1]["bobot"] == -1.3   # tanda bobot mengikuti label
    assert h[2]["label"] is None and h[2]["bobot"] is None                               # tidak terbaca
    assert [x for x in log if x[0] == "POST"].__len__() == 2                              # 154 teks dipecah 100 + 54
    assert all(x[1] == "https://sejuzz-capstone-production-362bebe7.up.railway.app/api/v1/sentimen/batch" for x in log)
    assert c.info()["sklearn"] == "1.5.0"
    with __import__("pytest").raises(SentimentAPIError, match="401"):
        SentimentClient("api", "https://x.app", "salah").predict(["a"])


def test_halaman_sentimen_mode_api(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "request", _api_palsu())
    monkeypatch.setenv("SENTIMENT_MODE", "api")
    monkeypatch.setenv("SENTIMENT_API_URL", "https://sejuzz-capstone-production-362bebe7.up.railway.app")
    monkeypatch.setenv("SENTIMENT_API_KEY", "rahasia")
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60).run()
    at.sidebar.radio[0].set_value("Sentimen pengunjung").run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


# ---------- Mode model lokal (joblib) ----------
def test_mode_joblib_memakai_model_tim():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sentiment_client import SentimentClient
    c = SentimentClient("joblib")
    h = c.predict(["Jusnya segar banget, pelayanannya ramah", "lama banget nunggunya, kecewa", "???"])
    assert h[0]["label"] == "Positif" and h[0]["bobot"] > 0
    assert h[1]["label"] == "Negatif" and h[1]["bobot"] < 0
    assert h[2]["label"] is None                       # tidak ada fitur yang dikenali
    info = c.info()
    assert info["kelas"] == ["negatif", "positif"] and info["jumlah_fitur"] > 1000


def test_pemuat_joblib_menolak_berkas_berbahaya(tmp_path):
    import os
    import sys
    import joblib
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sentiment_client import SentimentAPIError, muat_model
    tanda = tmp_path / "TERBUKA.txt"

    class Jahat:
        def __reduce__(self):
            return (os.system, (f"touch {tanda}",))

    berkas = tmp_path / "jahat.joblib"
    joblib.dump(Jahat(), berkas)
    with pytest.raises(SentimentAPIError, match="tidak aman"):
        muat_model(str(berkas))
    assert not tanda.exists()                          # perintah tidak sempat dijalankan


def test_halaman_sentimen_mode_joblib(monkeypatch):
    monkeypatch.setenv("SENTIMENT_MODE", "joblib")
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=90).run()
    at.sidebar.radio[0].set_value("Sentimen pengunjung").run()
    assert not at.exception, [e.value for e in at.exception]
    assert not any("MODE CONTOH" in w.value for w in at.warning)


# ---------- Pengaturan sentimen di sidebar ----------
def test_sidebar_pilih_api_dengan_key(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "request", _api_palsu(kunci="rahasia"))
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=90).run()
    at.session_state["mode_sesi"] = "api"
    at.session_state["api_key_sesi"] = "rahasia"
    at.run()
    at.sidebar.radio[0].set_value("Sentimen pengunjung").run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert any("API" in x.value for x in at.success)


def test_sidebar_api_key_salah_menampilkan_pesan_jelas(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "request", _api_palsu(kunci="rahasia"))
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=90).run()
    at.session_state["mode_sesi"] = "api"
    at.session_state["api_key_sesi"] = "salah"
    at.run()
    at.sidebar.radio[0].set_value("Sentimen pengunjung").run()
    assert not at.exception
    assert any("401" in e.value for e in at.error)
