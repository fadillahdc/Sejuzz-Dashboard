"""Klien sentimen. Tiga mode:
  mock   : skor berbasis kamus sederhana (HANYA contoh agar tampilan bisa diuji)
  api    : memanggil Sejuzz AI Service (API tim Gen AI). Atur SENTIMENT_MODE=api dan SENTIMENT_API_URL (SENTIMENT_API_KEY hanya bila API memintanya dan kamu diizinkan memakainya).
  joblib : memuat model tim langsung dari model/model_sentimen.joblib (tanpa jaringan dan tanpa API key). Perlu scikit-learn.

Kontrak API (dari openapi.json layanan):
  POST {host}/api/v1/sentimen/batch   header X-API-Key   body {"daftar_teks": ["...", ...]}   (maks 100 teks, tiap teks maks 1000 karakter)
  jawaban {"jumlah", "jumlah_positif", "jumlah_negatif", "jumlah_tidak_terbaca",
           "hasil": [{"teks", "sentimen": "positif"|"negatif"|null, "skor", "yakin", "peringatan"}]}
  GET  {host}/api/v1/sentimen/info    info model dan kecocokan versi scikit-learn
sentimen null = kalimat tidak berisi kata yang bisa dianalisis; dasbor melewatinya dan menghitung jumlahnya.
"""
import functools
import re
import threading
from urllib.parse import urlparse

import requests

import config


class SentimentAPIError(Exception):
    pass


# ---------------- Mode MOCK ----------------
LEKSIKON = {
    "segar": 1.2, "seger": 1.2, "enak": 1.5, "mantap": 1.3, "mantul": 1.3, "manis": 0.8, "manisnya": 0.8, "murah": 1.0,
    "ramah": 1.2, "bersih": 1.0, "nyaman": 1.2, "cepat": 1.0, "recommended": 1.4, "rekomen": 1.4, "suka": 1.2,
    "puas": 1.3, "favorit": 1.3, "juara": 1.3, "worth": 1.0, "terbaik": 1.5, "keren": 1.0, "top": 1.0, "asik": 1.0,
    "lama": -1.2, "mahal": -1.1, "kemahalan": -1.2, "pahit": -1.0, "hambar": -1.3, "kotor": -1.5, "kecewa": -1.5,
    "mengecewakan": -1.5, "jelek": -1.3, "lambat": -1.2, "bau": -1.3, "encer": -1.0, "asem": -0.8, "berisik": -1.0,
    "sempit": -0.9, "panas": -0.8,
}
NEGASI = {"tidak", "nggak", "gak", "ga", "kurang", "bukan", "tak"}
KONTRAS = {"tapi", "namun", "tetapi", "cuma", "sayang"}
PENEGAS = {"banget", "sekali", "bgt"}


def skor_mock(teks):
    total, last, kontras, neg = 0.0, None, 1.0, 0
    for t in re.findall(r"[a-z']+", str(teks).lower()):
        if t in KONTRAS:
            kontras, last = 1.5, None
        elif t in NEGASI:
            neg = 2
        elif t in PENEGAS:
            if last is not None:
                total += last * 0.5
                last = None
        elif t in LEKSIKON:
            w = LEKSIKON[t] * (-0.8 if neg > 0 else 1.0) * kontras
            total += w; last = w; neg = 0
        elif neg > 0:
            neg -= 1
    return round(total, 2)


def label_dari_bobot(b):
    return "Negatif" if b < config.BATAS_NEGATIF else "Positif"


# ---------------- Mode API ----------------
_PETA_LABEL = {"positif": "Positif", "positive": "Positif", "pos": "Positif",
               "negatif": "Negatif", "negative": "Negatif", "neg": "Negatif"}
MAKS_PANJANG = 1000
CONTOH_KALIMAT = "Jusnya segar banget, pelayanannya ramah"


def _endpoint(url, tujuan):
    """Susun alamat lengkap dari host saja atau alamat apa pun milik layanan. tujuan: 'batch' atau 'info'."""
    u = (url or "").strip()
    if u and "://" not in u:
        u = "https://" + u
    p = urlparse(u)
    return f"{p.scheme}://{p.netloc}/api/v1/sentimen/{tujuan}"


def _parse_item(item):
    """Ubah satu hasil API menjadi {'label', 'bobot', ...}. label None = kalimat tidak terbaca model."""
    sent = item.get("sentimen")
    if sent is None:
        return {"label": None, "bobot": None, "yakin": None, "peringatan": item.get("peringatan")}
    label = _PETA_LABEL.get(str(sent).strip().lower())
    skor = item.get("skor")
    if label is None or skor is None:
        raise SentimentAPIError(f"Format hasil tidak dikenali (sentimen harus positif atau negatif, skor harus ada): {item}")
    bobot = abs(float(skor)) * (1 if label == "Positif" else -1)   # tanda bobot mengikuti label supaya konsisten di grafik
    return {"label": label, "bobot": round(bobot, 2) + 0.0, "yakin": item.get("yakin"), "peringatan": item.get("peringatan")}


def _pesan_status(r):
    kode = r.status_code
    if kode == 401:
        return "API menolak akses (401): API key tidak ada atau salah. Bila key tidak boleh dipakai, pilih Model lokal di sidebar > Pengaturan sentimen."
    if kode == 503:
        return "Model sentimen di server belum tersedia atau gagal dimuat (503). Hubungi tim Gen AI."
    if kode == 422:
        return f"API menolak input (422): {r.text[:200]}"
    return None


# ---------------- Mode JOBLIB ----------------
_BOLEH_MODUL = ("sklearn", "numpy", "scipy", "joblib", "collections")
_BOLEH_BUILTIN = {"set", "frozenset", "dict", "list", "tuple", "slice", "complex", "bytearray", "range", "object", "int", "float", "str", "bool", "bytes"}
_kunci_muat = threading.Lock()


@functools.lru_cache(maxsize=2)
def muat_model(path):
    """Muat berkas joblib dengan daftar izin ketat. File joblib/pickle bisa menjalankan kode saat dibuka, jadi
    hanya kelas scikit-learn, numpy, scipy, dan joblib yang diizinkan; selain itu ditolak sebelum dijalankan."""
    try:
        import joblib
        import joblib.numpy_pickle as npp
    except ImportError as e:
        raise SentimentAPIError("scikit-learn/joblib belum terpasang. Jalankan: pip install -r requirements.txt") from e
    asli = npp.NumpyUnpickler.find_class

    def ketat(self, modul, nama):
        if (modul == "builtins" and nama in _BOLEH_BUILTIN) or (modul, nama) == ("_codecs", "encode") or modul.split(".")[0] in _BOLEH_MODUL:
            return asli(self, modul, nama)
        raise SentimentAPIError(f"Berkas model ditolak karena memuat objek tidak aman: {modul}.{nama}")

    with _kunci_muat:
        npp.NumpyUnpickler.find_class = ketat
        try:
            return joblib.load(path)
        except FileNotFoundError as e:
            raise SentimentAPIError(f"Berkas model tidak ditemukan: {path}") from e
        except SentimentAPIError:
            raise
        except Exception as e:  # noqa: BLE001
            raise SentimentAPIError(f"Model gagal dimuat ({type(e).__name__}). Cek kecocokan versi scikit-learn: {e}") from e
        finally:
            npp.NumpyUnpickler.find_class = asli


def _predict_joblib(model, texts):
    """Skor = decision_function (di atas 0 = kelas positif). Kalimat tanpa fitur yang dikenali = tidak terbaca, seperti API."""
    clf = model[-1]
    if set(map(str, clf.classes_)) != {"negatif", "positif"}:
        raise SentimentAPIError(f"Kelas model tidak sesuai (harus negatif dan positif): {list(clf.classes_)}")
    X = model[:-1].transform(texts)
    skor = clf.decision_function(X)
    ada_fitur = X.getnnz(axis=1) > 0
    hasil = []
    for sk, ada in zip(skor, ada_fitur):
        if not ada:
            hasil.append({"label": None, "bobot": None, "yakin": None, "peringatan": "Tidak ada kata yang dikenali model."})
            continue
        sk = float(sk)
        label = "Positif" if sk > 0 else "Negatif"
        hasil.append({"label": label, "bobot": round(abs(sk) * (1 if sk > 0 else -1), 2) + 0.0, "yakin": abs(sk) >= 0.5, "peringatan": None})
    return hasil


class SentimentClient:
    def __init__(self, mode="mock", url="", api_key="", timeout=30.0, batch=100, model_path=""):
        self.mode, self.url, self.api_key, self.timeout, self.batch = mode, url, api_key, timeout, min(int(batch), 100)
        self.model_path = model_path or str(config.MODEL_SENTIMEN)

    @property
    def is_mock(self):
        return self.mode not in ("api", "joblib")

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["X-API-Key"] = str(self.api_key)
        return h

    def _kirim(self, metode, tujuan, **kw):
        if not self.url:
            raise SentimentAPIError("SENTIMENT_API_URL belum diisi.")
        try:
            r = requests.request(metode, _endpoint(self.url, tujuan), headers=self._headers(), timeout=self.timeout, **kw)
            pesan = _pesan_status(r)
            if pesan:
                raise SentimentAPIError(pesan)
            r.raise_for_status()
            return r.json()
        except requests.Timeout as e:
            raise SentimentAPIError(f"API tidak menjawab dalam {self.timeout:.0f} detik (server mungkin sedang bangun, coba lagi). "
                                    "Sementara itu pilih Model lokal di sidebar > Pengaturan sentimen.") from e
        except requests.ConnectionError as e:
            raise SentimentAPIError("Tidak bisa terhubung ke server API. Cek koneksi internet dan alamat API. "
                                    "Sementara itu pilih Model lokal di sidebar > Pengaturan sentimen.") from e
        except requests.RequestException as e:
            raise SentimentAPIError(f"Gagal menghubungi API: {e}") from e
        except ValueError as e:
            raise SentimentAPIError("Jawaban API bukan JSON.") from e

    def predict(self, texts):
        texts = [str(t) for t in texts]
        if self.is_mock:
            hasil = []
            for t in texts:
                b = skor_mock(t)
                hasil.append({"label": label_dari_bobot(b), "bobot": b})
            return hasil
        if self.mode == "joblib":
            return _predict_joblib(muat_model(self.model_path), [t[:MAKS_PANJANG] for t in texts])
        hasil = []
        for i in range(0, len(texts), self.batch):
            chunk = texts[i:i + self.batch]
            data = self._kirim("POST", "batch", json={"daftar_teks": [t[:MAKS_PANJANG] for t in chunk]})
            items = data.get("hasil") if isinstance(data, dict) else None
            if not isinstance(items, list) or len(items) != len(chunk):
                raise SentimentAPIError("Jumlah hasil dari API tidak sama dengan jumlah kalimat yang dikirim.")
            hasil.extend(_parse_item(it) for it in items)
        return hasil

    def info(self):
        """Info model (dari server untuk mode api, dari berkas untuk mode joblib), termasuk kecocokan versi scikit-learn."""
        if self.mode == "joblib":
            import sklearn
            m = muat_model(self.model_path)
            vec = m[0]
            return {"mode": "joblib", "berkas": self.model_path, "langkah": [type(x).__name__ for _, x in m.steps],
                    "kelas": [str(c) for c in m[-1].classes_], "jumlah_fitur": len(getattr(vec, "vocabulary_", {})),
                    "sklearn_dipakai_menyimpan": getattr(m, "__getstate__", lambda: {})().get("_sklearn_version"), "sklearn_terpasang": sklearn.__version__}
        return self._kirim("GET", "info")

    def tes_koneksi(self):
        return self.predict([CONTOH_KALIMAT])[0]


def get_client():
    s = config.sentiment_settings()
    return SentimentClient(mode=s["mode"], url=s["url"], api_key=s["api_key"], timeout=s["timeout"], model_path=s["model_path"])
