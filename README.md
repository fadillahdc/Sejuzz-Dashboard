# Sejuzz Dashboard

Dasbor analisis data (Data Science) untuk mini cafe jus **Sejuzz**. Menjawab dua pertanyaan: *berapa buah yang harus disiapkan agar target cup tercapai*,
dan *jenis mana yang hasilnya paling tidak pasti*. Isinya: analisis yield per jenis buah (rata-rata dan P10 sebagai angka konservatif), kalkulator produksi,
model regresi berat bagian layak, temuan dan rekomendasi stok, serta analisis sentimen komentar pengunjung memakai model tim.

Dibuat dengan Streamlit, Plotly, dan scikit-learn. Bagian dari capstone Sejuzz (peran: Data Science).

> **Penting:** data buah di folder `data/` adalah **SIMULASI** (bukan hasil timbangan nyata), dan komentar sentimen
> bawaan adalah **dummy** dengan skor **MOCK**. Dasbor menampilkan peringatan untuk hal ini. Ganti dengan data nyata
> sebelum angkanya dipakai untuk keputusan cafe.

## Menjalankan

```bash
cd Sejuzz-Dashboard
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```
Browser terbuka di http://localhost:8501.

## Halaman

| Halaman | Isi |
|---|---|
| Ringkasan | Indikator utama, peringkat yield, ketidakstabilan yield, cuplikan sentimen |
| Yield per jenis | Filter kelompok dan jenis; peringkat, sebaran (boxplot, histogram), tabel dan unduhan |
| Kalkulator produksi | Estimasi cup dari buah yang ada, kebutuhan buah untuk target cup, perbandingan yield rata-rata vs konservatif (P10) |
| Model prediksi | Regresi berat bagian layak: perbandingan 3 model, hasil data uji, coba prediksi (rata-rata dan konservatif), penjelasan metode |
| Temuan & rekomendasi | Temuan otomatis dari data aktif, tabel rekomendasi stok per jenis (acuan rata-rata atau P10), kualitas data, unduh laporan Markdown |
| Sentimen pengunjung | Ringkasan, contoh komentar (kalimat, bobot, label Positif/Negatif), coba kalimat, evaluasi model, koneksi API |
| Data & pembaruan | Unggah CSV baru (divalidasi), template, unduh data aktif, kamus data |

## Memperbarui data buah
Halaman **Data & pembaruan**: unggah CSV dengan kolom wajib `jenis_buah`, `berat_buah_utuh_g`, `berat_bagian_layak_g`
(disarankan `sample_id`, `berat_sisa_g`, `sumber_data` berisi SIMULASI atau NYATA). Baris tidak valid dibuang dan dilaporkan.
Unggahan berlaku untuk sesi itu saja. Untuk mengganti data bawaan, timpa `data/dataset_buah_bersih.csv`.

## Model sentimen: tiga mode
| Mode | Kapan | Pengaturan |
|---|---|---|
| `joblib` | **Bawaan** bila `model/model_sentimen.joblib` ada dan scikit-learn terpasang. Tanpa jaringan dan tanpa API key | `SENTIMENT_MODE = "joblib"` (opsional) |
| `api` | Memanggil Sejuzz AI Service di server | lihat di bawah |
| `mock` | Hanya untuk menguji tampilan (kamus kata sederhana) | `SENTIMENT_MODE = "mock"` |

Model lokal: pipeline scikit-learn (`TfidfVectorizer` karakter n-gram 2-5 + `LinearSVC` seimbang), kelas `negatif` dan `positif`.
Skor = `decision_function` (di atas 0 = positif, `yakin` bila |skor| >= 0,5), sama dengan arti skor di API. Kalimat tanpa fitur yang dikenali dianggap tidak terbaca dan dilewati.
Berkas joblib dimuat lewat pemuat ketat (hanya kelas scikit-learn, numpy, scipy, joblib) karena joblib/pickle bisa menjalankan kode berbahaya; berkas dari sumber lain ditolak.
Versi scikit-learn harus cocok dengan saat model disimpan (1.9.x); butuh Python 3.11 atau lebih baru. Cek lewat **Koneksi API > Cek info model**.

Sumber skor bisa diganti tanpa mengedit file: sidebar > **Pengaturan sentimen** (model lokal, API Sejuzz, atau contoh). Untuk API, alamat sudah terisi
dan API key (bila diperlukan) diketik di kolom sandi; nilainya hanya hidup selama sesi dan tidak disimpan ke file.

## Menyambungkan model sentimen lewat API (Sejuzz AI Service)
Opsional. Dasbor sudah disesuaikan dengan kontrak API dari `/openapi.json` layanan tim Gen AI.
1. Salin `.streamlit/secrets.toml.example` menjadi `.streamlit/secrets.toml` (file ini sudah ada di `.gitignore`, jangan di-commit) dan isi:
   ```toml
   SENTIMENT_MODE = "api"
   SENTIMENT_API_URL = "https://sejuzz-capstone-production-362bebe7.up.railway.app"
   SENTIMENT_API_KEY = ""   # kosongkan bila tidak diperlukan atau tidak diizinkan
   ```
   Langkah ini **opsional**: tanpa API, dasbor sudah memakai model yang sama dari berkas lokal (mode `joblib`). Endpoint dilindungi header `X-API-Key` bila server mengaktifkannya;
   bila key tidak tersedia atau tidak boleh dipakai, biarkan kosong dan pakai mode `joblib`. Jangan pernah menulis key di kode atau repositori; untuk deploy isi lewat Secrets platform.
2. Jalankan dasbor, buka **Sentimen pengunjung > Koneksi API**, klik **Tes koneksi** dan **Cek info model** (menampilkan kecocokan versi scikit-learn).
3. Kontrak yang dipakai:
   ```
   POST /api/v1/sentimen/batch   {"daftar_teks": ["jusnya segar", "lama banget"]}      maks 100 teks, tiap teks maks 1000 karakter
   jawaban  {"hasil": [{"teks": "...", "sentimen": "positif|negatif|null", "skor": 1.2, "yakin": true, "peringatan": null}], ...}
   ```
   Dasbor memecah komentar per 100 secara otomatis. Tanda bobot mengikuti label (negatif = bobot di bawah 0). `sentimen: null` berarti kalimat tidak berisi kata yang bisa dianalisis, dan komentar itu dilewati dengan jumlahnya ditampilkan. Kode status 401/422/503 diterjemahkan jadi pesan yang jelas.
4. Tes otomatis memakai API palsu yang mengikuti skema ini (`tests/test_smoke.py`). Sambungan ke server asli perlu dicoba sendiri.

## Evaluasi model
Tab **Evaluasi model** menghitung accuracy, precision, recall, F1, dan confusion matrix dari CSV hasil uji dengan kolom
`komentar, label_asli, label_prediksi` (opsional `bobot`). Simpan file dari tim model sebagai `data/evaluasi_model.csv`
atau unggah langsung. Pastikan data uji tidak dipakai saat melatih model. Tanpa file itu, yang tampil adalah evaluasi
CONTOH (skor MOCK pada komentar dummy), bukan akurasi model tim.

## Struktur
```
app.py              pintu masuk dan navigasi
config.py           pengaturan (jalur data, takaran awal, warna, pengaturan API)
data_utils.py       validasi data, ringkasan, rumus kalkulator, metrik klasifikasi
sentiment_client.py klien sentimen (mock atau API)
charts.py           grafik Plotly
style.py            tema CSS dan komponen tampilan (hero, kartu)
model_utils.py      split tanpa kebocoran, model regresi, evaluasi (numpy saja)
ui_model.py         halaman model prediksi
ui_temuan.py        halaman temuan & rekomendasi, laporan, kamus data
ui_buah.py          halaman data buah
ui_sentimen.py      halaman sentimen
data/mentah/        18 file mentah satu per jenis (sumber dataset bersih)
data/               dataset buah, model_parameter.json, model_perbandingan.csv, pembagian_data.csv, komentar dummy, evaluasi contoh, kamus_data.csv
docs/               LAPORAN_ANALISIS.md (dibuat oleh tools/buat_laporan.py)
tools/              bangun_dataset.py (mentah ke bersih), latih_model.py, buat_laporan.py, data dummy sentimen
tests/              uji asap (python3 -m pytest tests -q)
```

## Integrasi web (tim Full-stack)
Cara paling sederhana: jalankan dasbor sebagai layanan terpisah (mis. Streamlit Community Cloud atau server sendiri) lalu
tanam di aplikasi web dengan iframe (`?embed=true` pada URL menyembunyikan elemen bawaan Streamlit). Rahasia API cukup
disimpan di secrets platform, bukan di repositori.

## Batasan
- Data buah simulasi; takaran awal 200 g per cup hanya asumsi dari contoh brief.
- Model prediksi dilatih pada data simulasi: baseline yield rata-rata menang, regresi lebih rumit tidak membantu. Ulangi evaluasi setelah data nyata ada.
- Skor MOCK dan komentar dummy hanya untuk menguji tampilan.
- Aturan "bobot di bawah 0 = Negatif" dipakai oleh MOCK dan garis batas grafik (`BATAS_NEGATIF` di `config.py`). Bobot tepat 0 dianggap Positif (perlu dikonfirmasi tim model); label dari API asli dipakai apa adanya.

## Alur data dan model
```bash
python3 tools/bangun_dataset.py            # cek mentah -> bersih (tambah --tulis untuk menimpa)
python3 tools/latih_model.py               # latih, uji, ekspor data/model_parameter.json
python3 tools/buat_laporan.py              # perbarui docs/LAPORAN_ANALISIS.md
```
Parameter model (`model_parameter.json`): per jenis, `yield` (layak = utuh x yield) atau `koef` (polinom, pangkat tertinggi dulu;
layak = np.polyval(koef, utuh)). `residu_p10` ditambahkan untuk perkiraan konservatif. Hasil dibatasi antara 0 dan berat utuh.

## Deploy ke Streamlit Community Cloud
1. Upload isi folder ini ke repositori GitHub (`app.py` di bagian paling atas repo; `.streamlit/secrets.toml` tidak ikut karena ada di `.gitignore`).
2. Di share.streamlit.io pilih **Create app**, pilih repo dan branch, main file path `app.py`.
3. Di **Advanced settings** pilih Python 3.12 (minimal 3.11).
4. Secrets dikosongkan; dasbor memakai model lokal. Setelah jalan, sidebar harus menulis "Sentimen: model lokal (joblib)".
Catatan: file yang diunggah di halaman Data & pembaruan hanya hidup selama sesi, dan model `.joblib` ikut terlihat bila repo publik.

## Menjalankan sebagai layanan (Docker)
```bash
docker build -t sejuzz-dashboard . && docker run -p 8501:8501 sejuzz-dashboard
```

## Dokumentasi untuk checklist DS
- Status semua jobdesk DS beserta buktinya: `docs/CHECKLIST_DS.md`.
- Kamus data: `data/kamus_data.csv` (juga tampil di halaman Data & pembaruan).
- Proses, pemeriksaan data, temuan, rekomendasi: `docs/LAPORAN_ANALISIS.md`. Buat ulang setelah data diganti: `python3 tools/buat_laporan.py`.
- Aturan flag: outlier = di luar 1,5 x IQR per jenis; duplikat = kombinasi jenis, berat utuh, layak, dan sisa kembar.
