# Checklist Data Scientist: status dan bukti

Status: **Selesai** = ada bukti di repositori. **Sebagian** = bagian kami selesai, sisanya butuh tim lain. **Belum** = tidak bisa dikerjakan dari sisi kode.

| No | Jobdesk | Wajib | Status | Bukti / catatan |
|---|---|---|---|---|
| 1 | Identifikasi masalah UMKM bersama tim | Wajib | Sebagian | Rumusan di `LAPORAN_ANALISIS.md` bagian 1. Perlu dikonfirmasi tim |
| 2 | Pertanyaan bisnis | Wajib | Selesai | `LAPORAN_ANALISIS.md` bagian 1 |
| 4 | Sepakati kebutuhan dan format data dengan Gen AI | Opsional | Sebagian | Dasbor memakai model tim dari `model/model_sentimen.joblib` (mode joblib) dan juga siap memakai API Sejuzz (butuh API key). Data simulasi diberi label jelas |
| 5 | Prioritaskan data primer UMKM | Opsional | **Belum** | Belum ada data nyata. Data bawaan berlabel SIMULASI. Ganti lewat halaman Data & pembaruan atau `data/mentah/` |
| 8 | Data wrangling | Wajib | Selesai | `tools/bangun_dataset.py` (mentah ke bersih, diuji di `tests/`) |
| 9 | Pelabelan, transformasi, feature engineering | Opsional | Selesai | `yield_pct`, `selisih_g/pct`, flag outlier dan duplikat, fitur kuadrat pada model |
| 10 | Kamus data | Opsional | Selesai | `data/kamus_data.csv`, tampil di halaman Data & pembaruan |
| 11 | Split train/test tanpa kebocoran | Opsional | Selesai | `model_utils.bagi_data`, `data/pembagian_data.csv`, tes kebocoran |
| 12 | Analisis untuk jawab pertanyaan bisnis | Wajib | Selesai | Halaman Yield, Kalkulator, Temuan; `LAPORAN_ANALISIS.md` |
| 13 | Periksa ketepatan data dan hasil | Wajib | Selesai | `LAPORAN_ANALISIS.md` bagian 3, flag kualitas, 9 tes otomatis |
| 14 | Serah-terima data dan cara pakai | Wajib | Selesai | `README.md`, `kamus_data.csv`, `model_parameter.json` |
| 16 | Temuan dan rekomendasi | Wajib | Selesai | Halaman Temuan & rekomendasi, `LAPORAN_ANALISIS.md` bagian 4 dan 5 |
| 17 | Model regresi | Opsional | Selesai | `model_utils.py`, halaman Model prediksi |
| 18 | Feature engineering dan pengaruhnya | Opsional | Selesai | Linier vs linier + kuadrat, tabel perbandingan |
| 19 | Uji pada data tak dipakai latih | Opsional | Selesai | Data uji 20%, MAE, RMSE, MAPE, R2 |
| 20 | Bandingkan model dan keterbatasan | Opsional | Selesai | 3 model, halaman Model prediksi tab Metode |
| 27 | Dasbor interaktif | Opsional | Selesai | Streamlit, 7 halaman |
| 28 | Indikator bisnis dan filter | Opsional | Selesai | KPI, filter kelompok dan jenis |
| 29 | Pembaruan data (unggah) | Opsional | Selesai | Halaman Data & pembaruan |
| 38 | Integrasi prediksi ke web (Full-stack) | Opsional | Sebagian | Siap: `model_parameter.json` dan `model_utils.prediksi`. Penyambungan oleh Full-stack |
| 39 | Integrasi dasbor ke web (Full-stack) | Opsional | Sebagian | Siap: `Dockerfile`, mode `?embed=true`. Penyambungan oleh Full-stack |

## Asal data simulasi (wajib dijelaskan, No 4)
Data bawaan berlabel SIMULASI dan **belum ada catatan cara pembuatannya di repositori**. Tim DS isi bagian ini:
- Cara pembuatan data: `[ISI TIM: skrip, rentang nilai, asumsi distribusi]`
- Sumber acuan angka yield: `[ISI TIM]`
- Keterbatasan: yield simulasi tidak bergantung pada ukuran buah, sehingga model regresi tidak lebih baik dari rata-rata sederhana.
