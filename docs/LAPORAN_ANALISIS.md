# Laporan Analisis Yield Buah

> **PERINGATAN:** data yang dianalisis adalah **SIMULASI**, bukan hasil timbangan nyata. Angka di bawah menunjukkan cara kerja analisis, bukan kondisi sebenarnya di cafe.

## 1. Pertanyaan bisnis
Berapa buah yang harus disiapkan agar target jumlah cup jus tercapai, dan jenis mana yang hasilnya paling tidak pasti?

## 2. Data dan pengolahan
- Satuan analisis: satu sampel buah (berat utuh, berat bagian layak, berat sisa).
- Yield = berat bagian layak / berat utuh.
- Baris tidak valid dibuang (kosong, bukan angka, berat tidak masuk akal, layak lebih besar dari utuh).
- Penanda kualitas: outlier yield dan berat (batas 1,5 x IQR per jenis) dan nilai kembar.
- Ringkasan per jenis: rata-rata, simpangan baku, P10, P90, koefisien variasi (CV), galat 95%.

## 3. Pemeriksaan ketepatan data
- **Jumlah sampel:** 1.800 dari 18 jenis.
- **Outlier yield:** 27 baris (1,5%).
- **Outlier berat utuh:** 12 baris (0,7%).
- **Baris bernilai kembar:** 32 baris (1,8%).
- **Konsistensi timbangan:** selisih utuh - layak - sisa rata-rata 1,9 g, maksimum 51 g (1,7% dari berat utuh). Selisih terbesar ada pada buah besar.
- **Sumber data:** SIMULASI.

## 4. Temuan
- **Yield tertinggi:** Blackberry (95,9%), Blueberry (95,3%), Stroberi (93,8%). Hampir seluruh bagian buah bisa dijadikan jus.
- **Yield terendah:** Kelapa (25,8%), Nanas (51,0%), Melon (51,1%). Untuk jenis ini, buah yang dibeli jauh lebih banyak dari volume jus yang jadi.
- **Paling tidak stabil (CV tertinggi):** Kelapa (20,3%), Melon (8,2%), Nanas (7,9%). Hasil antarbuah berbeda jauh, sehingga estimasi stok lebih berisiko.
- **Perbandingan kelompok:** Buah 69,5% (17.0 jenis), Sayuran 88,0% (1.0 jenis) (rata-rata sampel). Kelompok dengan sedikit jenis kurang layak dibandingkan.
- **Selisih rata-rata vs P10 terbesar:** Kelapa (+40% buah), Melon (+11% buah), Nanas (+11% buah). Angka ini adalah tambahan buah bila ingin aman pada hari hasil buruk.

## 5. Rekomendasi (target 100 cup, takaran 200 g)
Jenis dengan CV di atas median memakai yield konservatif (P10), sisanya rata-rata.

| Jenis | Acuan | Yield acuan (%) | Buah utuh (kg) | Jumlah buah (perkiraan) | Tambahan vs rata-rata (%) |
|---|---|---|---|---|---|
| Blackberry | Rata-rata | 95.9 | 20.9 | 168 | 0.0 |
| Blueberry | Rata-rata | 95.3 | 21.0 | 167 | 0.0 |
| Stroberi | Rata-rata | 93.8 | 21.3 | 83 | 0.0 |
| Tomat | Rata-rata | 91.5 | 21.9 | 187 | 0.0 |
| Pakcoy | Rata-rata | 88.0 | 22.7 | 96 | 0.0 |
| Jambu Merah | Rata-rata | 78.2 | 25.6 | 125 | 0.0 |
| Jeruk | Rata-rata | 73.1 | 27.4 | 152 | 0.0 |
| Terong Belanda | Rata-rata | 70.1 | 28.5 | 326 | 0.0 |
| Alpukat | Konservatif (P10) | 69.1 | 28.9 | 118 | 7.0 |
| Mangga | Konservatif (P10) | 65.1 | 30.7 | 86 | 8.7 |
| Pisang | Rata-rata | 63.9 | 31.3 | 244 | 0.0 |
| Buah Naga | Konservatif (P10) | 61.9 | 32.3 | 83 | 8.5 |
| Sirsak | Konservatif (P10) | 61.0 | 32.8 | 45 | 9.6 |
| Pepaya | Konservatif (P10) | 57.3 | 34.9 | 29 | 8.3 |
| Semangka | Konservatif (P10) | 46.6 | 42.9 | 13 | 9.7 |
| Nanas | Konservatif (P10) | 46.0 | 43.5 | 36 | 10.9 |
| Melon | Konservatif (P10) | 45.9 | 43.6 | 37 | 11.2 |
| Kelapa | Konservatif (P10) | 18.4 | 108.6 | 124 | 39.9 |

## 6. Model prediksi (regresi berat bagian layak)
Pembagian data per jenis 80:20 (1440 latih, 360 uji), seed 42. Baris bernilai kembar selalu berada di sisi yang sama (tanpa kebocoran). Model dipilih dengan validasi silang 5 lipatan pada data latih, data uji hanya untuk laporan.

| Model | RMSE validasi silang (g) | MAE (g) uji | RMSE (g) uji | MAPE (%) uji | R2 uji |
|---|---|---|---|---|---|
| Baseline: yield rata-rata per jenis | 40.0 | 19.84 | 41.99 | 4.72 | 0.99 |
| Regresi linier per jenis | 40.41 | 19.99 | 42.29 | 4.79 | 0.99 |
| Regresi + fitur kuadrat (feature engineering) | 41.46 | 20.05 | 42.41 | 4.81 | 0.99 |

Model terpilih: **Baseline: yield rata-rata per jenis**.
Model yang lebih rumit tidak menurunkan galat, artinya yield hampir tidak bergantung pada ukuran buah pada data ini.

## 7. Keterbatasan
- Takaran per cup adalah asumsi, ganti dengan resep cafe.
- Yield bisa berubah menurut musim, tingkat kematangan, dan cara mengupas.
- Jumlah buah memakai rata-rata berat buah pada data, bukan berat buah yang benar-benar dibeli.
- Rekomendasi baru layak dipakai setelah data nyata (disarankan minimal 30 sampel per jenis) menggantikan data simulasi.
