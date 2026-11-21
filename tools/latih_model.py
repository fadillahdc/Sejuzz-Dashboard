"""Latih, uji, dan ekspor model dari dataset bawaan.  Jalankan dari folder dashboard:  python3 tools/latih_model.py
Keluaran: data/model_parameter.json (untuk Full-stack), data/model_perbandingan.csv, data/pembagian_data.csv."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

import config  # noqa: E402
import model_utils as mu  # noqa: E402
from data_utils import validasi_dan_siapkan  # noqa: E402

df, _, _ = validasi_dan_siapkan(pd.read_csv(config.DATASET_BUAH))
r = mu.evaluasi(df)
r["tabel"].to_csv(config.DATA_DIR / "model_perbandingan.csv", index=False)
r["bagian"][["sample_id", "jenis_buah", "bagian"]].to_csv(config.DATA_DIR / "pembagian_data.csv", index=False)
mu.simpan_parameter(r["model"], config.DATA_DIR / "model_parameter.json")
print(r["tabel"].round(3).to_string(index=False))
print("Model terpilih:", r["terbaik"])
