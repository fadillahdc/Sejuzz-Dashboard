"""Buat docs/LAPORAN_ANALISIS.md dari dataset bawaan.  Jalankan dari folder dashboard:  python3 tools/buat_laporan.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

import config  # noqa: E402
from data_utils import ringkasan_per_jenis, validasi_dan_siapkan  # noqa: E402
from ui_temuan import laporan_markdown  # noqa: E402

df, _, _ = validasi_dan_siapkan(pd.read_csv(config.DATASET_BUAH))
out = config.BASE_DIR / "docs" / "LAPORAN_ANALISIS.md"
out.parent.mkdir(exist_ok=True)
out.write_text(laporan_markdown(df, ringkasan_per_jenis(df), config.TAKARAN_DEFAULT_G, config.TARGET_CUP_DEFAULT), encoding="utf-8")
print("Tersimpan:", out)
