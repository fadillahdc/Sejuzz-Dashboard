"""Bangun data/dataset_buah_bersih.csv dari 18 file mentah di data/mentah/ (satu file per jenis).

Tahapan (data wrangling dan feature engineering):
 1. Gabungkan semua file, baca dengan utf-8-sig (file mentah memakai BOM).
 2. Buang kolom kondisi_buah (kosong di seluruh baris) dan baris tidak valid (kosong, berat <= 0, layak > utuh).
 3. Tambah kode_buah (awalan sample_id) dan kelompok (Sayuran untuk Pakcoy, selain itu Buah).
 4. Hitung yield_pct, selisih_g, selisih_pct (selisih_g / berat utuh).
 5. Tandai outlier yield dan berat utuh (di luar 1,5 x IQR per jenis) dan nilai kembar
    (kombinasi jenis, utuh, layak, sisa sama dengan baris lain).
Jalankan dari folder dashboard:  python3 tools/bangun_dataset.py [--tulis]
Tanpa --tulis hanya memeriksa apakah hasilnya sama dengan file bersih yang ada.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

import config  # noqa: E402

SAYURAN = {"Pakcoy"}
KOLOM = ["sample_id", "kode_buah", "jenis_buah", "kelompok", "berat_buah_utuh_g", "berat_bagian_layak_g", "berat_sisa_g", "yield_pct",
         "selisih_g", "selisih_pct", "flag_outlier_yield", "flag_outlier_berat", "flag_duplikat_nilai", "sumber_data"]


def _iqr_flag(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    i = q3 - q1
    return ((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)).astype(int)


def bangun(folder=config.DATA_DIR / "mentah"):
    df = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in sorted(Path(folder).glob("*.csv"))], ignore_index=True)
    n0 = len(df)
    df = df.drop(columns=[c for c in ["kondisi_buah"] if c in df.columns and df[c].isna().all()])
    num = ["berat_buah_utuh_g", "berat_bagian_layak_g", "berat_sisa_g"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    ok = df[num].notna().all(axis=1) & (df["berat_buah_utuh_g"] > 0) & (df["berat_bagian_layak_g"] <= df["berat_buah_utuh_g"])
    df = df[ok].copy()
    df["kode_buah"] = df["sample_id"].str.split("-").str[0]
    df["kelompok"] = df["jenis_buah"].map(lambda j: "Sayuran" if j in SAYURAN else "Buah")
    df["yield_pct"] = (df["berat_bagian_layak_g"] / df["berat_buah_utuh_g"]).round(4)
    df["selisih_g"] = df["berat_buah_utuh_g"] - df["berat_bagian_layak_g"] - df["berat_sisa_g"]
    df["selisih_pct"] = (df["selisih_g"] / df["berat_buah_utuh_g"]).round(4)
    g = df.groupby("jenis_buah")
    df["flag_outlier_yield"] = g["yield_pct"].transform(_iqr_flag)
    df["flag_outlier_berat"] = g["berat_buah_utuh_g"].transform(_iqr_flag)
    df["flag_duplikat_nilai"] = df.duplicated(["jenis_buah"] + num, keep=False).astype(int)
    return df[KOLOM].reset_index(drop=True), n0


if __name__ == "__main__":
    hasil, n0 = bangun()
    print(f"{n0} baris mentah -> {len(hasil)} baris bersih")
    lama = pd.read_csv(config.DATASET_BUAH)
    beda = [c for c in KOLOM if not (hasil[c].astype(str).values == lama[c].astype(str).values).all()] if len(lama) == len(hasil) else ["jumlah baris"]
    print("Sama dengan dataset_buah_bersih.csv" if not beda else f"BERBEDA pada: {beda}")
    if "--tulis" in sys.argv:
        hasil.to_csv(config.DATASET_BUAH, index=False)
        print("Ditulis ke", config.DATASET_BUAH)
