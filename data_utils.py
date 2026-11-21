"""Validasi data buah, ringkasan per jenis, dan rumus kalkulator."""
import math
import numpy as np
import pandas as pd

WAJIB = ["jenis_buah", "berat_buah_utuh_g", "berat_bagian_layak_g"]
KOLOM_TEMPLATE = ["sample_id", "jenis_buah", "berat_buah_utuh_g", "berat_bagian_layak_g", "berat_sisa_g", "sumber_data"]


def fmt_pct(x, d=1):
    return f"{x * 100:.{d}f}%".replace(".", ",")


def fmt_num(x, d=1):
    return f"{x:,.{d}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def validasi_dan_siapkan(df_raw):
    """Kembalikan (df_bersih, pesan, ok). pesan = daftar (tingkat, teks)."""
    pesan = []
    df = df_raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    kurang = [c for c in WAJIB if c not in df.columns]
    if kurang:
        return None, [("error", f"Kolom wajib tidak ada: {', '.join(kurang)}")], False
    n0 = len(df)
    for c in ["berat_buah_utuh_g", "berat_bagian_layak_g", "berat_sisa_g"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["jenis_buah"] = df["jenis_buah"].astype(str).str.strip()
    buruk = df[WAJIB].isna().any(axis=1) | (df["jenis_buah"] == "") | (df["berat_buah_utuh_g"] <= 0) | (df["berat_bagian_layak_g"] < 0)
    layak_lebih = df["berat_bagian_layak_g"] > df["berat_buah_utuh_g"]
    if buruk.sum():
        pesan.append(("warning", f"{int(buruk.sum())} baris dibuang: nilai kosong, bukan angka, atau berat tidak valid."))
    if (layak_lebih & ~buruk).sum():
        pesan.append(("warning", f"{int((layak_lebih & ~buruk).sum())} baris dibuang: berat bagian layak lebih besar dari berat utuh."))
    df = df[~(buruk | layak_lebih)].copy()
    if df.empty:
        return None, pesan + [("error", "Tidak ada baris valid tersisa.")], False
    df["yield_pct"] = (df["berat_bagian_layak_g"] / df["berat_buah_utuh_g"]).round(4)
    if "sample_id" not in df.columns:
        df["sample_id"] = [f"ROW-{i + 1:05d}" for i in range(len(df))]
    if "kelompok" not in df.columns:
        df["kelompok"] = "Buah"
    if "sumber_data" not in df.columns:
        df["sumber_data"] = "TIDAK DIKETAHUI"
        pesan.append(("warning", "Kolom sumber_data tidak ada: diisi TIDAK DIKETAHUI. Isi SIMULASI atau NYATA agar label di dasbor benar."))
    if "berat_sisa_g" in df.columns:
        df["selisih_g"] = df["berat_buah_utuh_g"] - df["berat_bagian_layak_g"] - df["berat_sisa_g"]
    kecil = df.groupby("jenis_buah").size()
    kecil = kecil[kecil < 5]
    if len(kecil):
        pesan.append(("info", f"Jenis dengan kurang dari 5 sampel (estimasi kurang andal): {', '.join(kecil.index)}."))
    pesan.append(("success", f"{len(df):,} baris valid dari {n0:,}.".replace(",", ".")))
    return df.reset_index(drop=True), pesan, True


def ringkasan_per_jenis(df):
    g = df.groupby("jenis_buah")
    r = g.agg(
        kelompok=("kelompok", "first"),
        n=("sample_id", "count"),
        rata2_berat_utuh_g=("berat_buah_utuh_g", "mean"),
        yield_rata2=("yield_pct", "mean"),
        yield_sd=("yield_pct", "std"),
        yield_p10=("yield_pct", lambda s: s.quantile(0.10)),
        yield_p90=("yield_pct", lambda s: s.quantile(0.90)),
    )
    r["yield_sd"] = r["yield_sd"].fillna(0.0)
    r["yield_cv"] = r["yield_sd"] / r["yield_rata2"]
    r["ci95_poin"] = 1.96 * r["yield_sd"] / np.sqrt(r["n"])
    return r


def pilih_yield(baris, mode, manual=None):
    if mode == "Rata-rata":
        return float(baris["yield_rata2"])
    if mode == "Konservatif (P10)":
        return float(baris["yield_p10"])
    return float(manual)


def estimasi_cup(berat_utuh_g, y, takaran_g, kehilangan):
    layak = berat_utuh_g * y
    efektif = layak * (1 - kehilangan)
    cup = math.floor(efektif / takaran_g) if takaran_g > 0 else 0
    return layak, efektif, cup


def kebutuhan_buah(target_cup, y, takaran_g, kehilangan, rata2_berat_utuh_g):
    layak = target_cup * takaran_g / (1 - kehilangan)
    utuh_g = layak / y
    jumlah = math.ceil(utuh_g / rata2_berat_utuh_g) if rata2_berat_utuh_g > 0 else None
    return layak, utuh_g, jumlah


def metrik_klasifikasi(y_true, y_pred, labels):
    """Accuracy, precision/recall/F1 per kelas, macro-F1, dan confusion matrix (tanpa sklearn)."""
    y_true = pd.Series(list(y_true)); y_pred = pd.Series(list(y_pred))
    cm = pd.crosstab(y_true, y_pred).reindex(index=labels, columns=labels, fill_value=0)
    baris = []
    for l in labels:
        tp = cm.loc[l, l]; fp = cm[l].sum() - tp; fn = cm.loc[l].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        baris.append({"kelas": l, "precision": p, "recall": r, "f1": f, "jumlah_data": int(cm.loc[l].sum())})
    tabel = pd.DataFrame(baris)
    acc = float((y_true == y_pred).mean())
    return {"accuracy": acc, "macro_f1": float(tabel["f1"].mean()), "per_kelas": tabel, "confusion": cm, "n": len(y_true)}
