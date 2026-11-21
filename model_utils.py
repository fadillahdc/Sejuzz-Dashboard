"""Model regresi: memprediksi berat bagian layak (g) dari jenis dan berat buah utuh (g).

Hanya memakai numpy dan pandas. Dipakai halaman Model prediksi, tools/latih_model.py, dan integrasi web.

Pembagian data (No 11): per jenis, 80% latih dan 20% uji, acak dengan seed tetap. Baris bernilai kembar
(jenis + utuh + layak + sisa sama) dikelompokkan dan selalu masuk ke sisi yang sama, supaya salinan baris
tidak bocor dari data latih ke data uji. Pemilihan model memakai validasi silang 5 lipatan pada data latih saja;
data uji hanya dipakai sekali untuk laporan akhir.
"""
import json

import numpy as np
import pandas as pd

SEED = 42
PORSI_LATIH = 0.8
LIPATAN = 5

MODEL = {
    "Baseline: yield rata-rata per jenis": "yield",
    "Regresi linier per jenis": "linier",
    "Regresi + fitur kuadrat (feature engineering)": "kuadrat",
}
KUNCI = ["jenis_buah", "berat_buah_utuh_g", "berat_bagian_layak_g"]


def _kelompok_id(df):
    """Nomor kelompok: baris kembar mendapat nomor sama."""
    kol = KUNCI + (["berat_sisa_g"] if "berat_sisa_g" in df.columns else [])
    return df.groupby(kol, sort=False).ngroup().to_numpy()


def bagi_data(df, seed=SEED, porsi_latih=PORSI_LATIH):
    """Kembalikan df dengan kolom 'bagian' (latih/uji) dan 'lipatan' (0..4, hanya bermakna pada data latih)."""
    d = df.reset_index(drop=True).copy()
    d["_g"] = _kelompok_id(d)
    rng = np.random.default_rng(seed)
    d["bagian"], d["lipatan"] = "latih", -1
    for _, idx in d.groupby("jenis_buah").groups.items():
        sub = d.loc[idx]
        grup = sub["_g"].unique()
        rng.shuffle(grup)
        ukuran = sub.groupby("_g").size().reindex(grup).to_numpy()
        batas = porsi_latih * ukuran.sum()
        akum = np.cumsum(ukuran)
        n_latih = int(np.searchsorted(akum, batas, side="right"))
        n_latih = min(max(n_latih, 1), len(grup) - 1) if len(grup) > 1 else len(grup)
        uji = set(grup[n_latih:])
        d.loc[idx, "bagian"] = np.where(sub["_g"].isin(uji), "uji", "latih")
        for k, g in enumerate(grup[:n_latih]):
            d.loc[d["_g"] == g, "lipatan"] = k % LIPATAN
    return d.drop(columns="_g")


def _fit_jenis(x, y, jenis_model):
    if jenis_model == "yield":
        return {"yield": float(np.mean(y / x))}
    derajat = 1 if jenis_model == "linier" else 2
    if len(x) <= derajat + 1:
        return {"yield": float(np.mean(y / x))}
    return {"koef": np.polyfit(x, y, derajat).tolist()}


def _hitung(p, x):
    if "yield" in p:
        return x * p["yield"]
    return np.polyval(p["koef"], x)


def latih(df_latih, jenis_model):
    """Parameter model per jenis, plus kuantil sisa (residu) P10 untuk perkiraan konservatif."""
    params = {}
    for jenis, g in df_latih.groupby("jenis_buah"):
        x, y = g["berat_buah_utuh_g"].to_numpy(float), g["berat_bagian_layak_g"].to_numpy(float)
        p = _fit_jenis(x, y, jenis_model)
        p["residu_p10"] = float(np.quantile(y - _hitung(p, x), 0.10))
        params[jenis] = p
    return {"model": jenis_model, "params": params}


def prediksi(model, jenis, utuh_g, konservatif=False):
    p = model["params"][jenis]
    y = np.asarray(_hitung(p, np.asarray(utuh_g, float)), float)
    if konservatif:
        y = y + p["residu_p10"]
    return np.clip(y, 0, np.asarray(utuh_g, float))


def prediksi_df(model, df, konservatif=False):
    out = np.empty(len(df))
    for jenis, idx in df.reset_index(drop=True).groupby("jenis_buah").groups.items():
        out[list(idx)] = prediksi(model, jenis, df.reset_index(drop=True).loc[idx, "berat_buah_utuh_g"].to_numpy(float), konservatif)
    return out


def metrik(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    e = y - yhat
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {"MAE (g)": float(np.abs(e).mean()), "RMSE (g)": float(np.sqrt((e ** 2).mean())),
            "MAPE (%)": float(np.mean(np.abs(e) / np.maximum(y, 1)) * 100), "R2": float(1 - (e ** 2).sum() / ss_tot) if ss_tot else float("nan")}


def cv_rmse(d_latih, jenis_model):
    """RMSE rata-rata validasi silang 5 lipatan pada data latih."""
    hasil = []
    for k in range(LIPATAN):
        tr, va = d_latih[d_latih["lipatan"] != k], d_latih[d_latih["lipatan"] == k]
        if va.empty:
            continue
        m = latih(tr, jenis_model)
        hasil.append(metrik(va["berat_bagian_layak_g"], prediksi_df(m, va))["RMSE (g)"])
    return float(np.mean(hasil))


def evaluasi(df):
    """Latih dan bandingkan semua model. Kembalikan dict: tabel, terbaik, model terbaik, data bagian, per_jenis."""
    d = bagi_data(df)
    tr, te = d[d["bagian"] == "latih"], d[d["bagian"] == "uji"]
    baris, model_jadi = [], {}
    for nama, kode in MODEL.items():
        m = latih(tr, kode)
        model_jadi[nama] = m
        yh = prediksi_df(m, te)
        baris.append({"Model": nama, "RMSE validasi silang (g)": cv_rmse(tr, kode), **{f"{k} uji": v for k, v in metrik(te["berat_bagian_layak_g"], yh).items()}})
    tabel = pd.DataFrame(baris)
    terbaik = tabel.loc[tabel["RMSE validasi silang (g)"].idxmin(), "Model"]
    te = te.assign(prediksi=prediksi_df(model_jadi[terbaik], te))
    te["galat_abs"] = (te["berat_bagian_layak_g"] - te["prediksi"]).abs()
    per_jenis = te.groupby("jenis_buah").agg(n_uji=("galat_abs", "size"), MAE_g=("galat_abs", "mean"),
                                             rata2_layak_g=("berat_bagian_layak_g", "mean")).assign(
        MAE_persen=lambda x: x["MAE_g"] / x["rata2_layak_g"] * 100)
    return {"tabel": tabel, "terbaik": terbaik, "model": model_jadi[terbaik], "semua_model": model_jadi, "bagian": d, "uji": te, "per_jenis": per_jenis}


def simpan_parameter(model, jalur):
    """Simpan parameter model ke JSON untuk tim Full-stack. Rumus tiap jenis: polinom (koef, pangkat tertinggi dulu) atau yield."""
    with open(jalur, "w", encoding="utf-8") as f:
        json.dump({"target": "berat_bagian_layak_g", "fitur": "berat_buah_utuh_g", **model}, f, indent=2, ensure_ascii=False)
