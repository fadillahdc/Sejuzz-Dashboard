"""Membuat DATA DUMMY komentar pengunjung cafe jus (bukan komentar orang sungguhan).

Cara pembuatan: kalimat dirangkai dari templat (subjek + penilaian), sebagian dengan kata "tapi"
agar ambigu. Label asli ditentukan oleh penilaian pada klausa terakhir (klausa setelah "tapi").
Dua kelas saja (Positif/Negatif), sesuai model tim. Sebagian penilaian memakai kata di luar kamus skor MOCK sehingga ada kesalahan prediksi, agar
halaman evaluasi punya contoh yang tidak sempurna.

Keluaran:
  data/komentar_contoh.csv  : komentar, sumber_data=DUMMY
  data/evaluasi_contoh.csv  : komentar, label_asli, label_prediksi, bobot, sumber_data=DUMMY (hasil skor MOCK)
Jalankan dari folder dashboard:  python tools/buat_data_contoh_sentimen.py
"""
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sentiment_client as sc  # noqa: E402

random.seed(7)
SUBJEK = ["Jus alpukatnya", "Jus mangganya", "Jus semangkanya", "Jus melonnya", "Jus jambu merahnya", "Jus sirsaknya",
          "Jus buah naganya", "Tempatnya", "Pelayanannya", "Harganya", "Kasirnya", "Es jusnya", "Kafenya"]
POS = ["segar banget", "enak banget", "mantap", "manisnya pas", "murah", "ramah", "bersih", "nyaman", "cepat",
       "recommended", "juara", "suka banget", "puas", "bikin nagih", "pas banget di lidah", "bikin ketagihan"]
NEG = ["hambar", "kemahalan", "lama banget", "kotor", "kecewa", "pahit", "encer", "berisik", "sempit", "bau", "lambat",
       "jelek", "tidak enak", "nggak segar", "kurang manis", "bikin kapok", "males balik lagi"]


def buat_satu():
    r = random.random()
    s1, s2 = random.sample(SUBJEK, 2)
    if r < 0.40:
        return f"{s1} {random.choice(POS)}", "Positif"
    if r < 0.75:
        return f"{s1} {random.choice(NEG)}", "Negatif"
    if r < 0.88:
        return f"{s1} {random.choice(POS)} tapi {s2.lower()} {random.choice(NEG)}", "Negatif"
    return f"{s1} {random.choice(NEG)} tapi {s2.lower()} {random.choice(POS)}", "Positif"


def main(n=400):
    data = {}
    while len(data) < n:
        k, l = buat_satu()
        data.setdefault(k, l)
    df = pd.DataFrame({"komentar": list(data), "label_asli": list(data.values())})
    hasil = sc.SentimentClient(mode="mock").predict(df["komentar"])
    df["label_prediksi"] = [h["label"] for h in hasil]
    df["bobot"] = [h["bobot"] for h in hasil]
    df["sumber_data"] = "DUMMY"
    out = Path(__file__).resolve().parents[1] / "data"
    df[["komentar"]].assign(sumber_data="DUMMY").to_csv(out / "komentar_contoh.csv", index=False)
    df[["komentar", "label_asli", "label_prediksi", "bobot", "sumber_data"]].to_csv(out / "evaluasi_contoh.csv", index=False)
    acc = (df.label_asli == df.label_prediksi).mean()
    print(f"{len(df)} komentar dummy; kecocokan skor MOCK dengan label asli: {acc:.1%}")
    print(df.label_asli.value_counts().to_dict())


if __name__ == "__main__":
    main()
