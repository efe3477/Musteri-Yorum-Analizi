"""Duyarlılık analizi: yorum uzunluğu kontrol değişkeni olarak eklenirse sonuçlar değişir mi?

    python scripts/duyarlilik_analizi.py

Ana modelde yorum uzunluğu (log karakter sayısı) kontrol olarak KULLANILMAZ: şikâyet yorumu
uzatır (aracı değişken) ve kontrol edilirse konu etkileri olduğundan küçük görünür. Bu betik
aynı modelleri bir de log_uzunluk eklenmiş hâliyle kurar ve iki sonucu yan yana yazar.

Girdi : data/processed/etiketli_orneklem.csv
Çıktı : outputs/son_calisma/duyarlilik_uzunluk.csv

Not: Örneklemde 3 yıldızlı yorum olmadığı için (yalnızca 1–2 ve 4–5 çekildi) "3 yıldızı yüksek
puana ekleme" duyarlılık analizi bu veriyle yapılamaz; bunun için örnekleme 3 yıldızlı yorumlar da
alınmalıdır (pipeline: --uc-yildiz sifir).
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd

KOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK))
from src import clean, features, labeler, model  # noqa: E402
from src.config import ayarlari_yukle, konu_kodlari  # noqa: E402


def main() -> None:
    ayar = ayarlari_yukle()
    konular = konu_kodlari(ayar)
    ham = pd.read_csv(KOK / "data" / "processed" / "etiketli_orneklem.csv")
    temiz, _, _ = clean.temizle(ham, ayar)
    etiketli = labeler.HazirEtiketler(konular).etiketle(temiz)
    ozellik = features.ozellikleri_hazirla(etiketli, ayar, konular)

    ayar_uzun = copy.deepcopy(ayar)
    ayar_uzun["model"]["kontroller"] = [*ayar["model"]["kontroller"], "log_uzunluk"]

    satirlar = []
    for ad, a in (("ana", ayar), ("uzunluk_kontrollu", ayar_uzun)):
        for kapsam, s in model.tum_kapsamlar(ozellik, a, konular).items():
            t = s["or_tablosu"][["kapsam", "konu", "odds_ratio", "p_degeri", "atfedilebilir_pay"]]
            satirlar.append(t.assign(model=ad, roc_auc=s["metrikler"]["roc_auc"]))
    uzun = pd.concat(satirlar)
    tablo = uzun.pivot_table(index=["kapsam", "konu"], columns="model",
                             values=["odds_ratio", "p_degeri", "atfedilebilir_pay"]).reset_index()
    tablo.columns = ["_".join(c).strip("_") for c in tablo.columns]
    tablo["or_orani"] = tablo["odds_ratio_uzunluk_kontrollu"] / tablo["odds_ratio_ana"]
    tablo = tablo.sort_values(["kapsam", "atfedilebilir_pay_ana"], ascending=[True, False])
    yol = KOK / "outputs" / "son_calisma" / "duyarlilik_uzunluk.csv"
    tablo.to_csv(yol, index=False)

    auc = uzun.groupby(["kapsam", "model"])["roc_auc"].first().unstack()
    print(auc.round(3))
    print(tablo.round(3).to_string(index=False))
    print("->", yol)


if __name__ == "__main__":
    main()
