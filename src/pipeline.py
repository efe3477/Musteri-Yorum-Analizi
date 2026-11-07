"""Uçtan uca hat: topla -> temizle -> etiketle -> doğrula -> model -> rapor.

Örnekler:
    # API anahtarı olmadan, sentetik veriyle (hızlı deneme)
    python -m src.pipeline --kaynak ornek --etiketleyici kural

    # Kendi CSV'niz + LLM etiketleme + altın standart doğrulama
    python -m src.pipeline --kaynak csv --girdi data/raw/yorumlar.csv \
        --etiketleyici llm --altin data/gold/altin_standart.csv

    # URL listesinden toplama (kullanım koşullarını kontrol edin!)
    python -m src.pipeline --kaynak url --girdi data/raw/urller.csv --etiketleyici llm
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from . import clean, features, labeler, model, report, scraper, validate
from .config import CIKTI_DIZINI, ayarlari_yukle, konu_kodlari
from .ornek_veri import ornek_veri_uret

log = logging.getLogger("pipeline")


def veri_yukle(kaynak: str, girdi: str | None, ayar: dict) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Dönüş: (ham_veri, altin_standart_ya_da_None)"""
    if kaynak == "ornek":
        df = ornek_veri_uret(seed=ayar["seed"])
        altin = df[["review_id", "gercek_konular", "gercek_duygu"]].rename(
            columns={"gercek_konular": "konular", "gercek_duygu": "duygu"})
        return df.drop(columns=["gercek_konular", "gercek_duygu"]), altin
    if kaynak == "csv":
        return scraper.csv_ice_aktar(girdi), None
    if kaynak == "url":
        # girdi: 'url' ve 'kategori' sütunlu bir CSV
        u = pd.read_csv(girdi)
        parcalar = []
        for kat, grup in u.groupby("kategori"):
            d, hatalar = scraper.urlden_topla(grup["url"].tolist(), kat, ayar)
            for h in hatalar:
                log.warning("Hata: %s", h)
            parcalar.append(d)
        return pd.concat(parcalar, ignore_index=True), None
    raise ValueError(kaynak)


def calistir(kaynak: str, girdi: str | None, etiketleyici: str, altin_yolu: str | None,
             cikti: Path, uc_yildiz: str | None = None, sentetik: bool | None = None) -> dict:
    ayar = ayarlari_yukle()
    konular = konu_kodlari(ayar)
    cikti.mkdir(parents=True, exist_ok=True)
    if sentetik is None:  # örnek veri ya da data/sample içindeki dosya -> sentetik
        sentetik = kaynak == "ornek" or (girdi is not None and "sample" in Path(girdi).parts)

    log.info("1/6 Veri yükleniyor (%s)", kaynak)
    ham, altin = veri_yukle(kaynak, girdi, ayar)
    if altin_yolu:
        altin = pd.read_csv(altin_yolu)
    ham.to_csv(cikti / "ham_yorumlar.csv", index=False)

    log.info("2/6 Temizleniyor")
    temiz, kisa, t_rapor = clean.temizle(ham, ayar)
    temiz.to_csv(cikti / "temiz_yorumlar.csv", index=False)
    kisa.to_csv(cikti / "kisa_yorumlar.csv", index=False)
    log.info("   %s", t_rapor)

    log.info("3/6 Etiketleniyor (%s)", etiketleyici)
    if etiketleyici == "llm":
        et = labeler.LLMEtiketleyici(ayar, cikti / "cache" / "llm_etiketler.jsonl")
    elif etiketleyici == "hazir":
        et = labeler.HazirEtiketler(konular)
    else:
        et = labeler.KuralEtiketleyici(konular)
    etiketli = et.etiketle(temiz)
    etiketli.to_csv(cikti / "etiketli_yorumlar.csv", index=False)
    log.info("   etiketlenemeyen: %d", int(etiketli["etiketlenemedi"].sum()))

    d_ozet = None
    if altin is not None:
        log.info("4/6 Doğrulanıyor")
        altin = altin[altin["review_id"].isin(etiketli["review_id"])]
        if len(altin) > 300 and sentetik:
            altin = altin.sample(300, random_state=ayar["seed"])  # altın standart 300 yorum
        d_tablo, d_ozet = validate.dogrula(altin, etiketli[["review_id", "konular", "duygu"]], konular)
        d_tablo.to_csv(cikti / "dogrulama_konu_bazinda.csv", index=False)
        log.info("   %s", d_ozet)
    else:
        log.info("4/6 Altın standart verilmedi, doğrulama atlandı")

    log.info("5/6 Lojistik regresyon")
    ozellik = features.ozellikleri_hazirla(etiketli, ayar, konular, uc_yildiz)
    sonuclar = model.tum_kapsamlar(ozellik, ayar, konular)
    pd.concat([s["or_tablosu"] for s in sonuclar.values()]).to_csv(
        cikti / "odds_ratio_tablosu.csv", index=False)
    pd.DataFrame([s["metrikler"] for s in sonuclar.values()]).to_csv(
        cikti / "model_metrikleri.csv", index=False)
    pd.concat([s["vif"].assign(kapsam=k) for k, s in sonuclar.items()]).to_csv(
        cikti / "vif.csv", index=False)
    pd.concat([s["kontrol_tablosu"] for s in sonuclar.values()]).to_csv(
        cikti / "kontrol_degiskenleri.csv", index=False)

    log.info("6/6 Grafikler ve PDF")
    grafikler = []
    ana = report.atfedilebilir_pay_grafigi(sonuclar, cikti / "atfedilebilir_pay.png", sentetik)
    if ana:
        grafikler.append(ana)
    grafikler += [report.forest_plot(s["or_tablosu"], k, cikti / f"forest_{k}.png", sentetik)
                 for k, s in sonuclar.items()]
    karsi = report.kategori_karsilastirma(sonuclar, cikti / "kategori_karsilastirma.png", sentetik)
    if karsi:
        grafikler.append(karsi)
    grafikler.append(report.konu_dagilimi(etiketli, konular, cikti / "konu_dagilimi.png", sentetik))
    report.pdf_rapor(sonuclar, d_ozet, t_rapor, grafikler, cikti / "ozet_rapor.pdf", sentetik)

    ozet = {"kaynak": kaynak, "etiketleyici": etiketleyici, "sentetik": sentetik,
            "temizlik": t_rapor, "dogrulama": d_ozet,
            "metrikler": {k: s["metrikler"] for k, s in sonuclar.items()},
            "cikan_konular": {k: s["cikan_konular"] for k, s in sonuclar.items()},
            "yuksek_vif": {k: s["yuksek_vif"] for k, s in sonuclar.items()}}
    with open(cikti / "calisma_ozeti.json", "w", encoding="utf-8") as f:
        json.dump(ozet, f, ensure_ascii=False, indent=2, default=str)
    log.info("Bitti -> %s", cikti)
    return ozet


def main():
    p = argparse.ArgumentParser(description="Müşteri yorumu analizi hattı")
    p.add_argument("--kaynak", choices=["ornek", "csv", "url"], default="ornek")
    p.add_argument("--girdi", help="CSV yolu (csv: yorumlar, url: url+kategori listesi)")
    p.add_argument("--etiketleyici", choices=["kural", "llm", "hazir"], default="kural",
                   help="hazir: girdi CSV'de 'konular' ve 'duygu' sütunları zaten var")
    p.add_argument("--altin", help="İnsan etiketli altın standart CSV (review_id, konular, duygu)")
    p.add_argument("--uc-yildiz", choices=["cikar", "sifir"], help="duyarlılık analizi için")
    p.add_argument("--cikti", default=str(CIKTI_DIZINI / "son_calisma"))
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    calistir(a.kaynak, a.girdi, a.etiketleyici, a.altin, Path(a.cikti), a.uc_yildiz)


if __name__ == "__main__":
    main()
