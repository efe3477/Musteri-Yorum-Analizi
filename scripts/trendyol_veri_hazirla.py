"""Gerçek veri hazırlığı: Doukan/trendyol-turkish-product-reviews -> analiz örneklemi.

    python scripts/trendyol_veri_hazirla.py data/raw/trendyol_reviews.csv

Adımlar
1. Yalnızca Giyim ve Elektronik kategorileri alınır, sütunlar standart adlara çevrilir.
   Kullanıcı adı ('user') hiç okunmaz (KVKK).
2. src.clean.temizle ile boş, tekrar eden, 10 karakterden kısa yorumlar ayıklanır.
3. Düşük puan (1-2 yıldız) tüm yorumların yalnızca ~%2'si olduğu için örneklem SONUCA GÖRE
   TABAKALI çekilir: her kategoriden 750 düşük (1-2) + 750 yüksek (4-5) puanlı yorum.
   Lojistik regresyonda bu tür örnekleme (case-control) konu katsayılarını / odds ratio'ları
   değiştirmez, yalnızca sabit terimi kaydırır (Prentice & Pyke, 1979). 3 yıldızlı yorumlar
   ana modelde zaten kullanılmadığı için örnekleme alınmaz.
4. Çıktılar:
   data/processed/orneklem.csv            analiz örneklemi (puan dahil)
   data/processed/populasyon_ozeti.csv     örneklemin çekildiği evrenin kategori x puan dağılımı
   data/processed/etiketleme/parca_XX.csv  LLM etiketlemesi için PUANSIZ parçalar
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

KOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK))
from src.clean import temizle  # noqa: E402
from src.config import ayarlari_yukle  # noqa: E402
from src.scraper import review_id_uret  # noqa: E402

KATEGORI = {"Giyim": "giyim", "Elektronik": "elektronik"}
HER_HUCRE = 750          # kategori x (düşük / yüksek)
PARCA_BOYU = 150


def urun_id(url: str) -> str:
    m = re.search(r"-p-(\d+)", str(url))
    return m.group(1) if m else str(url)


def main(girdi: str) -> None:
    ayar = ayarlari_yukle()
    seed = ayar["seed"]
    ham = pd.read_csv(girdi, usecols=["category", "product_url", "date", "point", "comment", "seller"])
    ham = ham[ham["category"].isin(KATEGORI)].copy()

    df = pd.DataFrame({
        "site": "trendyol",
        "kategori": ham["category"].map(KATEGORI),
        "yorum_metni": ham["comment"],
        "puan": ham["point"],
        "tarih": ham["date"],
        # Ürünü Trendyol'un kendisi satıyorsa 0, pazaryeri satıcısıysa 1
        "satici_var_mi": (ham["seller"].astype(str).str.strip() != "Trendyol").astype(int),
        "urun_id": ham["product_url"].map(urun_id),
    })
    df["review_id"] = [review_id_uret("trendyol", f"{u}|{m}", str(t))
                       for u, m, t in zip(df["urun_id"], df["yorum_metni"], df["tarih"])]

    temiz, _, rapor = temizle(df, ayar)
    # Ürün numarası review_id üzerinden eşlenir. Ham veride aynı review_id'ye sahip tekrar
    # kayıtlar olduğundan bu eşleme bazı satırları çoğaltır; teslim edilen örneklem
    # (data/processed/orneklem.csv) bu adımla çekilmiştir ve etiketler ona aittir. Örneklemde
    # tekrar eden yorumlar, analiz hattının temizlik adımında (src/clean.py) ayıklanır.
    temiz = temiz.drop(columns="urun_id").join(df.set_index("review_id")["urun_id"], on="review_id")
    print("Temizlik:", rapor)

    evren = (temiz.assign(grup=pd.cut(temiz["puan"], [0, 2, 3, 5], labels=["dusuk_1_2", "orta_3", "yuksek_4_5"]))
             .groupby(["kategori", "grup"], observed=True).size().unstack(fill_value=0))
    cikti = KOK / "data" / "processed"
    (cikti / "etiketleme").mkdir(parents=True, exist_ok=True)
    evren.to_csv(cikti / "populasyon_ozeti.csv")
    print(evren)

    parcalar = []
    for kat in KATEGORI.values():
        k = temiz[temiz["kategori"] == kat]
        parcalar.append(k[k["puan"] <= 2].sample(HER_HUCRE, random_state=seed))
        parcalar.append(k[k["puan"] >= 4].sample(HER_HUCRE, random_state=seed))
    orneklem = pd.concat(parcalar).sample(frac=1, random_state=seed).reset_index(drop=True)
    orneklem.insert(0, "kisa_id", [f"y{i:04d}" for i in range(1, len(orneklem) + 1)])
    orneklem.to_csv(cikti / "orneklem.csv", index=False)

    # LLM etiketlemesine puan GİTMEZ (hedef sızıntısı olmasın)
    for i, bas in enumerate(range(0, len(orneklem), PARCA_BOYU), 1):
        p = orneklem.iloc[bas:bas + PARCA_BOYU][["kisa_id", "kategori", "yorum_metni"]]
        p.to_csv(cikti / "etiketleme" / f"parca_{i:02d}.csv", index=False)
    print(f"Örneklem: {len(orneklem)} yorum, {i} parça -> {cikti}")


if __name__ == "__main__":
    main(sys.argv[1])
