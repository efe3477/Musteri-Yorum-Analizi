"""UC-04 hazırlığı: elle etiketlenecek 300 yorumu rastgele seçer.

    python scripts/altin_ornek_sec.py outputs/son_calisma/temiz_yorumlar.csv

Çıktı: data/gold/etiketlenecek_300.csv
İki ekip üyesi 'konular' (noktalı virgülle: kargo_gecikme;paketleme) ve 'duygu'
(pozitif/notr/negatif) sütunlarını AYRI AYRI doldurur. Anlaşmazlıklar konuşularak
çözülür ve sonuç data/gold/altin_standart.csv olarak kaydedilir.
Etiketlerken yıldız puanına bakmayın (puan sütunu bu yüzden dosyada yok).
"""
import sys
from pathlib import Path

import pandas as pd

KOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK))
from src.config import ayarlari_yukle  # noqa: E402


def main(girdi: str, n: int = 300):
    ayar = ayarlari_yukle()
    df = pd.read_csv(girdi)
    # kategori ve site dengeli örneklem
    parca = df.groupby(["kategori", "site"], group_keys=False).apply(
        lambda g: g.sample(min(len(g), n // max(1, df.groupby(["kategori", "site"]).ngroups)),
                           random_state=ayar["seed"]))
    out = parca[["review_id", "site", "kategori", "yorum_metni"]].copy()
    out["konular"] = ""
    out["duygu"] = ""
    yol = KOK / "data" / "gold" / "etiketlenecek_300.csv"
    yol.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(yol, index=False, encoding="utf-8-sig")
    print(f"{len(out)} yorum yazıldı -> {yol}")


if __name__ == "__main__":
    main(sys.argv[1])
