"""LLM etiket parçalarını (etiket_XX.txt) örneklemle birleştirir ve doğrular.

    python scripts/etiketleri_birlestir.py

Girdi : data/processed/orneklem.csv, data/processed/etiketleme/etiket_XX.txt
Çıktı : data/processed/etiketli_orneklem.csv  (pipeline'a --etiketleyici hazir ile verilir)

Etiket satırı biçimi:  <kisa_id> <konu kısaltmaları, virgülle | '.'> <duygu: + 0 ->
  ör.  y0004 UK,FP -     y0002 . +

Etiketler bu projede Claude (LLM) tarafından, src/labeler.py'deki SISTEM_ISTEMI ve
config.yaml'daki 12 konu tanımıyla, yıldız puanı GÖSTERİLMEDEN üretilmiştir.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

KOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK))
from src.config import ayarlari_yukle, konu_kodlari  # noqa: E402

KISALTMA = {
    "KG": "kargo_gecikme", "KH": "kargo_hasar", "PK": "paketleme", "UK": "urun_kalite",
    "BU": "beden_uyum", "AU": "aciklama_uyumsuz", "EY": "eksik_yanlis_urun",
    "SI": "satici_iletisim", "II": "iade_iptal", "FP": "fiyat_performans",
    "AR": "ariza_calismama", "GS": "garanti_servis",
}
DUYGU = {"+": "pozitif", "0": "notr", "-": "negatif"}


def oku(dizin: Path) -> pd.DataFrame:
    satirlar, hatalar = [], []
    for yol in sorted(dizin.glob("etiket_*.txt")):
        for no, satir in enumerate(yol.read_text(encoding="utf-8").splitlines(), 1):
            if not satir.strip():
                continue
            parca = satir.split()
            if len(parca) != 3:
                hatalar.append(f"{yol.name}:{no} biçim hatalı: {satir!r}")
                continue
            kid, kon, duy = parca
            kodlar = [] if kon == "." else kon.split(",")
            bilinmeyen = [k for k in kodlar if k not in KISALTMA]
            if bilinmeyen or duy not in DUYGU:
                hatalar.append(f"{yol.name}:{no} bilinmeyen kod: {satir!r}")
                continue
            satirlar.append({"kisa_id": kid,
                             "konular": ";".join(dict.fromkeys(KISALTMA[k] for k in kodlar)),
                             "duygu": DUYGU[duy]})
    if hatalar:
        raise ValueError("\n".join(hatalar))
    return pd.DataFrame(satirlar)


def main():
    ayar = ayarlari_yukle()
    assert set(KISALTMA.values()) == set(konu_kodlari(ayar)), "kısaltmalar config ile uyuşmuyor"
    o = pd.read_csv(KOK / "data" / "processed" / "orneklem.csv")
    e = oku(KOK / "data" / "processed" / "etiketleme")
    if e["kisa_id"].duplicated().any():
        raise ValueError(f"tekrarlanan id: {e.loc[e['kisa_id'].duplicated(), 'kisa_id'].tolist()}")
    eksik = sorted(set(o["kisa_id"]) - set(e["kisa_id"]))
    fazla = sorted(set(e["kisa_id"]) - set(o["kisa_id"]))
    if eksik or fazla:
        raise ValueError(f"eksik: {eksik[:10]} ... ({len(eksik)}), fazla: {fazla[:10]}")
    b = o.merge(e, on="kisa_id", how="left")
    b["guven"] = pd.NA            # sohbet içi etiketlemede güven skoru üretilmedi
    b["etiketlenemedi"] = False
    yol = KOK / "data" / "processed" / "etiketli_orneklem.csv"
    b.to_csv(yol, index=False)
    print(f"{len(b)} yorum etiketlendi -> {yol}")
    print("Şikâyet içeren yorum oranı:", round((b["konular"] != "").mean(), 3))
    print(b["duygu"].value_counts().to_dict())


if __name__ == "__main__":
    main()
