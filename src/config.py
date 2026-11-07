"""Ayarları ve proje yollarını yükler."""
from __future__ import annotations

from pathlib import Path

import yaml

PROJE_KOKU = Path(__file__).resolve().parent.parent
VERI_DIZINI = PROJE_KOKU / "data"
CIKTI_DIZINI = PROJE_KOKU / "outputs"

# Veri setinin standart sütunları (Analiz Raporu 2.2)
STANDART_SUTUNLAR = [
    "review_id", "site", "kategori", "yorum_metni", "puan", "tarih", "satici_var_mi",
]


def ayarlari_yukle(yol: str | Path | None = None) -> dict:
    yol = Path(yol) if yol else PROJE_KOKU / "config.yaml"
    with open(yol, encoding="utf-8") as f:
        return yaml.safe_load(f)


def konu_kodlari(ayar: dict) -> list[str]:
    return list(ayar["konular"].keys())
