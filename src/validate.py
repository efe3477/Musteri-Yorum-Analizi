"""UC-04 — Etiketleri doğrula (FR-08, AC-06).

İnsan etiketli altın standart ile etiketleyici çıktısını karşılaştırır:
konu bazında precision / recall / F1 / Cohen's Kappa ve duygu için Kappa.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score

from .labeler import konu_listesi


def _ikili(seri: pd.Series, konu: str) -> np.ndarray:
    return np.array([int(konu in konu_listesi(v)) for v in seri])


def dogrula(altin: pd.DataFrame, tahmin: pd.DataFrame, konular: list[str]) -> tuple[pd.DataFrame, dict]:
    """altin ve tahmin: review_id, konular, duygu sütunları.

    Dönüş: (konu_tablosu, ozet)
    """
    b = altin.merge(tahmin, on="review_id", suffixes=("_altin", "_tahmin"))
    if b.empty:
        raise ValueError("Altın standart ile tahmin arasında ortak review_id yok")
    satirlar = []
    for k in konular:
        y, t = _ikili(b["konular_altin"], k), _ikili(b["konular_tahmin"], k)
        if y.sum() == 0 and t.sum() == 0:
            continue  # bu örneklemde hiç görünmeyen konu
        kappa = cohen_kappa_score(y, t) if len(set(y) | set(t)) > 1 else 1.0
        satirlar.append({
            "konu": k, "destek": int(y.sum()),
            "precision": precision_score(y, t, zero_division=0),
            "recall": recall_score(y, t, zero_division=0),
            "f1": f1_score(y, t, zero_division=0),
            "kappa": kappa,
        })
    tablo = pd.DataFrame(satirlar)
    ozet = {
        "n": len(b),
        "makro_f1": float(tablo["f1"].mean()),
        "ortalama_kappa": float(tablo["kappa"].mean()),
        "duygu_kappa": float(cohen_kappa_score(b["duygu_altin"], b["duygu_tahmin"]))
        if "duygu_altin" in b else float("nan"),
    }
    ozet["kabul_AC06"] = ozet["ortalama_kappa"] >= 0.70 and ozet["makro_f1"] >= 0.75
    return tablo, ozet
