"""FR-09 — Hedef değişken ve model değişkenlerinin üretimi."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .labeler import konu_listesi


def hedef_olustur(df: pd.DataFrame, ayar: dict, uc_yildiz: str | None = None) -> pd.DataFrame:
    """dusuk_puan: puan <= 2 -> 1, puan >= 4 -> 0. 3 yıldız 'cikar' ise atılır, 'sifir' ise 0."""
    h = ayar["hedef"]
    uc_yildiz = uc_yildiz or h["uc_yildiz"]
    df = df.copy()
    df["dusuk_puan"] = np.where(df["puan"] <= h["dusuk_puan_max"], 1,
                                np.where(df["puan"] >= h["yuksek_puan_min"], 0, -1))
    if uc_yildiz == "cikar":
        df = df[df["dusuk_puan"] >= 0]
    elif uc_yildiz == "sifir":
        df.loc[df["dusuk_puan"] < 0, "dusuk_puan"] = 0
    else:
        raise ValueError("uc_yildiz 'cikar' veya 'sifir' olmalı")
    return df.reset_index(drop=True)


def konu_kuklalari(df: pd.DataFrame, konular: list[str]) -> pd.DataFrame:
    """Her konu için 0/1 sütunu ekler: konu_<kod>."""
    df = df.copy()
    listeler = df["konular"].map(konu_listesi)
    for k in konular:
        df[f"konu_{k}"] = listeler.map(lambda l, k=k: int(k in l))
    return df


def kontrol_degiskenleri(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_uzunluk"] = np.log1p(df["yorum_metni"].str.len())
    df["satici_var_mi"] = pd.to_numeric(df["satici_var_mi"], errors="coerce").fillna(0).astype(int)
    return df


def ozellikleri_hazirla(df: pd.DataFrame, ayar: dict, konular: list[str],
                        uc_yildiz: str | None = None) -> pd.DataFrame:
    df = df[~df.get("etiketlenemedi", False).astype(bool)] if "etiketlenemedi" in df else df
    df = hedef_olustur(df, ayar, uc_yildiz)
    df = konu_kuklalari(df, konular)
    return kontrol_degiskenleri(df)
