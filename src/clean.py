"""UC-02 — Veriyi temizle (FR-03, FR-04, NFR-06)."""
from __future__ import annotations

import re

import pandas as pd

from .config import STANDART_SUTUNLAR
from .scraper import review_id_uret

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF‍️]+"
)
_BOSLUK = re.compile(r"\s+")


def metni_temizle(metin) -> str:
    """URL ve emojileri siler, boşlukları sadeleştirir. Türkçe karakterler korunur."""
    if metin is None or (isinstance(metin, float) and pd.isna(metin)):
        return ""
    m = _URL.sub(" ", str(metin))
    m = _EMOJI.sub(" ", m)
    return _BOSLUK.sub(" ", m).strip()


def temizle(df: pd.DataFrame, ayar: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Ham veriyi temizler.

    Dönüş: (temiz, kisa_yorumlar, rapor)
      - temiz: ana analize giren yorumlar
      - kisa_yorumlar: min uzunluktan kısa olanlar (ayrı tutulur, silinmez)
      - rapor: her adımda kaç satır çıktığı
    """
    t = ayar["temizlik"]
    rapor = {"ham": len(df)}
    df = df.copy()

    # FR-04 / NFR-06: kişisel veri sütunlarını at
    pii = [c for c in df.columns if c.lower() in set(t["kisisel_veri_sutunlari"])]
    df = df.drop(columns=pii)
    rapor["silinen_kisisel_sutunlar"] = pii

    df["yorum_metni"] = df["yorum_metni"].map(metni_temizle)

    # boş yorumlar
    bos = df["yorum_metni"].str.len() == 0
    rapor["bos"] = int(bos.sum())
    df = df[~bos]

    # geçersiz puan
    df["puan"] = pd.to_numeric(df["puan"], errors="coerce")
    gecersiz = ~df["puan"].between(1, 5)
    rapor["gecersiz_puan"] = int(gecersiz.sum())
    df = df[~gecersiz].copy()
    df["puan"] = df["puan"].round().astype(int)

    # kimliği olmayanlara hash kimlik
    eksik = df["review_id"].isna() | (df["review_id"].astype(str).str.len() == 0)
    df.loc[eksik, "review_id"] = [
        review_id_uret(str(r.site), r.yorum_metni, str(r.tarih)) for r in df.loc[eksik].itertuples()
    ]

    # tekrarlar: aynı kimlik veya aynı (site, metin, puan, tarih)
    once = len(df)
    df = df.drop_duplicates(subset="review_id")
    df = df.drop_duplicates(subset=["site", "yorum_metni", "puan", "tarih"])
    rapor["tekrar"] = once - len(df)

    # kısa yorumlar ayrılır
    kisa_maske = df["yorum_metni"].str.len() < t["min_yorum_uzunlugu"]
    kisa = df[kisa_maske].copy()
    df = df[~kisa_maske].copy()
    rapor["kisa"] = len(kisa)
    rapor["temiz"] = len(df)

    # standart sütunlar önce; ek sütunlar (ör. urun_id, konular) korunur
    sutunlar = [c for c in STANDART_SUTUNLAR if c in df.columns]
    sutunlar += [c for c in df.columns if c not in sutunlar]
    return df[sutunlar].reset_index(drop=True), kisa[sutunlar].reset_index(drop=True), rapor
