"""UC-01 — Yorumları topla (FR-01, FR-02, NFR-07).

İki yol vardır:

1. ``csv_ice_aktar``: açık bir veri setini (ör. Kaggle'daki Türkçe yorum setleri) ya da
   elle toplanan bir CSV'yi standart sütunlara çevirir. **Önerilen yol budur.**
2. ``urlden_topla``: ürün sayfalarındaki schema.org ``Review`` (JSON-LD) verisini okur.
   robots.txt kurallarına uyar ve istekler arasında bekler.

ÖNEMLİ: Trendyol ve Hepsiburada'nın kullanım koşulları otomatik veri toplamayı
kısıtlayabilir ve yorumları çoğu zaman JavaScript ile yükler. Kazımadan önce sitenin
kullanım koşullarını okuyun; izin yoksa açık veri setini kullanın. Bu modül kişisel
bilgi (kullanıcı adı vb.) toplamaz.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from urllib import robotparser
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import STANDART_SUTUNLAR

log = logging.getLogger(__name__)


def review_id_uret(site: str, metin: str, tarih: str = "") -> str:
    ham = f"{site}|{metin}|{tarih}".encode("utf-8")
    return hashlib.sha1(ham).hexdigest()[:16]


def site_bul(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if "trendyol" in host:
        return "trendyol"
    if "hepsiburada" in host:
        return "hepsiburada"
    return host or "bilinmiyor"


# ---------------------------------------------------------------- CSV içe aktarma

# Sık görülen sütun adlarını standart adlara çevirir.
SUTUN_ESLEME = {
    "yorum": "yorum_metni", "comment": "yorum_metni", "review": "yorum_metni",
    "text": "yorum_metni", "metin": "yorum_metni", "review_text": "yorum_metni",
    "rating": "puan", "star": "puan", "stars": "puan", "yildiz": "puan", "score": "puan",
    "date": "tarih", "created_at": "tarih",
    "category": "kategori", "source": "site", "platform": "site",
}


def csv_ice_aktar(yol, site: str | None = None, kategori: str | None = None) -> pd.DataFrame:
    """FR-02: dış CSV'yi standart şemaya çevirir. Eksik sütunlar boş bırakılır."""
    df = pd.read_csv(yol)
    df = df.rename(columns={c: SUTUN_ESLEME.get(c.lower().strip(), c.lower().strip())
                            for c in df.columns})
    if site is not None:
        df["site"] = site
    if kategori is not None:
        df["kategori"] = kategori
    for s in STANDART_SUTUNLAR:
        if s not in df.columns:
            df[s] = pd.NA
    eksik_id = df["review_id"].isna()
    df.loc[eksik_id, "review_id"] = [
        review_id_uret(str(r.site), str(r.yorum_metni), str(r.tarih))
        for r in df.loc[eksik_id].itertuples()
    ]
    # Kişisel veri sütunları da şimdilik kalır; clean.temizle() siler.
    return df


# ---------------------------------------------------------------- URL'den toplama

class NazikIstemci:
    """robots.txt'e uyan, istekler arasında bekleyen basit HTTP istemcisi."""

    def __init__(self, user_agent: str, bekleme_sn: float = 2.5, zaman_asimi: float = 15):
        self.user_agent = user_agent
        self.bekleme_sn = bekleme_sn
        self.zaman_asimi = zaman_asimi
        self._robots: dict[str, robotparser.RobotFileParser | None] = {}
        self._son_istek = 0.0

    def izinli_mi(self, url: str) -> bool:
        p = urlparse(url)
        kok = f"{p.scheme}://{p.netloc}"
        if kok not in self._robots:
            rp = robotparser.RobotFileParser()
            try:
                r = requests.get(kok + "/robots.txt", timeout=self.zaman_asimi,
                                 headers={"User-Agent": self.user_agent})
                rp.parse(r.text.splitlines() if r.ok else [])
                self._robots[kok] = rp
            except requests.RequestException:
                self._robots[kok] = None  # robots.txt okunamadı -> temkinli: izin yok
        rp = self._robots[kok]
        return bool(rp and rp.can_fetch(self.user_agent, url))

    def getir(self, url: str) -> str:
        bekle = self.bekleme_sn - (time.monotonic() - self._son_istek)
        if bekle > 0:
            time.sleep(bekle)
        self._son_istek = time.monotonic()
        r = requests.get(url, timeout=self.zaman_asimi, headers={"User-Agent": self.user_agent})
        r.raise_for_status()
        return r.text


def jsonld_yorumlari(html: str) -> list[dict]:
    """Sayfadaki schema.org Review kayıtlarını döndürür (yazar bilgisi alınmaz)."""
    soup = BeautifulSoup(html, "html.parser")
    yorumlar = []
    for etiket in soup.find_all("script", type="application/ld+json"):
        try:
            veri = json.loads(etiket.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        adaylar = veri if isinstance(veri, list) else [veri]
        for nesne in adaylar:
            if not isinstance(nesne, dict):
                continue
            for rv in nesne.get("review", []) or []:
                if not isinstance(rv, dict):
                    continue
                metin = rv.get("reviewBody") or rv.get("description")
                puan = (rv.get("reviewRating") or {}).get("ratingValue")
                if metin and puan is not None:
                    yorumlar.append({
                        "yorum_metni": str(metin),
                        "puan": int(float(puan)),
                        "tarih": rv.get("datePublished", ""),
                    })
    return yorumlar


def urlden_topla(urller: list[str], kategori: str, ayar: dict,
                 istemci: NazikIstemci | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """FR-01: URL listesinden yorum toplar.

    Hatalı/erişilemeyen URL'ler loglanır ve atlanır; süreç durmaz (TC-02).
    Dönüş: (yorumlar, hatalar)
    """
    sc = ayar["scraper"]
    istemci = istemci or NazikIstemci(sc["user_agent"], sc["bekleme_sn"], sc["zaman_asimi_sn"])
    satirlar, hatalar = [], []
    for url in urller:
        try:
            if not istemci.izinli_mi(url):
                raise PermissionError("robots.txt bu adrese izin vermiyor")
            html = istemci.getir(url)
            site = site_bul(url)
            for y in jsonld_yorumlari(html):
                y.update(site=site, kategori=kategori, satici_var_mi=pd.NA,
                         review_id=review_id_uret(site, y["yorum_metni"], y["tarih"]))
                satirlar.append(y)
        except Exception as e:  # noqa: BLE001 — bir URL'nin hatası diğerlerini durdurmaz
            log.warning("URL atlandı: %s (%s)", url, e)
            hatalar.append({"url": url, "hata": str(e)})
    df = pd.DataFrame(satirlar, columns=STANDART_SUTUNLAR)
    return df, hatalar
