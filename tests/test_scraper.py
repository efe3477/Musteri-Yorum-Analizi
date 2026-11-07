"""TC-01, TC-02, TC-22 — Yorum toplama ve CSV içe aktarma."""
import json
import os

import pandas as pd
import pytest

from src import scraper

JSONLD_SAYFA = """<html><head><script type="application/ld+json">{}</script></head></html>""".format(
    json.dumps({"@type": "Product", "name": "Tişört", "review": [
        {"@type": "Review", "author": {"name": "Gizli Kişi"}, "reviewBody": "Kumaşı çok ince.",
         "reviewRating": {"ratingValue": "2"}, "datePublished": "2026-04-01"},
        {"@type": "Review", "reviewBody": "Harika ürün, teşekkürler.",
         "reviewRating": {"ratingValue": 5}, "datePublished": "2026-04-02"},
    ]}, ensure_ascii=False))


class SahteIstemci:
    """Ağa çıkmadan istemci davranışını taklit eder."""

    def __init__(self, sayfalar):
        self.sayfalar = sayfalar
        self.cagrilar = []

    def izinli_mi(self, url):
        return "yasak" not in url

    def getir(self, url):
        self.cagrilar.append(url)
        if url not in self.sayfalar:
            raise ConnectionError("adres çözümlenemedi")
        return self.sayfalar[url]


def test_jsonld_yorumlari_yazar_bilgisi_almaz():
    y = scraper.jsonld_yorumlari(JSONLD_SAYFA)
    assert len(y) == 2
    assert y[0] == {"yorum_metni": "Kumaşı çok ince.", "puan": 2, "tarih": "2026-04-01"}
    assert all("author" not in r for r in y)


def test_TC01_standart_alanlar_dolu(ayar):
    ist = SahteIstemci({"https://www.trendyol.com/u1": JSONLD_SAYFA})
    df, hatalar = scraper.urlden_topla(["https://www.trendyol.com/u1"], "giyim", ayar, ist)
    assert not hatalar
    assert len(df) == 2
    for s in ["yorum_metni", "puan", "tarih", "site", "kategori", "review_id"]:
        assert df[s].notna().all(), s
    assert set(df["site"]) == {"trendyol"}


def test_TC02_gecersiz_url_programi_durdurmaz(ayar):
    ist = SahteIstemci({"https://www.hepsiburada.com/u2": JSONLD_SAYFA})
    urller = ["https://ornek/xyz", "https://www.hepsiburada.com/yasak", "https://www.hepsiburada.com/u2"]
    df, hatalar = scraper.urlden_topla(urller, "elektronik", ayar, ist)
    assert len(hatalar) == 2                      # bozuk URL + robots.txt yasağı loglandı
    assert len(df) == 2                           # geçerli URL yine işlendi
    assert "https://www.hepsiburada.com/yasak" not in ist.cagrilar   # izinsiz adrese istek atılmadı


def test_TC22_csv_ice_aktar_sutun_esleme(tmp_path):
    # FR-02 / AC-16: açık veri seti CSV'si standart sütunlara çevrilir, her satıra kimlik atanır
    yol = tmp_path / "kaggle.csv"
    pd.DataFrame({"Review": ["Çok güzel bir ürün."], "Rating": [5]}).to_csv(yol, index=False)
    df = scraper.csv_ice_aktar(yol, site="trendyol", kategori="giyim")
    assert df.loc[0, "yorum_metni"] == "Çok güzel bir ürün."
    assert df.loc[0, "puan"] == 5
    assert isinstance(df.loc[0, "review_id"], str) and len(df.loc[0, "review_id"]) == 16


@pytest.mark.skipif(os.environ.get("AG_TESTI") != "1",
                    reason="Gerçek ağ testi: AG_TESTI=1 ile ve izinli bir URL listesiyle çalıştırın")
def test_TC01_gercek_ag(ayar):
    urller = pd.read_csv(os.environ["URL_LISTESI"])["url"].tolist()[:20]
    df, _ = scraper.urlden_topla(urller, "giyim", ayar)
    assert len(df) > 0
