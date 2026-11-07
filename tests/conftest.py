import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ayarlari_yukle, konu_kodlari  # noqa: E402


@pytest.fixture
def ayar():
    return ayarlari_yukle()


@pytest.fixture
def konular(ayar):
    return konu_kodlari(ayar)


@pytest.fixture
def ham_df():
    satirlar = [
        # aynı yorumdan 3 kopya (TC-03)
        *[{"review_id": "tekrar1", "site": "trendyol", "kategori": "giyim",
           "yorum_metni": "Bir beden küçük geldi, iade edeceğim.", "puan": 2,
           "tarih": "2026-05-01", "satici_var_mi": 1, "kullanici_adi": "A*** B***"}] * 3,
        # kısa yorumlar (TC-04)
        {"review_id": "k1", "site": "trendyol", "kategori": "giyim", "yorum_metni": "güzel",
         "puan": 5, "tarih": "2026-05-02", "satici_var_mi": 0, "kullanici_adi": "C*** D***"},
        {"review_id": "k2", "site": "hepsiburada", "kategori": "elektronik", "yorum_metni": "ok",
         "puan": 4, "tarih": "2026-05-03", "satici_var_mi": 0, "kullanici_adi": "E*** F***"},
        # boş ve geçersiz puan
        {"review_id": "b1", "site": "trendyol", "kategori": "giyim", "yorum_metni": "   ",
         "puan": 3, "tarih": "2026-05-04", "satici_var_mi": 0, "kullanici_adi": "G***"},
        {"review_id": "g1", "site": "trendyol", "kategori": "giyim",
         "yorum_metni": "Puanı bozuk bir satır burada.", "puan": 9, "tarih": "2026-05-04",
         "satici_var_mi": 0, "kullanici_adi": "H***"},
        # normal yorum, URL ve emoji ile
        {"review_id": "n1", "site": "hepsiburada", "kategori": "elektronik",
         "yorum_metni": "Ürün çalışmıyor 😡 bakın http://ornek.com/x", "puan": 1,
         "tarih": "2026-05-05", "satici_var_mi": 1, "kullanici_adi": "I*** J***"},
    ]
    return pd.DataFrame(satirlar)
