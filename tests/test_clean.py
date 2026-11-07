"""TC-03, TC-04, TC-05 — Temizlik."""
from src import clean


def test_TC03_tekrar_ayiklama(ham_df, ayar):
    temiz, _, rapor = clean.temizle(ham_df, ayar)
    assert (temiz["review_id"] == "tekrar1").sum() == 1
    assert temiz["review_id"].is_unique
    assert rapor["tekrar"] == 2


def test_TC04_kisa_yorum_filtresi(ham_df, ayar):
    temiz, kisa, _ = clean.temizle(ham_df, ayar)
    assert set(kisa["yorum_metni"]) == {"güzel", "ok"}
    assert (temiz["yorum_metni"].str.len() >= ayar["temizlik"]["min_yorum_uzunlugu"]).all()


def test_TC05_kisisel_veri_yok(ham_df, ayar):
    temiz, kisa, rapor = clean.temizle(ham_df, ayar)
    for d in (temiz, kisa):
        assert "kullanici_adi" not in d.columns
    assert rapor["silinen_kisisel_sutunlar"] == ["kullanici_adi"]


def test_bos_ve_gecersiz_puan_atilir(ham_df, ayar):
    temiz, _, rapor = clean.temizle(ham_df, ayar)
    assert rapor["bos"] == 1 and rapor["gecersiz_puan"] == 1
    assert temiz["puan"].between(1, 5).all()


def test_url_ve_emoji_temizlenir_turkce_korunur():
    assert clean.metni_temizle("Ürün çalışmıyor 😡 bakın http://ornek.com/x") == "Ürün çalışmıyor bakın"
    assert clean.metni_temizle("ĞÜŞİÖÇ ğüşıöç") == "ĞÜŞİÖÇ ğüşıöç"
