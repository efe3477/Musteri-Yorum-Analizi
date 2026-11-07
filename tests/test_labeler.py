"""TC-06 … TC-10 — Etiketleme."""
import json
import os

import pandas as pd
import pytest

from src import labeler


def _df(metinler, kategori="giyim"):
    return pd.DataFrame({"review_id": [f"r{i}" for i in range(len(metinler))],
                         "kategori": kategori, "yorum_metni": metinler})


def _ids(kullanici_istemi):
    veri = json.loads(kullanici_istemi.split("\n", 1)[1])
    return [d["id"] for d in veri]


# ---------------------------------------------------------------- kural tabanlı (çevrimdışı)

def test_TC06_tek_konu(konular):
    e = labeler.KuralEtiketleyici(konular).tek("Kargo 12 günde geldi, çok bekledim")
    assert e["konular"] == ["kargo_gecikme"] and e["duygu"] == "negatif"


def test_TC07_cok_konu(konular):
    e = labeler.KuralEtiketleyici(konular).tek("Kutu ezik geldi, ürün de bir beden küçük")
    assert {"kargo_hasar", "beden_uyum"} <= set(e["konular"]) and e["duygu"] == "negatif"


def test_TC08_olumlu_yorum(konular):
    e = labeler.KuralEtiketleyici(konular).tek("Hızlı kargo, kaliteli ürün, teşekkürler")
    assert e["konular"] == [] and e["duygu"] == "pozitif"


def test_eksiksiz_eksik_urun_sayilmaz(konular):
    e = labeler.KuralEtiketleyici(konular).tek("Ürün eksiksiz ve hızlı geldi, teşekkürler")
    assert e["konular"] == [] and e["duygu"] == "pozitif"


def test_turkce_buyuk_harf(konular):
    e = labeler.KuralEtiketleyici(konular).tek("ÜRÜN ÇALIŞMIYOR, İADE ETTİM")
    assert {"ariza_calismama", "iade_iptal"} <= set(e["konular"])


# ---------------------------------------------------------------- LLM (sahte istemciyle)

def test_istem_puan_icermez(ayar):
    df = _df(["Bir beden küçük geldi."]).assign(puan=1)
    sistem, kullanici = labeler.istem_olustur(ayar["konular"], df)
    assert "puan" not in kullanici.lower()          # hedef sızıntısı yok
    assert "beden_uyum" in sistem


def test_LLM_gecerli_yanit(ayar, tmp_path):
    def cagir(sistem, kullanici):
        return "```json\n" + json.dumps([{"id": i, "konular": ["kargo_gecikme", "uydurma_konu"],
                                           "duygu": "negatif", "guven": 0.9}
                                          for i in _ids(kullanici)]) + "\n```"
    et = labeler.LLMEtiketleyici(ayar, tmp_path / "c.jsonl", cagir=cagir)
    out = et.etiketle(_df(["Kargo geç geldi."] * 3))
    assert (out["konular"] == "kargo_gecikme").all()   # listede olmayan konu atıldı
    assert not out["etiketlenemedi"].any()


def test_TC09_bozuk_json_uc_deneme_sonra_devam(ayar, tmp_path):
    # FR-07 / AC-05: şemaya uymayan yanıtta toplam en fazla 3 deneme (ilk istek dahil)
    cagri = {"n": 0}

    def bozuk(sistem, kullanici):
        cagri["n"] += 1
        return "konular: kargo"                     # JSON değil
    et = labeler.LLMEtiketleyici(ayar, tmp_path / "c.jsonl", cagir=bozuk)
    out = et.etiketle(_df(["Kargo geç geldi, çok bekledim."]))
    assert cagri["n"] == ayar["llm"]["max_deneme"] == 3
    assert out.loc[0, "etiketlenemedi"]              # işaretlendi, süreç durmadı


def test_kismi_yanit_eksikler_yeniden_istenir(ayar, tmp_path):
    gonderilen = []

    def yarim(sistem, kullanici):
        ids = _ids(kullanici)
        gonderilen.append(ids)
        return json.dumps([{"id": ids[0], "konular": [], "duygu": "pozitif", "guven": 0.8}])
    et = labeler.LLMEtiketleyici(ayar, tmp_path / "c.jsonl", cagir=yarim)
    out = et.etiketle(_df(["Güzel bir ürün aldım.", "Harika, teşekkürler."]))
    assert gonderilen[1] == ["r1"]                    # yalnızca eksik olan yeniden gönderildi
    assert not out["etiketlenemedi"].any()


def _iyi_yanit(ids):
    return json.dumps([{"id": i, "konular": ["kargo_gecikme"], "duygu": "negatif", "guven": 1}
                       for i in ids])


def test_TC10_api_zaman_asimi_bekler_ve_kaldigi_yerden_devam(ayar, tmp_path):
    # UC-03 alternatif akış 3a + NFR-03: API zaman aşımı sürerse sistem bekleyip tekrar dener,
    # yine olmazsa durur; yeniden başlatınca biten gruplar tekrar gönderilmez.
    ayar["llm"]["grup_boyutu"] = 10
    ayar["llm"]["api_max_deneme"] = 3
    df = _df([f"Yorum numarası {i}, kargo geç geldi." for i in range(100)])
    onbellek = tmp_path / "c.jsonl"
    ilk, beklemeler = [], []

    def kopan_api(sistem, kullanici):
        ids = _ids(kullanici)
        if len(ilk) >= 30:
            raise TimeoutError("API 30 sn içinde yanıt vermedi")
        ilk.extend(ids)
        return _iyi_yanit(ids)

    et = labeler.LLMEtiketleyici(ayar, onbellek, cagir=kopan_api)
    et.uyu = beklemeler.append                        # gerçekten beklemesin
    with pytest.raises(labeler.APIHatasi):
        et.etiketle(df)
    assert beklemeler == [5, 10]                      # 3 denemede 2 bekleme, süre artar
    kayitlar = [json.loads(s) for s in onbellek.read_text(encoding="utf-8").splitlines() if s.strip()]
    assert len(kayitlar) == 30                        # yalnızca biten 3 grup önbellekte
    assert not any(k["etiketlenemedi"] for k in kayitlar)   # API hatası "etiketlenemedi" yazılmadı

    ikinci = []

    def iyi(sistem, kullanici):
        ids = _ids(kullanici)
        ikinci.extend(ids)
        return _iyi_yanit(ids)

    out = labeler.LLMEtiketleyici(ayar, onbellek, cagir=iyi).etiketle(df)
    assert len(out) == 100 and not out["etiketlenemedi"].any()
    assert set(ilk).isdisjoint(ikinci)                # biten gruplar tekrar gönderilmedi
    assert len(ikinci) == 70


def test_gecici_api_hatasinda_bekleyip_devam_eder(ayar, tmp_path):
    hatalar = {"kalan": 2}
    beklemeler = []

    def limitli(sistem, kullanici):
        if hatalar["kalan"]:
            hatalar["kalan"] -= 1
            raise ConnectionError("429 istek limiti")
        return _iyi_yanit(_ids(kullanici))

    et = labeler.LLMEtiketleyici(ayar, tmp_path / "c.jsonl", cagir=limitli)
    et.uyu = beklemeler.append
    out = et.etiketle(_df(["Kargo geç geldi."] * 3))
    assert beklemeler == [5, 10]
    assert not out["etiketlenemedi"].any()            # API hatası "etiketlenemedi" sayılmadı


@pytest.mark.skipif(os.environ.get("LLM_TESTI") != "1",
                    reason="Gerçek API testi: LLM_TESTI=1 ve .env içinde API anahtarı gerekir")
def test_TC06_08_gercek_llm(ayar, tmp_path):
    df = _df(["Kargo 12 günde geldi, çok bekledim",
              "Kutu ezik geldi, ürün de bir beden küçük",
              "Hızlı kargo, kaliteli ürün, teşekkürler"])
    out = labeler.LLMEtiketleyici(ayar, tmp_path / "c.jsonl").etiketle(df)
    k = out["konular"].map(labeler.konu_listesi)
    assert k[0] == ["kargo_gecikme"] and out.loc[0, "duygu"] == "negatif"
    assert {"kargo_hasar", "beden_uyum"} <= set(k[1])
    assert k[2] == [] and out.loc[2, "duygu"] == "pozitif"
