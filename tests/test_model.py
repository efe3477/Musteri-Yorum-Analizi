"""TC-11 … TC-16, TC-23 — Doğrulama, hedef ve kukla değişkenler, model, değerlendirme, tekrarlanabilirlik."""
import numpy as np
import pandas as pd
import pytest

from src import features, model, validate
from src.ornek_veri import ornek_veri_uret


def test_TC11_dogrulama_metrikleri(konular):
    altin = pd.DataFrame({"review_id": ["a", "b", "c", "d"],
                          "konular": ["kargo_gecikme", "", "beden_uyum;iade_iptal", "kargo_gecikme"],
                          "duygu": ["negatif", "pozitif", "negatif", "negatif"]})
    tahmin = altin.copy()
    tahmin.loc[3, "konular"] = ""                    # bir kaçırma
    tablo, ozet = validate.dogrula(altin, tahmin, konular)
    kg = tablo.set_index("konu").loc["kargo_gecikme"]
    assert kg["precision"] == 1.0 and kg["recall"] == 0.5
    assert pytest.approx(kg["f1"], 1e-6) == 2 / 3
    assert ozet["n"] == 4 and ozet["duygu_kappa"] == 1.0
    assert {"makro_f1", "ortalama_kappa", "kabul_AC06"} <= set(ozet)


def _sentetik(n=30000, beta=np.log(3), seed=7):
    """Bilinen katsayılı veri: konu_kargo_gecikme için beta = ln(3)."""
    rng = np.random.default_rng(seed)
    x1 = rng.random(n) < 0.3
    x2 = rng.random(n) < 0.2
    logit = -1.5 + beta * x1 + 0.5 * x2
    y = rng.random(n) < 1 / (1 + np.exp(-logit))
    konu = np.where(x1 & x2, "kargo_gecikme;paketleme",
                    np.where(x1, "kargo_gecikme", np.where(x2, "paketleme", "")))
    return pd.DataFrame({"review_id": [f"s{i}" for i in range(n)], "site": rng.choice(["trendyol", "hepsiburada"], n),
                         "kategori": "giyim", "yorum_metni": "yorum metni burada", "puan": np.where(y, 1, 5),
                         "tarih": "2026-01-01", "satici_var_mi": 0, "konular": konu})


@pytest.fixture
def ozellik(ayar, konular):
    df = ornek_veri_uret(n_hucre=800, seed=3, tekrar_orani=0, kisa_orani=0)
    df["konular"] = df["gercek_konular"]
    return features.ozellikleri_hazirla(df, ayar, konular)


def test_TC23_hedef_ve_kukla_degiskenler(ayar, konular):
    # FR-09 / AC-17: 1-2 yıldız -> 1, 4-5 -> 0, 3 yıldız çıkarılır; her konu için 0/1 sütunu
    df = pd.DataFrame({"puan": [1, 2, 3, 4, 5], "yorum_metni": "x" * 20})
    assert features.hedef_olustur(df, ayar, "cikar")["dusuk_puan"].tolist() == [1, 1, 0, 0]
    assert features.hedef_olustur(df, ayar, "sifir")["dusuk_puan"].tolist() == [1, 1, 0, 0, 0]
    k = features.konu_kuklalari(pd.DataFrame({"konular": ["kargo_gecikme;paketleme", ""]}), konular)
    assert [c for c in k.columns if c.startswith("konu_")] == [f"konu_{x}" for x in konular]
    assert k.loc[0, "konu_kargo_gecikme"] == 1 and k.loc[0, "konu_paketleme"] == 1
    assert k.loc[1, [f"konu_{x}" for x in konular]].sum() == 0


def test_TC12_or_tablosu_sutunlari(ozellik, ayar, konular):
    s = model.model_egit(ozellik, ayar, konular, "birlesik")
    t = s["or_tablosu"]
    for c in ["konu", "beta", "odds_ratio", "or_ga_alt", "or_ga_ust", "p_degeri", "yorum"]:
        assert c in t.columns and t[c].notna().all(), c
    assert (t["or_ga_alt"] <= t["odds_ratio"]).all() and (t["odds_ratio"] <= t["or_ga_ust"]).all()
    assert np.allclose(t["odds_ratio"], np.exp(t["beta"]))


def test_TC13_bilinen_or_geri_bulunur(ayar, konular):
    f = features.ozellikleri_hazirla(_sentetik(), ayar, konular)
    s = model.model_egit(f, ayar, konular, "giyim")
    or_ = s["or_tablosu"].set_index("konu").loc["kargo_gecikme", "odds_ratio"]
    assert 2.7 <= or_ <= 3.3, or_


def test_TC14_or_azalan_sirada(ozellik, ayar, konular):
    for kapsam in ["birlesik", "giyim", "elektronik"]:
        t = model.model_egit(ozellik, ayar, konular, kapsam)["or_tablosu"]
        assert t["odds_ratio"].is_monotonic_decreasing
        assert t["sira"].tolist() == list(range(1, len(t) + 1))
        ilk = t.iloc[0]
        if ilk.p_degeri < 0.05:
            assert "kat daha yüksek" in ilk.yorum


def test_TC15_test_performansi_raporlanir(ozellik, ayar, konular):
    m = model.model_egit(ozellik, ayar, konular, "elektronik")["metrikler"]
    for k in ["roc_auc", "f1_dusuk_puan", "mcfadden_r2", "tn", "fp", "fn", "tp", "kabul_AC09"]:
        assert k in m
    assert 0.5 < m["roc_auc"] <= 1
    assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == m["n_test"]


def test_kategoriye_ozgu_az_konu_cikarilir(ozellik, ayar, konular):
    s = model.model_egit(ozellik, ayar, konular, "giyim")
    assert "ariza_calismama" in s["cikan_konular"]     # giyimde neredeyse hiç görülmez


def test_TC16_tekrarlanabilirlik(ozellik, ayar, konular):
    a = model.model_egit(ozellik, ayar, konular, "birlesik")["or_tablosu"]
    b = model.model_egit(ozellik, ayar, konular, "birlesik")["or_tablosu"]
    assert np.abs(a["beta"].values - b["beta"].values).max() < 1e-4
