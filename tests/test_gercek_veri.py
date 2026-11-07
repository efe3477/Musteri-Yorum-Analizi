"""Gerçek veri hattı: Firth regresyonu, hazır etiketler, kategori modelleri (TC-21)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src import labeler, model

KOK = Path(__file__).resolve().parent.parent


def test_firth_ayrisma_yokken_klasik_regresyona_yakin():
    rng = np.random.default_rng(1)
    n = 5000
    x = rng.random((n, 2)) < 0.3
    logit = -1 + 1.2 * x[:, 0] + 0.4 * x[:, 1]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    X = sm.add_constant(x.astype(float))
    mle = sm.Logit(y, X).fit(disp=0).params
    f = model.firth_logit(X, y)
    assert f["yakinsadi"]
    assert np.abs(f["beta"] - mle).max() < 0.02


def test_firth_tam_ayrismada_sonlu_tahmin():
    # konu=1 olan tüm yorumlar düşük puanlı: klasik MLE sonsuza kaçar
    rng = np.random.default_rng(2)
    n = 400
    konu = (rng.random(n) < 0.2).astype(float)
    y = np.where(konu == 1, 1.0, (rng.random(n) < 0.4).astype(float))
    X = sm.add_constant(konu)
    f = model.firth_logit(X, y)
    assert f["yakinsadi"] and np.isfinite(f["beta"]).all()
    alt, ust, p = model.profil_cikarim(X, y, f, 1)
    assert alt < f["beta"][1] < ust          # aralık tahmini içerir
    assert p < 0.001                         # etki anlamlı
    assert alt > 0                           # OR alt sınırı 1'in üstünde


def test_profil_araligi_wald_ile_uyumlu_buyuk_ornekte():
    rng = np.random.default_rng(3)
    n = 8000
    x = (rng.random(n) < 0.4).astype(float)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-0.5 + 0.8 * x)))).astype(float)
    X = sm.add_constant(x)
    f = model.firth_logit(X, y)
    alt, ust, _ = model.profil_cikarim(X, y, f, 1)
    wald = f["beta"][1] + np.array([-1.96, 1.96]) * f["se"][1]
    assert np.allclose([alt, ust], wald, atol=0.02)


def test_hazir_etiketler_dogrulama(konular):
    df = pd.DataFrame({"review_id": ["a", "b"], "yorum_metni": ["x" * 20] * 2,
                       "konular": ["urun_kalite;beden_uyum", np.nan], "duygu": ["negatif", "pozitif"]})
    out = labeler.HazirEtiketler(konular).etiketle(df)
    assert out.loc[1, "konular"] == "" and not out["etiketlenemedi"].any()
    with pytest.raises(ValueError):
        labeler.HazirEtiketler(konular).etiketle(df.assign(konular=["uydurma", ""]))
    with pytest.raises(ValueError):
        labeler.HazirEtiketler(konular).etiketle(df.assign(duygu=["kizgin", "pozitif"]))


def test_etiket_parcasi_bicim_hatasi_yakalanir(tmp_path):
    sys.path.insert(0, str(KOK / "scripts"))
    import etiketleri_birlestir as eb
    (tmp_path / "etiket_01.txt").write_text("y0001 UK,BU -\ny0002 . +\ny0003 XX -\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bilinmeyen kod"):
        eb.oku(tmp_path)
    (tmp_path / "etiket_01.txt").write_text("y0001 UK,BU -\ny0002 . +\n", encoding="utf-8")
    e = eb.oku(tmp_path)
    assert e.loc[0, "konular"] == "urun_kalite;beden_uyum" and e.loc[1, "duygu"] == "pozitif"


GERCEK = KOK / "outputs" / "son_calisma"


@pytest.mark.skipif(not (GERCEK / "odds_ratio_tablosu.csv").exists(),
                    reason="Gerçek veri çıktısı yok (önce pipeline'ı --etiketleyici hazir ile çalıştırın)")
def test_TC21_kategori_modelleri_gercek_veri():
    o = pd.read_csv(GERCEK / "odds_ratio_tablosu.csv")
    assert set(o["kapsam"]) == {"birlesik", "giyim", "elektronik"}
    assert (GERCEK / "kategori_karsilastirma.png").stat().st_size > 0
    ozet = json.loads((GERCEK / "calisma_ozeti.json").read_text(encoding="utf-8"))
    assert "ariza_calismama" in ozet["cikan_konular"]["giyim"]   # giyimde yetersiz örnek
    assert all(m["yakinsadi"] for m in ozet["metrikler"].values())
    for k in o["kapsam"].unique():
        assert o.loc[o["kapsam"] == k, "odds_ratio"].is_monotonic_decreasing
