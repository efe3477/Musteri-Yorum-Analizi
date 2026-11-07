"""TC-17 … TC-20 — Panel, dışa aktarma, güvenlik, uçtan uca."""
import re
import time
from pathlib import Path

import pandas as pd
import pytest

import app
from src import pipeline
from src.labeler import tr_kucuk

KOK = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def calisma(tmp_path_factory):
    dizin = tmp_path_factory.mktemp("calisma")
    ozet = pipeline.calistir("ornek", None, "kural", None, dizin)
    return dizin, ozet


def test_uctan_uca_ciktilar(calisma):
    dizin, ozet = calisma
    for f in ["temiz_yorumlar.csv", "etiketli_yorumlar.csv", "odds_ratio_tablosu.csv",
              "model_metrikleri.csv", "forest_birlesik.png", "forest_giyim.png",
              "forest_elektronik.png", "kategori_karsilastirma.png", "ozet_rapor.pdf"]:
        assert (dizin / f).stat().st_size > 0, f
    assert set(ozet["metrikler"]) == {"birlesik", "giyim", "elektronik"}


def test_TC17_filtre_hizi(calisma, konular):
    dizin, _ = calisma
    v = app.sonuclari_yukle(dizin)
    bas = time.perf_counter()
    for site in (["trendyol"], ["hepsiburada"]):
        app.konu_paylari(app.filtrele(v["etiketli"], site, "birlesik"), konular)
    assert time.perf_counter() - bas < 3


def test_TC18_ornek_yorumlar(calisma):
    dizin, _ = calisma
    df = app.sonuclari_yukle(dizin)["etiketli"]
    o = app.ornek_yorumlar(df, "iade_iptal")
    assert len(o) >= 5
    # Türkçe küçük harf: "İptal".lower() -> "i̇ptal" olur ve "iptal" ile eşleşmez
    assert o["yorum_metni"].map(tr_kucuk).str.contains("iade|iptal|ödeme|param").all()


def test_TC19_disa_aktarma_dosyalari_acilir(calisma):
    dizin, _ = calisma
    t = pd.read_csv(dizin / "odds_ratio_tablosu.csv")
    assert len(t) > 0
    assert (dizin / "forest_birlesik.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert (dizin / "ozet_rapor.pdf").read_bytes()[:5] == b"%PDF-"


def test_TC20_kodda_api_anahtari_yok():
    desen = re.compile(r"sk-ant-[A-Za-z0-9_-]{10,}|sk-[A-Za-z0-9]{32,}|"
                       r"(?i:api_key\s*=\s*['\"][A-Za-z0-9_-]{8,}['\"])")
    bulunan = []
    for yol in KOK.rglob("*"):
        if yol.is_file() and yol.suffix in {".py", ".yaml", ".yml", ".md", ".txt", ".json", ".toml", ".example"} \
                and ".venv" not in yol.parts and yol.name != ".env":
            if desen.search(yol.read_text(encoding="utf-8", errors="ignore")):
                bulunan.append(str(yol))
    assert not bulunan, bulunan
    assert ".env" in (KOK / ".gitignore").read_text().splitlines()
