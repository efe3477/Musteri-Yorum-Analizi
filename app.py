"""Sonuç paneli (UC-06, UC-07, UC-08 · FR-13, FR-14, FR-15).

Çalıştırma:
    streamlit run app.py
    streamlit run app.py -- --cikti outputs/son_calisma
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.config import CIKTI_DIZINI, ayarlari_yukle, konu_kodlari
from src.labeler import konu_listesi

KAPSAM_AD = {"birlesik": "Giyim + elektronik", "giyim": "Giyim", "elektronik": "Elektronik"}


# ------------------------------------------------------------ saf fonksiyonlar (test edilir)

def cikti_dizini_bul(arg: str | None = None) -> Path:
    if arg:
        return Path(arg)
    for aday in ("son_calisma", "ornek_calisma"):
        if (CIKTI_DIZINI / aday / "odds_ratio_tablosu.csv").exists():
            return CIKTI_DIZINI / aday
    return CIKTI_DIZINI / "ornek_calisma"


def sonuclari_yukle(dizin: Path) -> dict:
    return {
        "etiketli": pd.read_csv(dizin / "etiketli_yorumlar.csv"),
        "or": pd.read_csv(dizin / "odds_ratio_tablosu.csv"),
        "metrik": pd.read_csv(dizin / "model_metrikleri.csv"),
    }


def filtrele(df: pd.DataFrame, siteler: list[str] | None, kategori: str) -> pd.DataFrame:
    if siteler:
        df = df[df["site"].isin(siteler)]
    if kategori != "birlesik":
        df = df[df["kategori"] == kategori]
    return df


def konu_paylari(df: pd.DataFrame, konular: list[str]) -> pd.DataFrame:
    listeler = df["konular"].map(konu_listesi)
    pay = {k: listeler.map(lambda l, k=k: k in l).mean() * 100 if len(df) else 0.0 for k in konular}
    return (pd.DataFrame({"konu": list(pay), "pay_yuzde": list(pay.values())})
            .sort_values("pay_yuzde", ascending=False).reset_index(drop=True))


def ornek_yorumlar(df: pd.DataFrame, konu: str, n: int = 10, seed: int = 42) -> pd.DataFrame:
    """FR-14: seçilen konuyu içeren yorumlardan örnek (önce düşük puanlılar)."""
    maske = df["konular"].map(lambda v: konu in konu_listesi(v))
    secili = df[maske].sort_values("puan")
    if len(secili) > n:
        dusuk = secili[secili["puan"] <= 2]
        secili = pd.concat([dusuk.sample(min(len(dusuk), n // 2 + n % 2), random_state=seed),
                            secili[secili["puan"] > 2].sample(
                                min(len(secili[secili["puan"] > 2]), n // 2), random_state=seed)])
        secili = secili.head(n)
    return secili[["site", "kategori", "puan", "yorum_metni"]].reset_index(drop=True)


def etki_siralamasi(or_tablosu: pd.DataFrame, kapsam: str) -> pd.DataFrame:
    """FR-11: kapsamdaki konular OR'ye göre azalan sırada."""
    t = or_tablosu[or_tablosu["kapsam"] == kapsam].sort_values("odds_ratio", ascending=False)
    return t.reset_index(drop=True)


# ------------------------------------------------------------ arayüz

def arayuz():
    import streamlit as st

    st.set_page_config(page_title="Yorum Analizi Paneli", layout="wide")
    arg = None
    if "--cikti" in sys.argv:
        arg = sys.argv[sys.argv.index("--cikti") + 1]
    dizin = cikti_dizini_bul(arg)
    if not (dizin / "odds_ratio_tablosu.csv").exists():
        st.error(f"Sonuç bulunamadı: {dizin}. Önce `python -m src.pipeline` çalıştırın.")
        return

    ayar = ayarlari_yukle()
    konular = konu_kodlari(ayar)
    veri = st.cache_data(sonuclari_yukle)(dizin)
    df_tum, or_tab, metrik = veri["etiketli"], veri["or"], veri["metrik"]

    st.title("Müşteri Yorumu Analizi")
    if dizin.name == "ornek_calisma":
        st.warning("Bu sonuçlar SENTETİK örnek veriyle üretildi; gerçek bulgu değildir.")

    # filtreler tek satırda
    c1, c2 = st.columns([2, 1])
    siteler = sorted(df_tum["site"].dropna().unique())
    sec_site = c1.multiselect("Site", siteler, default=siteler)
    kapsamlar = [k for k in ("birlesik", "giyim", "elektronik") if k in set(or_tab["kapsam"])]
    kapsam = c2.selectbox("Kategori", kapsamlar, format_func=lambda k: KAPSAM_AD.get(k, k))
    df = filtrele(df_tum, sec_site, kapsam)

    if len(df) < ayar["model"]["min_yorum_uyari"]:
        st.warning(f"Seçili filtrede yalnızca {len(df)} yorum var; sonuç güvenilir değil.")

    m = metrik[metrik["kapsam"] == kapsam].iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Yorum (filtre)", f"{len(df):,}".replace(",", "."))
    k2.metric("Düşük puan oranı", f"%{m['dusuk_puan_orani'] * 100:.1f}")
    k3.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    k4.metric("McFadden R²", f"{m['mcfadden_r2']:.3f}")

    sekme1, sekme2, sekme3, sekme4 = st.tabs(
        ["Etki sıralaması", "Şikâyet dağılımı", "Örnek yorumlar", "Dışa aktar"])

    with sekme1:
        sira = etki_siralamasi(or_tab, kapsam)
        st.caption("Atfedilebilir pay: düşük puanlı yorumların ne kadarının bu şikâyete bağlanabileceği. "
                   "Odds ratio (OR) > 1: şikâyet düşük puan olasılığını artırır. "
                   "Model kategori bazındadır; site filtresi bu tabloyu değiştirmez.")
        if "atfedilebilir_pay" in sira:
            for r in sira.sort_values("atfedilebilir_pay", ascending=False).head(3).itertuples():
                st.markdown(f"**{r.konu}** — düşük puanların %{r.atfedilebilir_pay * 100:.0f}'i; {r.yorum}")
        # AC-08 / UC-06 adım 3: her konu için "odds'u X kat daha yüksek" cümlesi tabloda
        sutunlar = [c for c in ["sira", "konu", "n_konu", "atfedilebilir_pay", "odds_ratio", "or_ga_alt",
                                "or_ga_ust", "p_degeri", "yorum"] if c in sira]
        st.dataframe(sira[sutunlar].round(3), hide_index=True, width="stretch")
        ana = dizin / "atfedilebilir_pay.png"
        if ana.exists() and kapsam != "birlesik":
            st.image(str(ana))
        png = dizin / f"forest_{kapsam}.png"
        if png.exists():
            st.image(str(png))

    with sekme2:
        pay = konu_paylari(df, konular)
        st.bar_chart(pay.set_index("konu")["pay_yuzde"], horizontal=True,
                     x_label="Yorumların payı (%)", y_label="")

    with sekme3:
        konu = st.selectbox("Konu", konular)
        st.dataframe(ornek_yorumlar(df, konu), hide_index=True, width="stretch")

    with sekme4:
        st.download_button("OR tablosu (CSV)", or_tab.to_csv(index=False).encode("utf-8-sig"),
                           "odds_ratio_tablosu.csv", "text/csv")
        if png.exists():
            st.download_button("Forest plot (PNG)", png.read_bytes(), png.name, "image/png")
        pdf = dizin / "ozet_rapor.pdf"
        if pdf.exists():
            st.download_button("Özet rapor (PDF)", pdf.read_bytes(), pdf.name, "application/pdf")


if __name__ == "__main__":
    arayuz()
