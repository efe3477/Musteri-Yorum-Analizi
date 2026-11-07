"""UC-08 — Grafikler ve PDF rapor (FR-15)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

from .labeler import konu_listesi  # noqa: E402

# Renkler (doğrulanmış kategorik palet, açık tema)
MAVI, TURUNCU = "#2a78d6", "#eb6834"
METIN, IKINCIL, IZGARA, YUZEY = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
KATEGORI_RENK = {"giyim": MAVI, "elektronik": TURUNCU, "birlesik": "#5b5a56"}
SITE_RENK = {"trendyol": MAVI, "hepsiburada": TURUNCU}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": IZGARA,
    "axes.labelcolor": IKINCIL, "xtick.color": IKINCIL, "ytick.color": METIN,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": METIN,
    "figure.facecolor": YUZEY, "axes.facecolor": YUZEY, "savefig.facecolor": YUZEY,
})

KAPSAM_AD = {"birlesik": "Giyim + elektronik", "giyim": "Giyim", "elektronik": "Elektronik"}
KONU_AD = {
    "kargo_gecikme": "Kargo gecikmesi", "kargo_hasar": "Kargoda hasar", "paketleme": "Paketleme",
    "urun_kalite": "Ürün kalitesi", "beden_uyum": "Beden / kalıp", "aciklama_uyumsuz": "Açıklamaya uymuyor",
    "eksik_yanlis_urun": "Eksik / yanlış ürün", "satici_iletisim": "Satıcı iletişimi",
    "iade_iptal": "İade / iptal", "fiyat_performans": "Fiyatına değmez",
    "ariza_calismama": "Arıza / çalışmıyor", "garanti_servis": "Garanti / servis",
}
OR_MIN, OR_MAX = 0.05, 1e4   # forest plot ekseni; dışına taşan aralık okla gösterilir


def ad(konu: str) -> str:
    return KONU_AD.get(konu, konu)


def sayi(v: float, ondalik: int = 0) -> str:
    """Türkçe sayı biçimi: binlik ayırıcı nokta, ondalık virgül."""
    if not np.isfinite(v):
        return "∞"
    m = f"{v:,.{ondalik}f}"
    return m.replace(",", "§").replace(".", ",").replace("§", ".")


def _log_eksen(ax):
    ax.set_xscale("log")
    ax.set_xlim(OR_MIN, OR_MAX * 1.6)
    ax.xaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100, 1000, 10000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", ".") if v >= 1 else f"{v:g}".replace(".", ",")))
    ax.xaxis.set_minor_locator(NullLocator())


def _sade(ax, eksen="x"):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis=eksen, color=IZGARA, linewidth=0.8)
    ax.set_axisbelow(True)


def _filigran(fig, sentetik: bool):
    if sentetik:
        fig.text(0.99, 0.995, "SENTETİK ÖRNEK VERİ — gerçek bulgu değildir", ha="right",
                 va="top", fontsize=8, color="#e34948")


def _aralik_ciz(ax, y, alt, ust, renk):
    """Güven aralığını çizer; eksen dışına taşan uç ok ile gösterilir."""
    alt_k = np.clip(alt, OR_MIN, OR_MAX)
    ust_k = np.clip(ust, OR_MIN, OR_MAX)
    ax.hlines(y, alt_k, ust_k, color=renk, linewidth=2)
    for yi, u in zip(np.atleast_1d(y), np.atleast_1d(ust)):
        if not np.isfinite(u) or u > OR_MAX:
            ax.plot(OR_MAX, yi, marker=">", color=renk, markersize=6)


def forest_plot(or_tablosu: pd.DataFrame, kapsam: str, yol: Path, sentetik: bool = False) -> Path:
    t = or_tablosu.sort_values("odds_ratio").reset_index(drop=True)
    renk = KATEGORI_RENK.get(kapsam, MAVI)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(t) + 1.9))
    y = np.arange(len(t))
    _aralik_ciz(ax, y, t["or_ga_alt"].values, t["or_ga_ust"].values, renk)
    anl = t["anlamli"].values
    ax.scatter(t["odds_ratio"][anl], y[anl], s=60, color=renk, zorder=3,
               edgecolor=YUZEY, linewidth=2, label="p < 0,05")
    ax.scatter(t["odds_ratio"][~anl], y[~anl], s=60, facecolor=YUZEY, edgecolor=renk,
               linewidth=2, zorder=3, label="anlamlı değil")
    ax.axvline(1, color=IKINCIL, linewidth=1, linestyle="--")
    _log_eksen(ax)
    ax.set_yticks(y, [ad(k) for k in t["konu"]])
    ax.set_xlabel("Odds ratio (log ölçek) ve %95 profil güven aralığı · 1'in sağı düşük puan olasılığını artırır")
    en = t.iloc[-1]
    ax.set_title(f"{KAPSAM_AD.get(kapsam, kapsam)}: en güçlü sinyal '{ad(en.konu)}'", loc="left")
    _sade(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    _filigran(fig, sentetik)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(yol, dpi=160)
    plt.close(fig)
    return yol


def kategori_karsilastirma(sonuclar: dict, yol: Path, sentetik: bool = False) -> Path | None:
    kat = [k for k in ("giyim", "elektronik") if k in sonuclar]
    if len(kat) < 2:
        return None
    tablolar = {k: sonuclar[k]["or_tablosu"].set_index("konu") for k in kat}
    konular = sorted(set().union(*[t.index for t in tablolar.values()]),
                     key=lambda k: max(tablolar[c]["odds_ratio"].get(k, 0) for c in kat))
    fig, ax = plt.subplots(figsize=(8, 0.46 * len(konular) + 1.9))
    y = np.arange(len(konular))
    for i, k in enumerate(kat):
        t = tablolar[k].reindex(konular)
        kay = (0.5 - i) * 0.28
        var = t["odds_ratio"].notna().values
        _aralik_ciz(ax, (y + kay)[var], t["or_ga_alt"].values[var], t["or_ga_ust"].values[var],
                    KATEGORI_RENK[k])
        ax.scatter(t["odds_ratio"], y + kay, s=50, color=KATEGORI_RENK[k], zorder=3,
                   edgecolor=YUZEY, linewidth=2, label=KAPSAM_AD[k])
    ax.axvline(1, color=IKINCIL, linewidth=1, linestyle="--")
    _log_eksen(ax)
    ax.set_yticks(y, [ad(k) for k in konular])
    ax.set_xlabel("Odds ratio (log ölçek), %95 profil GA · eksik nokta: o kategoride yetersiz örnek")
    ax.set_title("Giyim ve elektronikte puanı düşüren şikâyetler farklı", loc="left")
    _sade(ax)
    ax.legend(loc="lower right", frameon=False)
    _filigran(fig, sentetik)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(yol, dpi=160)
    plt.close(fig)
    return yol


def atfedilebilir_pay_grafigi(sonuclar: dict, yol: Path, sentetik: bool = False) -> Path | None:
    """Ana bulgu: düşük puanların ne kadarı hangi şikâyete bağlanabilir (kategori bazında)."""
    kat = [k for k in ("giyim", "elektronik") if k in sonuclar]
    if not kat:
        return None
    pay = pd.DataFrame({k: sonuclar[k]["or_tablosu"].set_index("konu")["atfedilebilir_pay"] for k in kat})
    pay = pay.fillna(0) * 100
    pay = pay[pay.max(axis=1) >= 1]
    pay = pay.loc[pay.max(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(pay) + 1.9))
    y = np.arange(len(pay))
    h = 0.8 / len(kat)
    for i, k in enumerate(kat):
        yy = y + (len(kat) / 2 - i - 0.5) * h
        ax.barh(yy, pay[k], height=h - 0.04, color=KATEGORI_RENK[k], label=KAPSAM_AD[k])
        for yi, v in zip(yy, pay[k]):
            if v >= 1:
                ax.text(v + 0.6, yi, f"%{v:.0f}", va="center", fontsize=8, color=IKINCIL)
    ax.set_yticks(y, [ad(k) for k in pay.index])
    ax.set_xlabel("Düşük puanlı yorumların bu şikâyete atfedilebilen payı (%)")
    ilk = {k: ad(pay[k].idxmax()).lower() for k in kat}
    baslik = " · ".join(f"{KAPSAM_AD[k]}: {ilk[k]}" for k in kat)
    ax.set_title(f"Düşük puanların en büyük kaynağı\n{baslik}", loc="left", fontsize=11)
    ax.set_xlim(0, pay.values.max() * 1.15)
    _sade(ax)
    ax.legend(loc="lower right", frameon=False)
    _filigran(fig, sentetik)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(yol, dpi=160)
    plt.close(fig)
    return yol


def konu_dagilimi(df: pd.DataFrame, konular: list[str], yol: Path, sentetik: bool = False) -> Path:
    """Şikâyet konularının dağılımı. Birden fazla site varsa siteye, yoksa kategoriye göre."""
    grup = "site" if df["site"].nunique() > 1 else "kategori"
    renkler = SITE_RENK if grup == "site" else KATEGORI_RENK
    listeler = df["konular"].map(konu_listesi)
    oran = pd.DataFrame({
        g: [listeler[df[grup] == g].map(lambda l, k=k: k in l).mean() * 100 for k in konular]
        for g in sorted(df[grup].unique())
    }, index=konular)
    oran = oran.loc[oran.max(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(konular) + 1.6))
    y = np.arange(len(oran))
    h = 0.8 / oran.shape[1]
    for i, g in enumerate(oran.columns):
        ax.barh(y + (oran.shape[1] / 2 - i - 0.5) * h, oran[g], height=h - 0.04,
                color=renkler.get(g, MAVI), label=KAPSAM_AD.get(g, g))
    ax.set_yticks(y, [ad(k) for k in oran.index])
    ax.set_xlabel("Bu şikâyeti içeren yorumların payı (%, analiz örneklemi)")
    ax.set_title(f"Şikâyet konularının {'sitelere' if grup == 'site' else 'kategorilere'} göre dağılımı",
                 loc="left")
    _sade(ax)
    ax.legend(loc="lower right", frameon=False)
    _filigran(fig, sentetik)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(yol, dpi=160)
    plt.close(fig)
    return yol


def pdf_rapor(sonuclar: dict, dogrulama: dict | None, temizlik: dict, grafikler: list[Path],
              yol: Path, sentetik: bool = False) -> Path:
    with PdfPages(yol) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        y = 0.95

        def yaz(metin, boy=10, kalin=False, renk=METIN, bosluk=0.022):
            import textwrap
            nonlocal y
            genislik = int(118 * 9 / boy)
            satirlar = textwrap.wrap(metin, genislik, subsequent_indent="    ") or [""]
            for i, parca in enumerate(satirlar):
                fig.text(0.07, y, parca, fontsize=boy, weight="bold" if kalin else "normal",
                         color=renk, va="top")
                y -= bosluk if i == len(satirlar) - 1 else 0.016

        yaz("Müşteri Yorumu Analizi — Sonuç Özeti", 16, True, bosluk=0.035)
        if sentetik:
            yaz("UYARI: Bu rapor SENTETİK örnek veriyle üretildi; sonuçlar gerçek bulgu değildir.",
                9, renk="#e34948", bosluk=0.03)
        yaz("Veri temizliği", 12, True)
        yaz(f"Ham: {temizlik['ham']}  ·  Tekrar: {temizlik['tekrar']}  ·  Kısa: {temizlik['kisa']}"
            f"  ·  Boş: {temizlik['bos']}  ·  Analize giren: {temizlik['temiz']}", bosluk=0.035)
        if dogrulama:
            yaz("Etiket doğrulama (altın standarda karşı)", 12, True)
            yaz(f"n = {dogrulama['n']}  ·  makro F1 = {dogrulama['makro_f1']:.2f}  ·  "
                f"ortalama Kappa = {dogrulama['ortalama_kappa']:.2f}  ·  "
                f"duygu Kappa = {dogrulama['duygu_kappa']:.2f}", bosluk=0.035)
        for kapsam, s in sonuclar.items():
            m = s["metrikler"]
            yaz(f"Model: {KAPSAM_AD.get(kapsam, kapsam)}", 12, True)
            yaz(f"n(eğitim/test) = {m['n_egitim']}/{m['n_test']}  ·  ROC-AUC = {sayi(m['roc_auc'], 3)} "
                f"(temel model {sayi(m['roc_auc_temel_model'], 3)})  ·  F1 = {sayi(m['f1_dusuk_puan'], 3)}  ·  "
                f"McFadden R² = {sayi(m['mcfadden_r2'], 3)}", 9)
            t = s["or_tablosu"]
            yaz("  Düşük puanlara en çok katkı (atfedilebilir pay):", 9, True)
            for r in t.sort_values("atfedilebilir_pay", ascending=False).head(3).itertuples():
                yaz(f"    {ad(r.konu)}: %{r.atfedilebilir_pay * 100:.0f}  (OR {sayi(r.odds_ratio)}; "
                    f"%95 GA {sayi(r.or_ga_alt, 1)}–{sayi(r.or_ga_ust)})", 9)
            if s["cikan_konular"]:
                yaz("  Modele girmeyen: " + ", ".join(f"{ad(k)} ({v})" for k, v in s["cikan_konular"].items()),
                    8, renk=IKINCIL)
            y -= 0.012
        pdf.savefig(fig)
        plt.close(fig)
        for g in grafikler:
            img = plt.imread(g)
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)
    return yol
