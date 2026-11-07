"""UC-05 — Lojistik regresyon (FR-10, FR-11, FR-12, FR-16, NFR-04).

Üç kapsam kurulur:
  * birlesik   : giyim + elektronik, 'kategori' kontrol değişkeni
  * giyim      : yalnızca giyim yorumları
  * elektronik : yalnızca elektronik yorumları

Çıkarım (katsayı, odds ratio, %95 güven aralığı, p-değeri) FIRTH DÜZELTMELİ lojistik
regresyonla yapılır. Neden: gerçek veride bazı şikâyetler (ör. eksik/yanlış ürün) neredeyse
yalnızca düşük puanlı yorumlarda geçer. Bu "tam ayrışma" durumunda klasik en çok olabilirlik
tahmini sonsuza kaçar; Firth (1993) cezası sonlu ve daha az yanlı tahmin verir. Ayrışma yoksa
sonuç klasik lojistik regresyona çok yakındır (test: TC-13).

Güven aralıkları ve p-değerleri PROFİL CEZALI OLABİLİRLİK ile hesaplanır (Heinze & Schemper,
2002); ayrışmaya yakın durumda Wald aralığı güvenilmez. Ayrıca her konu için ATFEDİLEBİLİR PAY
raporlanır: düşük puanlı yorumların ne kadarının o şikâyete bağlanabileceği (OR + yaygınlık).

Tahmin başarısı (ROC-AUC, F1, karışıklık matrisi) scikit-learn ile hesaplanır.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- Firth lojistik regresyon

def _loglik(X, y, beta):
    eta = X @ beta
    return float(np.sum(y * eta - np.logaddexp(0, eta)))


def _cezali_loglik(X, y, beta):
    """Firth cezalı log-olabilirlik: l(β) + ½·log|X'W X|."""
    p = 1 / (1 + np.exp(-(X @ beta)))
    w = p * (1 - p)
    _, logdet = np.linalg.slogdet(X.T @ (X * w[:, None]))
    return _loglik(X, y, beta) + 0.5 * logdet


def firth_logit(X: np.ndarray, y: np.ndarray, maxiter: int = 300, tol: float = 1e-8,
                sabit: dict[int, float] | None = None, beta0: np.ndarray | None = None) -> dict:
    """Firth düzeltmeli lojistik regresyon (Jeffreys önsellikli ceza).

    X sabit terimi içermelidir. 'sabit' = {sütun indeksi: değer} verilirse o katsayılar
    sabitlenir, diğerleri cezalı olabilirliği en büyükleyecek şekilde tahmin edilir
    (profil olabilirlik için). Dönüş: beta, se (model tabanlı), loglik, cezali_loglik, yakinsadi.
    """
    n, k = X.shape
    sabit = sabit or {}
    serbest = np.array([j for j in range(k) if j not in sabit])
    beta = np.zeros(k) if beta0 is None else beta0.astype(float).copy()
    for j, v in sabit.items():
        beta[j] = v
    yakinsadi = False
    for _ in range(maxiter):
        p = 1 / (1 + np.exp(-(X @ beta)))
        w = p * (1 - p)
        inv_tam = np.linalg.pinv(X.T @ (X * w[:, None]))
        # hat matrisinin köşegeni: h_i = w_i * x_i' (X'WX)^-1 x_i  (tam model ile)
        h = w * np.einsum("ij,jk,ik->i", X, inv_tam, X)
        Xs = X[:, serbest]
        U = Xs.T @ (y - p + h * (0.5 - p))
        adim = np.linalg.pinv(Xs.T @ (Xs * w[:, None])) @ U
        m = np.max(np.abs(adim))
        if m > 5:  # adımı sınırla (ayrışmada büyük sıçramaları önler)
            adim *= 5 / m
        beta[serbest] += adim
        if m < tol:
            yakinsadi = True
            break
    p = 1 / (1 + np.exp(-(X @ beta)))
    w = p * (1 - p)
    kov = np.linalg.pinv(X.T @ (X * w[:, None]))
    return {"beta": beta, "se": np.sqrt(np.clip(np.diag(kov), 0, None)),
            "loglik": _loglik(X, y, beta), "cezali_loglik": _cezali_loglik(X, y, beta),
            "yakinsadi": yakinsadi}


def profil_cikarim(X: np.ndarray, y: np.ndarray, tam: dict, j: int, seviye: float = 0.95,
                   sinir: float = 15.0) -> tuple[float, float, float]:
    """j. katsayı için profil cezalı olabilirlik güven aralığı ve p-değeri.

    Firth modelinde önerilen çıkarım budur (Heinze & Schemper, 2002); Wald aralığı
    ayrışmaya yakın durumlarda güvenilmez. Dönüş: (alt, üst, p_değeri). Aralık sınırı
    'sinir' içinde bulunamazsa ±inf döner.
    """
    from scipy.optimize import brentq

    kritik = stats.chi2.ppf(seviye, 1)
    Lmax = tam["cezali_loglik"]
    bhat = tam["beta"][j]

    def fark(b):
        f = firth_logit(X, y, sabit={j: b}, beta0=tam["beta"])
        return 2 * (Lmax - f["cezali_loglik"])

    p = float(stats.chi2.sf(max(fark(0.0), 0.0), 1))

    def kok(yon):
        adim = max(tam["se"][j], 0.25)
        a, b = bhat, bhat + yon * adim
        while fark(b) < kritik:
            a, b = b, b + yon * adim
            adim *= 1.6
            if abs(b - bhat) > sinir:
                return yon * np.inf
        lo, hi = sorted((a, b))
        return brentq(lambda x: fark(x) - kritik, lo, hi, xtol=1e-4)

    return kok(-1), kok(+1), p


# ---------------------------------------------------------------- yardımcılar

def tasarim_matrisi(df: pd.DataFrame, konular: list[str], kontroller: list[str],
                    kategori_ekle: bool) -> pd.DataFrame:
    X = df[[f"konu_{k}" for k in konular]].astype(float)
    for c in kontroller:
        if c == "site":
            X = X.join(pd.get_dummies(df["site"], prefix="site", drop_first=True, dtype=float))
        else:
            X[c] = df[c].astype(float)
    if kategori_ekle:
        X = X.join(pd.get_dummies(df["kategori"], prefix="kategori", drop_first=True, dtype=float))
    # tek değerli sütunlar (ör. tek site) modele bir şey katmaz
    X = X.loc[:, X.nunique() > 1]
    return X


def kullanilabilir_konular(df: pd.DataFrame, konular: list[str], min_sayi: int) -> tuple[list[str], dict]:
    """Bu kapsamda çok az görülen konuları çıkarır (tahmin anlamsız olur)."""
    tut, cikan = [], {}
    for k in konular:
        n = int(df[f"konu_{k}"].sum())
        if n < min_sayi:
            cikan[k] = f"yetersiz örnek (n={n})"
        else:
            tut.append(k)
    return tut, cikan


def vif_hesapla(X: pd.DataFrame) -> pd.DataFrame:
    Xc = sm.add_constant(X, has_constant="add").values
    return pd.DataFrame({
        "degisken": X.columns,
        "vif": [variance_inflation_factor(Xc, i + 1) for i in range(X.shape[1])],
    })


def or_cumlesi(konu: str, or_: float, p: float) -> str:
    if p >= 0.05:
        return f"{konu}: anlamlı bir etki bulunamadı (p = {p:.3f})."
    if or_ >= 1:
        return (f"{konu} şikâyeti içeren yorumun düşük puanlı olma odds'u "
                f"{or_:.1f} kat daha yüksek.")
    return f"{konu} şikâyeti içeren yorumun düşük puanlı olma odds'u %{(1 - or_) * 100:.0f} daha düşük."


# ---------------------------------------------------------------- model

def model_egit(df: pd.DataFrame, ayar: dict, konular: list[str], kapsam: str = "birlesik") -> dict:
    """Bir kapsam için modeli kurar. Dönüş: or_tablosu, metrikler, vif, cikan_konular, ..."""
    m = ayar["model"]
    seed = ayar["seed"]
    if kapsam != "birlesik":
        df = df[df["kategori"] == kapsam]
    if df["dusuk_puan"].nunique() < 2:
        raise ValueError(f"{kapsam}: hedefte iki sınıf yok")

    egitim, test = train_test_split(df, test_size=m["test_orani"], random_state=seed,
                                    stratify=df["dusuk_puan"])
    secili, cikan = kullanilabilir_konular(egitim, konular, m["min_konu_sayisi"])
    X_e = tasarim_matrisi(egitim, secili, m["kontroller"], kategori_ekle=(kapsam == "birlesik"))
    X_t = tasarim_matrisi(test, secili, m["kontroller"], kategori_ekle=(kapsam == "birlesik"))
    X_t = X_t.reindex(columns=X_e.columns, fill_value=0.0)
    y_e, y_t = egitim["dusuk_puan"].values.astype(float), test["dusuk_puan"].values

    vif = vif_hesapla(X_e)
    yuksek_vif = vif.loc[vif["vif"] >= m["vif_esigi"], "degisken"].tolist()
    if yuksek_vif:
        log.warning("%s: VIF >= %s olan değişkenler: %s", kapsam, m["vif_esigi"], yuksek_vif)

    # --- çıkarım (Firth + profil cezalı olabilirlik aralıkları)
    Xc = sm.add_constant(X_e, has_constant="add")
    Xv = Xc.values
    f = firth_logit(Xv, y_e)
    sutun = list(Xc.columns)
    ll0 = _loglik(np.ones((len(y_e), 1)), y_e, np.array([np.log(y_e.mean() / (1 - y_e.mean()))]))

    dusuk = egitim["dusuk_puan"] == 1
    satirlar = []
    for k in secili:
        c = f"konu_{k}"
        j = sutun.index(c)
        b = f["beta"][j]
        alt, ust, p = profil_cikarim(Xv, y_e, f, j)
        or_ = float(np.exp(b))
        pay = float(egitim.loc[dusuk, c].mean())
        satirlar.append({
            "kapsam": kapsam, "konu": k, "n_konu": int(egitim[c].sum()),
            "dusuk_puan_orani": float(egitim.loc[egitim[c] == 1, "dusuk_puan"].mean()),
            "dusuk_puanlilarda_pay": pay,
            "beta": b, "std_hata": f["se"][j], "odds_ratio": or_,
            "or_ga_alt": float(np.exp(alt)), "or_ga_ust": float(np.exp(ust)),
            "p_degeri": p, "anlamli": bool(p < 0.05),
            # Atfedilebilir pay (Miettinen): düşük puanların ne kadarı bu şikâyete bağlanabilir.
            # Düşük puan nadir (%2) olduğundan OR ≈ RR; sonuca göre örneklemede geçerlidir.
            "atfedilebilir_pay": max(0.0, pay * (1 - 1 / or_)) if p < 0.05 else 0.0,
        })
    or_tablosu = (pd.DataFrame(satirlar)
                  .sort_values("odds_ratio", ascending=False).reset_index(drop=True))
    or_tablosu.insert(0, "sira", range(1, len(or_tablosu) + 1))
    or_tablosu["yorum"] = [or_cumlesi(r.konu, r.odds_ratio, r.p_degeri)
                           for r in or_tablosu.itertuples()]
    kontrol = [c for c in sutun if not c.startswith("konu_")]
    kontrol_tablosu = pd.DataFrame({
        "kapsam": kapsam, "degisken": kontrol,
        "beta": [f["beta"][sutun.index(c)] for c in kontrol],
        "std_hata": [f["se"][sutun.index(c)] for c in kontrol],
    })
    kontrol_tablosu["p_degeri_wald"] = 2 * stats.norm.sf(
        np.abs(kontrol_tablosu["beta"] / kontrol_tablosu["std_hata"]))

    # --- tahmin başarısı (scikit-learn, sınıf ağırlıklı)
    sk = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
    sk.fit(X_e, y_e)
    olas = sk.predict_proba(X_t)[:, 1]
    tah = (olas >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_t, tah, labels=[0, 1]).ravel()

    # temel model: yalnızca kontrol değişkenleri
    kontrol_sut = [c for c in X_e.columns if not c.startswith("konu_")]
    if kontrol_sut:
        sk0 = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
        sk0.fit(X_e[kontrol_sut], y_e)
        auc0 = roc_auc_score(y_t, sk0.predict_proba(X_t[kontrol_sut])[:, 1])
    else:
        auc0 = 0.5

    metrikler = {
        "kapsam": kapsam, "n_egitim": len(egitim), "n_test": len(test),
        "dusuk_puan_orani": float(df["dusuk_puan"].mean()),
        "roc_auc": float(roc_auc_score(y_t, olas)),
        "roc_auc_temel_model": float(auc0),
        "f1_dusuk_puan": float(f1_score(y_t, tah)),
        "mcfadden_r2": float(1 - f["loglik"] / ll0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "yakinsadi": bool(f["yakinsadi"]),
    }
    metrikler["kabul_AC09"] = metrikler["roc_auc"] >= 0.80
    return {"or_tablosu": or_tablosu, "kontrol_tablosu": kontrol_tablosu, "metrikler": metrikler,
            "vif": vif, "yuksek_vif": yuksek_vif, "cikan_konular": cikan}


def tum_kapsamlar(df: pd.DataFrame, ayar: dict, konular: list[str]) -> dict[str, dict]:
    sonuclar = {}
    for kapsam in ["birlesik", *ayar["kategoriler"]]:
        if kapsam != "birlesik" and (df["kategori"] == kapsam).sum() == 0:
            log.warning("%s kategorisinde veri yok, atlandı", kapsam)
            continue
        sonuclar[kapsam] = model_egit(df, ayar, konular, kapsam)
    return sonuclar
