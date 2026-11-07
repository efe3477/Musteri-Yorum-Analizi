"""SENTETİK örnek veri üretici.

Gerçek veri toplanmadan önce hattı uçtan uca çalıştırmak, testleri koşmak ve
modelin bilinen katsayıları geri bulup bulmadığını kontrol etmek için kullanılır.

!!! Bu verinin ürettiği sonuçlar GERÇEK BULGU DEĞİLDİR. Raporda yalnızca
gerçek (toplanmış) veriyle elde edilen sonuçları kullanın. !!!
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .scraper import review_id_uret

# kategori -> konu -> (görülme oranı, gerçek beta)
GERCEK_PARAMETRELER = {
    "giyim": {
        "sabit": -2.6,
        "kargo_gecikme": (0.10, 0.8), "kargo_hasar": (0.03, 1.1), "paketleme": (0.05, 0.4),
        "urun_kalite": (0.12, 1.7), "beden_uyum": (0.18, 1.0), "aciklama_uyumsuz": (0.08, 1.9),
        "eksik_yanlis_urun": (0.03, 2.0), "satici_iletisim": (0.04, 1.2), "iade_iptal": (0.05, 1.6),
        "fiyat_performans": (0.06, 0.9), "ariza_calismama": (0.005, 1.5), "garanti_servis": (0.005, 1.0),
    },
    "elektronik": {
        "sabit": -2.8,
        "kargo_gecikme": (0.10, 0.7), "kargo_hasar": (0.06, 1.5), "paketleme": (0.06, 0.6),
        "urun_kalite": (0.08, 1.3), "beden_uyum": (0.01, 0.5), "aciklama_uyumsuz": (0.05, 1.4),
        "eksik_yanlis_urun": (0.05, 1.8), "satici_iletisim": (0.04, 1.1), "iade_iptal": (0.05, 1.7),
        "fiyat_performans": (0.06, 0.8), "ariza_calismama": (0.12, 2.4), "garanti_servis": (0.05, 1.5),
    },
}
SITE_ETKISI = {"trendyol": 0.0, "hepsiburada": 0.1}
SATICI_ETKISI = 0.3

# Her konu için ifadeler. Bazıları kural etiketleyicinin anahtar kelimelerini içermez
# (gerçek hayattaki gibi), böylece kural yöntemi kusursuz çıkmaz.
IFADELER = {
    "kargo_gecikme": ["Kargo 12 günde geldi, çok bekledim.", "Teslimat çok gecikti.",
                      "Siparişim hala gelmedi, bir haftadır bekliyorum.",
                      "Söylenen tarihten 5 gün sonra elime ulaştı."],
    "kargo_hasar": ["Kutu ezik geldi.", "Ürün kırık geldi.", "Kargoda hasar görmüş.",
                    "Paketin içinden parçalar dökülüyordu, kargocu fırlatmış herhalde."],
    "paketleme": ["Paketleme çok özensizdi.", "Poşete atıp göndermişler.",
                  "Hiçbir koruma yok, öylece kutuya koymuşlar."],
    "urun_kalite": ["Kumaşı çok kalitesiz.", "Bir yıkamada söküldü.", "Malzemesi kötü, dayanıksız.",
                    "Bir hafta kullandım, hemen yıprandı."],
    "beden_uyum": ["Bir beden küçük geldi.", "Kalıbı çok dar.", "Bol geldi, iade edeceğim.",
                   "Normal giydiğim numarayı aldım ama olmadı."],
    "aciklama_uyumsuz": ["Fotoğraftakiyle alakası yok.", "Renk görseldekinden çok farklı.",
                         "Açıklamada yazan özellikler yok.", "Sitede gördüğümle aynı ürün değil sanki."],
    "eksik_yanlis_urun": ["Kutudan bir parça eksik çıktı.", "Yanlış ürün gönderilmiş.",
                          "Farklı model gönderdiler."],
    "satici_iletisim": ["Satıcı mesajlara cevap vermedi.", "Satıcıya ulaşamadım.",
                        "Satıcı çok ilgisiz.", "Sorumu sordum, kimse ilgilenmedi."],
    "iade_iptal": ["İade ettim, param 3 haftadır yatmadı.", "İptal talebim kabul edilmedi.",
                   "Geri gönderdim ama hala ödeme alamadım."],
    "fiyat_performans": ["Bu fiyata değmez.", "Çok pahalı, paranıza yazık.",
                         "Verdiğim paraya göre beklentimin çok altında."],
    "ariza_calismama": ["Ürün çalışmıyor.", "İki gün sonra bozuldu.", "Şarj olmuyor.",
                        "Kutudan çıktığı gibi açılmıyor.", "Bir süre sonra kendi kendine kapanıyor."],
    "garanti_servis": ["Garanti kapsamında kabul etmediler.", "Yetkili servis ilgilenmedi.",
                       "Teknik destek hattına kimse bakmıyor."],
}
URUN = {"giyim": ["elbise", "tişört", "pantolon", "mont", "ayakkabı", "gömlek"],
        "elektronik": ["kulaklık", "telefon kılıfı", "akıllı saat", "şarj aleti", "hoparlör", "blender"]}
OLUMLU_ORTAK = ["Çok güzel bir {u}, tavsiye ederim.", "{u} harika, kargo da hızlıydı.",
                "Beklediğimden kaliteli bir {u}, teşekkürler.", "Fiyat performans ürünü, memnunum.",
                "Sorunsuz geldi, gayet iyi."]
OLUMLU = {"giyim": OLUMLU_ORTAK + ["Beden tam oldu, çok beğendim.", "Kumaşı yumuşacık, bayıldım."],
          "elektronik": OLUMLU_ORTAK + ["Şarjı uzun gidiyor, çok memnunum.", "Ses kalitesi çok iyi."]}
OLUMSUZ_GENEL = ["Hiç beğenmedim.", "Tavsiye etmem.", "Hayal kırıklığı.", "Maalesef memnun kalmadım."]
YUMUSAK = ["Yine de idare eder.", "Ama genel olarak memnunum.", "Onun dışında güzel."]
KISA = ["güzel", "süper", "iyi", "harika!", "ok"]


def ornek_veri_uret(n_hucre: int = 1500, seed: int = 42, tekrar_orani: float = 0.01,
                    kisa_orani: float = 0.02) -> pd.DataFrame:
    """Her (site, kategori) hücresi için n_hucre yorum üretir.

    Dönen tabloda 'gercek_konular' ve 'gercek_duygu' sütunları vardır; bunlar
    altın standart (insan etiketi) yerine geçer ve modele GİRMEZ.
    """
    rng = np.random.default_rng(seed)
    satirlar = []
    for kategori, par in GERCEK_PARAMETRELER.items():
        konular = [k for k in par if k != "sabit"]
        for site, s_etki in SITE_ETKISI.items():
            for _ in range(n_hucre):
                x = {k: rng.random() < par[k][0] for k in konular}
                satici = int(rng.random() < 0.6)
                logit = par["sabit"] + sum(par[k][1] for k in konular if x[k]) + s_etki + SATICI_ETKISI * satici
                dusuk = rng.random() < 1 / (1 + np.exp(-logit))
                if dusuk:
                    puan = int(rng.choice([1, 2], p=[0.6, 0.4]))
                else:
                    puan = int(rng.choice([3, 4, 5], p=[0.15, 0.25, 0.60]))
                secili = [k for k in konular if x[k]]
                u = rng.choice(URUN[kategori])
                if secili:
                    parca = [rng.choice(IFADELER[k]) for k in secili]
                    if dusuk:
                        parca.append(rng.choice(OLUMSUZ_GENEL))
                    else:
                        parca.append(rng.choice(YUMUSAK))
                    metin = " ".join(parca)
                else:
                    metin = rng.choice(OLUMSUZ_GENEL) + f" {u.capitalize()} beklediğim gibi değil." \
                        if dusuk else rng.choice(OLUMLU[kategori]).format(u=u)
                duygu = "negatif" if puan <= 2 else ("notr" if (puan == 3 or secili) else "pozitif")
                tarih = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(rng.integers(0, 640)))).date().isoformat()
                satirlar.append({
                    "site": site, "kategori": kategori, "yorum_metni": metin, "puan": puan,
                    "tarih": tarih, "satici_var_mi": satici,
                    "kullanici_adi": f"{chr(65 + rng.integers(0, 26))}*** {chr(65 + rng.integers(0, 26))}***",
                    "gercek_konular": ";".join(secili), "gercek_duygu": duygu,
                })
    df = pd.DataFrame(satirlar)
    df["review_id"] = [review_id_uret(r.site, r.yorum_metni, f"{r.tarih}#{i}")
                       for i, r in enumerate(df.itertuples())]

    # gerçekçi kirlilik: tekrarlar ve çok kısa yorumlar
    tekrar = df.sample(frac=tekrar_orani, random_state=seed)
    kisa = df.sample(frac=kisa_orani, random_state=seed + 1).copy()
    kisa["yorum_metni"] = rng.choice(KISA, size=len(kisa))
    kisa["review_id"] = [f"kisa{i:012d}" for i in range(len(kisa))]
    kisa["gercek_konular"] = ""
    df = pd.concat([df, tekrar, kisa], ignore_index=True).sample(frac=1, random_state=seed)
    return df.reset_index(drop=True)
