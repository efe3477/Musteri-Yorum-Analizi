"""UC-03 — LLM ile konu ve duygu çıkar (FR-05, FR-06, FR-07, NFR-01, NFR-03).

İki etiketleyici vardır:

* ``LLMEtiketleyici`` — projenin asıl yöntemi. Yorumları gruplar halinde LLM'e gönderir,
  JSON yanıtı şemaya göre doğrular; şemaya uymayan yanıtta toplam en fazla ``max_deneme``
  deneme yapar (ilk istek dahil), yine olmazsa yorumu "etiketlenemedi" işaretler ve devam eder.
  API hatasında (zaman aşımı, istek limiti) artan sürelerle bekleyip tekrar dener; hata
  sürerse durur. Sonuçlar her gruptan sonra önbelleğe yazılır, yeniden çalıştırınca kaldığı
  gruptan devam eder (NFR-03).
* ``KuralEtiketleyici`` — anahtar kelimeye dayalı temel (baseline) yöntem. API anahtarı
  olmadan tüm hattı çalıştırmak ve LLM'i karşılaştırmak için kullanılır.

Not: LLM'e yıldız puanı GÖNDERİLMEZ. Puan hedef değişkendir; modele sızarsa
(target leakage) etiketler puana göre yazılır ve regresyon sonuçları anlamsızlaşır.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Callable

import pandas as pd

log = logging.getLogger(__name__)

DUYGULAR = ("pozitif", "notr", "negatif")


def tr_kucuk(metin: str) -> str:
    """Türkçe'ye uygun küçük harf (I -> ı, İ -> i)."""
    return str(metin).replace("I", "ı").replace("İ", "i").lower()


# ====================================================================== Kural tabanlı

ANAHTAR_KELIMELER: dict[str, list[str]] = {
    "kargo_gecikme": ["geç geldi", "geç teslim", "günde geldi", "gecikti", "gecikme",
                      "hala gelmedi", "hâlâ gelmedi", "kargo yavaş", "çok bekledim",
                      "teslimat uzun", "kargoya geç"],
    "kargo_hasar": ["ezik geldi", "ezik", "kırık geldi", "hasarlı", "yırtık geldi",
                    "çizik geldi", "parçalanmış", "kargoda hasar", "kırılmış"],
    "paketleme": ["paketleme", "poşete", "koruma yok", "özensiz paket", "kutusuz",
                  "paket kötü", "balonlu naylon yok", "paketi açık"],
    "urun_kalite": ["kalitesiz", "kalite kötü", "kalitesi kötü", "söküldü", "boyası çıktı",
                    "dayanıksız", "yırtıldı", "tüylendi", "malzemesi kötü", "ince kumaş",
                    "kötü işçilik", "dikişleri", "plastik ucuz"],
    "beden_uyum": ["beden", "dar geldi", "bol geldi", "küçük geldi", "büyük geldi",
                   "kalıbı", "ölçü", "boyu kısa"],
    "aciklama_uyumsuz": ["fotoğraftaki", "fotoğraftan farklı", "resimdeki", "görseldeki",
                         "açıklamada", "farklı renk", "alakası yok", "gösterildiği gibi değil",
                         "açıklamayla uyuşmuyor"],
    "eksik_yanlis_urun": ["eksik", "yanlış ürün", "farklı model", "farklı ürün", "parça yok",
                          "yanlış gönder", "yanlış renk gönder"],
    "satici_iletisim": ["satıcı ilgisiz", "satıcıya ulaş", "mesajlara", "cevap vermedi",
                        "dönüş yapmadı", "muhatap", "satıcı kaba", "yanıt vermedi"],
    "iade_iptal": ["iade", "iptal", "para iadesi", "geri ödeme", "param yatmadı",
                   "parası yatmadı"],
    "fiyat_performans": ["değmez", "pahalı", "paranıza yazık", "paraya yazık",
                         "fiyatını hak etmiyor", "fiyatına göre kötü"],
    "ariza_calismama": ["çalışmıyor", "çalışmadı", "arızalı", "bozuk geldi", "açılmıyor",
                        "şarj olmuyor", "bozuldu", "ses gelmiyor", "aşırı ısınıyor",
                        "görüntü gelmiyor"],
    "garanti_servis": ["garanti", "servis", "teknik destek", "yetkili servis"],
}

POZITIF = ["güzel", "harika", "mükemmel", "teşekkür", "memnun", "tavsiye ederim", "süper",
           "kaliteli", "hızlı kargo", "çok iyi", "beğendim", "sorunsuz", "fiyat performans ürünü",
           "eksiksiz"]

# Bir anahtar kelimeyi içeren ama şikâyet olmayan kalıplar (ör. "eksiksiz" -> "eksik" sayılmaz)
OLUMLU_KALIPLAR = ["eksiksiz"]
NEGATIF = ["kötü", "berbat", "rezalet", "pişman", "memnun değil", "beğenmedim",
           "tavsiye etmem", "asla", "hayal kırıklığı", "maalesef", "sakın"]


class KuralEtiketleyici:
    """Anahtar kelime eşleştirmesiyle çok etiketli konu + duygu (temel yöntem)."""

    ad = "kural"

    def __init__(self, konular: list[str]):
        self.konular = konular

    def tek(self, metin: str) -> dict:
        m = tr_kucuk(metin)
        m_konu = m
        for kalip in OLUMLU_KALIPLAR:
            m_konu = m_konu.replace(kalip, " ")
        konular = [k for k in self.konular
                   if any(a in m_konu for a in ANAHTAR_KELIMELER.get(k, []))]
        # "fiyat performans ürünü" olumlu bir kalıptır, 'pahalı' gibi kelime yoksa konu sayma
        pos = sum(p in m for p in POZITIF)
        neg = sum(n in m for n in NEGATIF) + len(konular)
        if "memnun değil" in m:
            pos -= 1
        duygu = "negatif" if neg > pos else ("pozitif" if pos > neg else "notr")
        return {"konular": konular, "duygu": duygu, "guven": 0.5, "etiketlenemedi": False}

    def etiketle(self, df: pd.DataFrame) -> pd.DataFrame:
        sonuc = [self.tek(t) for t in df["yorum_metni"]]
        return _birlestir(df, sonuc)


# ====================================================================== LLM

SISTEM_ISTEMI = """Sen bir e-ticaret müşteri yorumu analistisin. Türkçe ürün yorumlarını \
etiketliyorsun.

Her yorum için:
1. "konular": Yorumda ŞİKÂYET EDİLEN konuları aşağıdaki kapalı listeden seç. Birden fazla \
olabilir. Şikâyet yoksa boş liste [] döndür. Övgüleri konu olarak sayma \
(ör. "kargo hızlıydı" kargo_gecikme DEĞİLDİR).
2. "duygu": Yorumun genel duygusu: "pozitif", "notr" veya "negatif".
3. "guven": Etiketlerine ne kadar güvendiğin, 0 ile 1 arasında bir sayı.

Konu listesi:
{konu_listesi}

YALNIZCA geçerli bir JSON dizisi döndür, başka hiçbir metin yazma. Biçim:
[{{"id": "<yorum id>", "konular": ["kod1", "kod2"], "duygu": "negatif", "guven": 0.9}}]"""


def istem_olustur(konular: dict[str, str], grup: pd.DataFrame) -> tuple[str, str]:
    konu_listesi = "\n".join(f"- {k}: {v}" for k, v in konular.items())
    sistem = SISTEM_ISTEMI.format(konu_listesi=konu_listesi)
    satirlar = [{"id": str(r.review_id), "kategori": str(r.kategori), "yorum": r.yorum_metni}
                for r in grup.itertuples()]
    kullanici = "Aşağıdaki yorumları etiketle:\n" + json.dumps(satirlar, ensure_ascii=False)
    return sistem, kullanici


def yaniti_dogrula(metin: str, beklenen_idler: list[str], izinli_konular: set[str]) -> dict[str, dict]:
    """FR-07: LLM yanıtını şemaya göre doğrular.

    Geçerli kayıtları {id: etiket} olarak döndürür. Hiç geçerli kayıt yoksa ValueError.
    Kod bloğu (```json ... ```) içine sarılmış yanıtlar kabul edilir.
    """
    temiz = re.sub(r"^```(?:json)?\s*|\s*```$", "", metin.strip())
    veri = json.loads(temiz)  # bozuk JSON -> JSONDecodeError (ValueError alt sınıfı)
    if not isinstance(veri, list):
        raise ValueError("yanıt bir JSON dizisi değil")
    gecerli = {}
    for k in veri:
        if not isinstance(k, dict):
            continue
        rid = str(k.get("id", ""))
        konular = k.get("konular")
        duygu = k.get("duygu")
        guven = k.get("guven", 0.5)
        if rid not in beklenen_idler or not isinstance(konular, list):
            continue
        if duygu not in DUYGULAR or not all(isinstance(c, str) for c in konular):
            continue
        try:
            guven = float(guven)
        except (TypeError, ValueError):
            continue
        if not 0 <= guven <= 1:
            continue
        konular = [c for c in konular if c in izinli_konular]
        gecerli[rid] = {"konular": konular, "duygu": duygu, "guven": guven,
                        "etiketlenemedi": False}
    if not gecerli:
        raise ValueError("şemaya uyan kayıt yok")
    return gecerli


def anthropic_istemcisi(model: str, sicaklik: float, max_token: int) -> Callable[[str, str], str]:
    import anthropic  # isteğe bağlı bağımlılık

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def cagir(sistem: str, kullanici: str) -> str:
        # Güncel Anthropic SDK'sı 'temperature' parametresini kabul etmiyor; tutarlılık
        # sabit istem + kapalı konu listesi + JSON şema doğrulamasıyla sağlanır.
        yanit = client.messages.create(
            model=model, max_tokens=max_token, system=sistem,
            messages=[{"role": "user", "content": kullanici}],
        )
        return "".join(b.text for b in yanit.content if getattr(b, "type", "") == "text")

    return cagir


def openai_istemcisi(model: str, sicaklik: float, max_token: int) -> Callable[[str, str], str]:
    import openai  # isteğe bağlı bağımlılık

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def cagir(sistem: str, kullanici: str) -> str:
        yanit = client.chat.completions.create(
            model=model, temperature=sicaklik, max_tokens=max_token,
            messages=[{"role": "system", "content": sistem}, {"role": "user", "content": kullanici}],
        )
        return yanit.choices[0].message.content or ""

    return cagir


class APIHatasi(RuntimeError):
    """LLM API'sine bekleyip tekrar denemeye rağmen ulaşılamadı; etiketleme durdurulur."""


class LLMEtiketleyici:
    ad = "llm"

    def __init__(self, ayar: dict, onbellek_yolu: str | Path,
                 cagir: Callable[[str, str], str] | None = None):
        self.konular: dict[str, str] = ayar["konular"]
        l = ayar["llm"]
        self.grup_boyutu = l["grup_boyutu"]
        self.max_deneme = l["max_deneme"]              # şema hatasında toplam deneme
        self.api_max_deneme = l.get("api_max_deneme", 5)  # API hatasında toplam deneme
        self.api_bekleme_sn = l.get("api_bekleme_sn", 5)  # ilk bekleme; her seferinde 2 katı
        self.uyu = time.sleep                           # testlerde değiştirilebilir
        self.onbellek = Path(onbellek_yolu)
        if cagir is None:
            from dotenv import load_dotenv
            load_dotenv()
            fabrika = {"anthropic": anthropic_istemcisi, "openai": openai_istemcisi}[l["saglayici"]]
            cagir = fabrika(l["model"], l["sicaklik"], l["max_token"])
        self.cagir = cagir

    # ---- önbellek (NFR-03: kaldığı yerden devam)
    def _onbellegi_oku(self) -> dict[str, dict]:
        if not self.onbellek.exists():
            return {}
        kayit = {}
        with open(self.onbellek, encoding="utf-8") as f:
            for satir in f:
                if satir.strip():
                    d = json.loads(satir)
                    kayit[d.pop("review_id")] = d
        return kayit

    def _onbellege_yaz(self, etiketler: dict[str, dict]) -> None:
        self.onbellek.parent.mkdir(parents=True, exist_ok=True)
        with open(self.onbellek, "a", encoding="utf-8") as f:
            for rid, e in etiketler.items():
                f.write(json.dumps({"review_id": rid, **e}, ensure_ascii=False) + "\n")

    # ---- API çağrısı (UC-03 alternatif akış 3a, NFR-03)
    def _api_cagir(self, sistem: str, kullanici: str) -> str:
        """API hatasında (zaman aşımı, istek limiti, bağlantı) bekleyip tekrar dener.

        Bekleme süreleri api_bekleme_sn, 2x, 4x ... Hata sürerse APIHatasi fırlatılır: o ana
        kadar biten gruplar önbellekte olduğu için yeniden çalıştırınca kaldığı gruptan devam
        edilir. API hatası yüzünden hiçbir yorum "etiketlenemedi" diye kaydedilmez.
        """
        for deneme in range(1, self.api_max_deneme + 1):
            try:
                return self.cagir(sistem, kullanici)
            except Exception as e:  # noqa: BLE001 — SDK'ya göre değişen hata türleri
                if deneme == self.api_max_deneme:
                    raise APIHatasi(f"LLM API'sine {deneme} denemede ulaşılamadı: {e}") from e
                bekle = self.api_bekleme_sn * 2 ** (deneme - 1)
                log.warning("API hatası (%s); %.0f sn beklenip tekrar denenecek (%d/%d)",
                            e, bekle, deneme, self.api_max_deneme)
                self.uyu(bekle)
        raise AssertionError("buraya gelinmez")

    # ---- bir grup
    def _grubu_etiketle(self, grup: pd.DataFrame) -> dict[str, dict]:
        kalan = grup.copy()
        sonuc: dict[str, dict] = {}
        for deneme in range(1, self.max_deneme + 1):  # FR-07: toplam en fazla max_deneme
            if kalan.empty:
                break
            sistem, kullanici = istem_olustur(self.konular, kalan)
            idler = [str(i) for i in kalan["review_id"]]
            yanit = self._api_cagir(sistem, kullanici)   # APIHatasi -> süreç durur
            try:
                gecerli = yaniti_dogrula(yanit, idler, set(self.konular))
            except (ValueError, TypeError, AttributeError) as e:  # bozuk JSON / şema dışı
                log.warning("Şemaya uymayan yanıt, deneme %d/%d: %s", deneme, self.max_deneme, e)
                continue
            sonuc.update(gecerli)
            kalan = kalan[~kalan["review_id"].astype(str).isin(gecerli)]
        for rid in kalan["review_id"].astype(str):  # AC-05: süreç durmaz
            sonuc[rid] = {"konular": [], "duygu": "notr", "guven": 0.0, "etiketlenemedi": True}
        return sonuc

    def etiketle(self, df: pd.DataFrame) -> pd.DataFrame:
        hazir = self._onbellegi_oku()
        yapilacak = df[~df["review_id"].astype(str).isin(hazir)]
        log.info("Önbellekte %d, etiketlenecek %d yorum", len(df) - len(yapilacak), len(yapilacak))
        for bas in range(0, len(yapilacak), self.grup_boyutu):
            grup = yapilacak.iloc[bas:bas + self.grup_boyutu]
            etiket = self._grubu_etiketle(grup)
            self._onbellege_yaz(etiket)
            hazir.update(etiket)
        sonuc = [hazir[str(r)] for r in df["review_id"]]
        return _birlestir(df, sonuc)


class HazirEtiketler:
    """Etiketleri önceden üretilmiş veri (ör. scripts/etiketleri_birlestir.py çıktısı).

    'konular' ve 'duygu' sütunları zaten vardır; yalnızca biçim doğrulanır.
    """

    ad = "hazir"

    def __init__(self, konular: list[str]):
        self.konular = set(konular)

    def etiketle(self, df: pd.DataFrame) -> pd.DataFrame:
        eksik = {"konular", "duygu"} - set(df.columns)
        if eksik:
            raise ValueError(f"hazir etiket modu için eksik sütun: {eksik}")
        out = df.copy().reset_index(drop=True)
        out["konular"] = out["konular"].fillna("").astype(str)
        bilinmeyen = {k for v in out["konular"] for k in konu_listesi(v)} - self.konular
        if bilinmeyen:
            raise ValueError(f"tanımsız konu kodu: {bilinmeyen}")
        if not out["duygu"].isin(DUYGULAR).all():
            raise ValueError("duygu yalnızca pozitif / notr / negatif olabilir")
        if "guven" not in out:
            out["guven"] = pd.NA
        out["etiketlenemedi"] = out.get("etiketlenemedi", False)
        out["etiketlenemedi"] = out["etiketlenemedi"].fillna(False).astype(bool)
        return out


def _birlestir(df: pd.DataFrame, sonuc: list[dict]) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["konular"] = [";".join(s["konular"]) for s in sonuc]
    out["duygu"] = [s["duygu"] for s in sonuc]
    out["guven"] = [s["guven"] for s in sonuc]
    out["etiketlenemedi"] = [s["etiketlenemedi"] for s in sonuc]
    return out


def konu_listesi(deger) -> list[str]:
    """'a;b' biçimindeki konu hücresini listeye çevirir."""
    if deger is None or (isinstance(deger, float) and pd.isna(deger)) or deger == "":
        return []
    if isinstance(deger, list):
        return deger
    return [k for k in str(deger).split(";") if k]
