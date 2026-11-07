# Müşteri Yorumu Analizi — Hangi Şikâyet Puanı Düşürüyor?


> **Proje raporu:** [`PROJE_RAPORU.pdf`](PROJE_RAPORU.pdf) — analiz raporu, müşteri gereksinimleri,
> fonksiyonel ve genel gereksinimler, use case diyagramı, kabul kriterleri ve test case'ler.

| Ad Soyad | Öğrenci No |
| --- | --- |
| Cahit Şenkardeşler | 95230177 |
| Efe Tuna Bebek | 95230053 |
| Miray Tekin | 95230077 |
| Yusuf Can Başaran | 95230067 |
| Fırat Keskin | 95230079 |

Trendyol'daki **giyim** ve **elektronik** ürün yorumlarından büyük dil modeli (LLM)
ile **şikâyet konusu** ve **duygu** çıkarılır; ardından **lojistik regresyon** ile hangi şikâyetin
düşük puan (1–2 yıldız) olasılığını en çok artırdığı ölçülür. Sonuçlar odds ratio (OR) tablosu,
grafikler, PDF rapor ve bir Streamlit panelinde sunulur.

## Sonuç (gerçek veri)

Trendyol'dan 3.000 yorumluk analiz örnekleminde (giyim ve elektronik, her birinden 750 düşük +
750 yüksek puanlı) düşük puanların (1–2 yıldız) en büyük kaynakları:

| Kategori | 1. | 2. | 3. |
| --- | --- | --- | --- |
| Giyim | Ürün kalitesi (%42) | Beden / kalıp (%31) | Açıklamaya uymuyor (%18) |
| Elektronik | Arıza / çalışmıyor (%47) | Ürün kalitesi (%24) | Kargoda hasar (%7) |

Yüzdeler **atfedilebilir paydır**: düşük puanlı yorumların ne kadarının o şikâyete
bağlanabileceğini gösterir. Modellerin ROC-AUC değeri 0,90–0,93; yalnızca kontrol
değişkenleriyle kurulan temel modelde 0,53–0,58. Ayrıntılar `outputs/son_calisma/` klasöründe ve
`PROJE_RAPORU.pdf` dosyasında.

> `data/sample/` ve `outputs/ornek_calisma/` içindekiler **SENTETİKTİR**; yalnızca kodun bilinen
> katsayıları doğru bulduğunu göstermek içindir. Gerçek sonuçlar `outputs/son_calisma/` içindedir.

## Gerçek veri ve sonuçları yeniden üretme

Veri: [Doukan/trendyol-turkish-product-reviews](https://huggingface.co/datasets/Doukan/trendyol-turkish-product-reviews)
(Hugging Face, CC BY-NC 4.0; 582.601 yorum). Ham dosya 131 MB olduğu için depoda yok.

```bash
# Analiz örneklemi ve etiketler depoda; ham veri indirmeden sonuçları üretmek için:
python scripts/etiketleri_birlestir.py
python -m src.pipeline --kaynak csv --girdi data/processed/etiketli_orneklem.csv \
    --etiketleyici hazir --cikti outputs/son_calisma

# Örneklemi ham veriden baştan çekmek için (CSV'yi data/raw/ içine indirin):
python scripts/trendyol_veri_hazirla.py data/raw/trendyol_reviews.csv
```

**Etiketleme hakkında:** Ekipte API anahtarı olmadığı için 3.000 yorum, `src/labeler.py`'deki
istem ve `config.yaml`'daki 12 konu tanımıyla Claude (Anthropic'in LLM'i) tarafından bir sohbet
oturumunda, yıldız puanı gösterilmeden etiketlendi. Ham etiketler
`data/processed/etiketleme/etiket_XX.txt` dosyalarındadır. API anahtarıyla aynı iş
`--etiketleyici llm` ile tekrar yapılabilir.

**Elle doğrulama (ileri çalışma, bu teslimde yapılmadı):** `python scripts/altin_excel.py uret` iki etiketleyici için Excel üretir.
Doldurulan dosyalar `python scripts/altin_excel.py birlestir data/gold/etiketleyici_1.xlsx data/gold/etiketleyici_2.xlsx`
ile birleştirilir. Ardından hat `--altin data/gold/altin_standart.csv` eklenerek yeniden çalıştırılır.

**Yöntem notları:**
- Düşük puan yalnızca %2 olduğundan örneklem sonuca göre tabakalıdır; bu, odds ratio'ları değiştirmez.
- Bazı şikâyetler neredeyse yalnızca düşük puanlı yorumlarda geçtiği için (tam ayrışma) **Firth
  düzeltmeli lojistik regresyon** ve **profil cezalı olabilirlik güven aralıkları** kullanıldı.
- Yorumlar veri setinde en fazla 172 karaktere kısaltılmıştır.

## Hızlı başlangıç

```bash
git clone <depo-adresi> && cd musteri-yorum-analizi
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) API anahtarı olmadan, sentetik veriyle tüm hattı dene (≈5 sn)
python -m src.pipeline --kaynak ornek --etiketleyici kural

# 2) Paneli aç
streamlit run app.py

# 3) Testleri çalıştır
pytest -q
```

## Gerçek veriyle çalıştırma

1. **Veriyi toplayın** ve `data/raw/yorumlar.csv` olarak kaydedin. Sütunlar:
   `site, kategori, yorum_metni, puan, tarih, satici_var_mi` (`review_id` yoksa otomatik üretilir;
   `yorum/rating/review` gibi yaygın sütun adları otomatik eşlenir).
   * Önerilen: açık Türkçe yorum veri setleri (ör. Kaggle) ya da elle kopyalanan yorumlar.
   * `--kaynak url` seçeneği ürün sayfalarındaki schema.org yorum verisini okur, robots.txt'e
     uyar ve istekler arasında 2,5 sn bekler. Sitelerin **kullanım koşullarını okumadan**
     kullanmayın; yorumlar JavaScript ile yükleniyorsa bu yol az veri getirir.
   * Kullanıcı adı / profil bilgisi toplamayın (KVKK). Böyle bir sütun varsa temizlikte silinir.
2. **API anahtarını ayarlayın:** `.env.example` dosyasını `.env` olarak kopyalayıp
   `ANTHROPIC_API_KEY=` satırını doldurun (OpenAI için `config.yaml` → `llm.saglayici: openai`).
3. **Altın standart hazırlayın** (LLM doğrulaması, AC-06):
   ```bash
   python -m src.pipeline --kaynak csv --girdi data/raw/yorumlar.csv --etiketleyici kural   # temizlik için
   python scripts/altin_ornek_sec.py outputs/son_calisma/temiz_yorumlar.csv
   ```
   `data/gold/etiketlenecek_300.csv` dosyasını iki kişi ayrı ayrı etiketler, anlaşmazlıkları
   çözer ve `data/gold/altin_standart.csv` olarak kaydeder.
4. **Asıl çalıştırma:**
   ```bash
   python -m src.pipeline --kaynak csv --girdi data/raw/yorumlar.csv \
       --etiketleyici llm --altin data/gold/altin_standart.csv
   ```
   Etiketleme kesilirse aynı komutu tekrar çalıştırın; `outputs/son_calisma/cache/` sayesinde
   kaldığı yerden devam eder.
5. **Duyarlılık analizi:**
   * Yorum uzunluğu kontrol edilince sonuç değişiyor mu? `python scripts/duyarlilik_analizi.py`
     → `outputs/son_calisma/duyarlilik_uzunluk.csv` (sonuç: sıralama ve anlamlılık değişmedi).
   * 3 yıldızlıları düşük olmayan sınıfa katarak: `... --uc-yildiz sifir --cikti outputs/duyarlilik`
     (bunun için örnekleme 3 yıldızlı yorumlar da alınmalıdır; teslim edilen örneklemde yoktur).

## Çıktılar (`outputs/<çalışma>/`)

| Dosya | İçerik |
| --- | --- |
| `etiketli_yorumlar.csv` | Her yorum için konular, duygu, güven (güven yalnızca `llm` modunda dolar) |
| `dogrulama_konu_bazinda.csv` | Konu bazında precision / recall / F1 / Kappa |
| `odds_ratio_tablosu.csv` | 3 kapsam (birleşik, giyim, elektronik) için β, OR, %95 GA, p, yorum cümlesi |
| `model_metrikleri.csv` | ROC-AUC, temel model AUC, F1, McFadden R², karışıklık matrisi |
| `vif.csv` | Çoklu doğrusal bağlantı kontrolü |
| `forest_*.png`, `kategori_karsilastirma.png`, `atfedilebilir_pay.png`, `konu_dagilimi.png` | Grafikler |
| `duyarlilik_uzunluk.csv` | Yorum uzunluğu kontrollü modelle karşılaştırma (`scripts/duyarlilik_analizi.py`) |
| `ozet_rapor.pdf` | Tek dosyalık sonuç özeti |

## Yöntem özeti

* **Hedef:** `dusuk_puan = 1` (1–2 yıldız), `0` (4–5 yıldız); 3 yıldız ana modelden çıkarılır.
* **Değişkenler:** 12 şikâyet konusu için 0/1 kukla (`config.yaml` → `konular`). Kontroller: site,
  satıcı türü; birleşik modelde kategori.
* **Neden yorum uzunluğu kontrol değil?** Şikâyet yorumu uzatır (aracı değişken); kontrol etmek
  konu etkilerini olduğundan küçük gösterebilir. Duyarlılık analizinde uzunluk eklenince OR'ler
  küçülmedi, büyüdü (şikâyetsiz düşük puanlı yorumlar çok kısa: "berbat", "beğenmedim");
  atfedilebilir paylar, sıralama ve anlamlılık değişmedi.
* **Neden puan LLM'e gönderilmiyor?** Puan hedef değişkendir; LLM görürse etiketleri puana göre yazar
  (hedef sızıntısı) ve regresyon anlamsızlaşır. Bu, bir testle de kontrol edilir.
* **Az görülen konular:** Bir kapsamın eğitim setinde 10'dan az örneği olan konu o kapsamın modeline girmez
  (ör. giyimde `ariza_calismama`). Bu, PDF raporda ve `calisma_ozeti.json`'da listelenir.
* Katsayı/OR için Firth düzeltmeli lojistik regresyon, GA/p için profil cezalı olabilirlik;
  tahmin başarısı için `class_weight="balanced"` `LogisticRegression`. Tohum `seed=42`.
* **Atfedilebilir pay** = (düşük puanlılarda konunun payı) × (1 − 1/OR): düşük puanların ne
  kadarının o şikâyete bağlanabileceği.

## Klasör yapısı

```
├── app.py                  Streamlit paneli (UC-06, 07, 08)
├── config.yaml             Tüm ayarlar (konular, LLM, model)
├── src/
│   ├── scraper.py          UC-01  Yorum toplama / CSV içe aktarma
│   ├── clean.py            UC-02  Temizlik, kişisel veri silme
│   ├── labeler.py          UC-03  LLM ve kural tabanlı etiketleme
│   ├── validate.py         UC-04  Altın standarda karşı doğrulama
│   ├── features.py         FR-09  Hedef ve kukla değişkenler
│   ├── model.py            UC-05  Lojistik regresyon + değerlendirme
│   ├── report.py           UC-08  Grafikler ve PDF
│   ├── pipeline.py         Uçtan uca hat (komut satırı)
│   └── ornek_veri.py       Sentetik veri üretici (bilinen katsayılarla)
├── scripts/                Gerçek veri hazırlığı, etiket birleştirme, elle etiketleme Excel'i, duyarlılık analizi
├── tests/                  pytest — TC-01 … TC-23
├── data/processed/         Gerçek analiz örneklemi (3.000 yorum) ve LLM etiketleri
├── data/gold/              Elle doğrulama için klasör (ileri çalışma)
├── data/sample/            SENTETİK örnek veri
├── outputs/son_calisma/    GERÇEK veri sonuçları (tablolar, grafikler, PDF)
├── outputs/ornek_calisma/  SENTETİK veriyle örnek çıktı
└── PROJE_RAPORU.pdf        Proje raporu (analiz, gereksinimler, use case, kabul kriterleri, test case'ler)
```

## Test case ↔ test eşlemesi

| TC | Test | Dosya |
| --- | --- | --- |
| TC-01 | Standart alanlar dolu / gerçek ağ (isteğe bağlı) | `test_scraper.py` |
| TC-02 | Geçersiz URL programı durdurmaz, robots.txt'e uyulur | `test_scraper.py` |
| TC-03 / 04 / 05 | Tekrar, kısa yorum, kişisel veri | `test_clean.py` |
| TC-06 / 07 / 08 | Tek konu, çok konu, olumlu yorum (kural + isteğe bağlı gerçek LLM) | `test_labeler.py` |
| TC-09 | Bozuk JSON → toplam 3 deneme → "etiketlenemedi", süreç devam | `test_labeler.py` |
| TC-10 | API zaman aşımında bekleyip tekrar dener; durursa kaldığı gruptan devam | `test_labeler.py` |
| TC-11 | Doğrulama metrikleri | `test_model.py` |
| TC-12 / 13 / 14 | OR tablosu, bilinen OR'yi geri bulma (ln 3), sıralama | `test_model.py` |
| TC-15 / 16 | Test performansı raporu, tekrarlanabilirlik | `test_model.py` |
| TC-17 / 18 / 19 | Filtre hızı, örnek yorumlar, dışa aktarılan dosyalar | `test_panel_ve_guvenlik.py` |
| TC-20 | Kodda API anahtarı yok | `test_panel_ve_guvenlik.py` |
| TC-21 | Kategori modelleri (gerçek veri), Firth regresyonu, hazır etiketler | `test_gercek_veri.py` |
| TC-22 | Açık veri seti CSV'sinin içe aktarılması (FR-02) | `test_scraper.py` |
| TC-23 | Hedef değişken ve konu kukla değişkenleri (FR-09) | `test_model.py` |

Gerçek API ve ağ testleri varsayılan olarak atlanır:
`LLM_TESTI=1 pytest tests/test_labeler.py -k gercek` · `AG_TESTI=1 URL_LISTESI=urller.csv pytest -k gercek_ag`
