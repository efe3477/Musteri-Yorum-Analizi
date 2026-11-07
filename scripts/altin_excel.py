"""İnsan etiketlemesi (altın standart) için Excel üretir ve doldurulan dosyaları birleştirir.

Üret:      python scripts/altin_excel.py uret
           -> data/gold/etiketleyici_1.xlsx, data/gold/etiketleyici_2.xlsx (aynı 300 yorum, puansız)
Birleştir: python scripts/altin_excel.py birlestir data/gold/etiketleyici_1.xlsx data/gold/etiketleyici_2.xlsx
           -> data/gold/uyusmazliklar.csv  (iki kişinin farklı etiketlediği yorumlar)
              data/gold/altin_standart.csv (uyuşmazlık sütunu 'karar' ile çözülmüşse onu, yoksa
                                            etiketleyici_1'i esas alır) + iki kişi arası Kappa
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

KOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK))
from src.config import ayarlari_yukle, konu_kodlari  # noqa: E402

N_HUCRE = 75  # kategori x (düşük/yüksek) başına -> 300 yorum


def uret():
    ayar = ayarlari_yukle()
    konular = konu_kodlari(ayar)
    o = pd.read_csv(KOK / "data" / "processed" / "orneklem.csv")
    o["grup"] = o["kategori"] + (o["puan"] <= 2).map({True: "_d", False: "_y"})
    sec = (o.groupby("grup", group_keys=False).apply(lambda g: g.sample(N_HUCRE, random_state=7))
           .sample(frac=1, random_state=7).reset_index(drop=True))
    (KOK / "data" / "gold").mkdir(parents=True, exist_ok=True)
    sec[["kisa_id", "review_id"]].to_csv(KOK / "data" / "gold" / "altin_idler.csv", index=False)

    for n in (1, 2):
        wb = Workbook()
        ws = wb.active
        ws.title = "Etiketleme"
        basliklar = ["sira", "kisa_id", "kategori", "yorum", *konular, "duygu", "not"]
        ws.append(basliklar)
        for i, r in enumerate(sec.itertuples(), 1):
            ws.append([i, r.kisa_id, r.kategori, r.yorum_metni, *[""] * len(konular), "", ""])
        kalin = Font(bold=True)
        for c in ws[1]:
            c.font = kalin
            c.alignment = Alignment(text_rotation=90 if c.value in konular else 0, vertical="bottom")
        ws.column_dimensions["A"].width = 5
        ws.column_dimensions["B"].width = 8
        ws.column_dimensions["C"].width = 11
        ws.column_dimensions["D"].width = 70
        for i in range(len(konular)):
            ws.column_dimensions[ws.cell(1, 5 + i).column_letter].width = 4
        d_kol = ws.cell(1, 5 + len(konular)).column_letter
        ws.column_dimensions[d_kol].width = 10
        ws.row_dimensions[1].height = 130
        for row in ws.iter_rows(min_row=2, min_col=4, max_col=4):
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")
        dv = DataValidation(type="list", formula1='"pozitif,notr,negatif"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"{d_kol}2:{d_kol}{len(sec) + 1}")
        ws.freeze_panes = "E2"
        sari = PatternFill("solid", fgColor="FFF4CC")
        for c in ws[1][4:4 + len(konular)]:
            c.fill = sari

        k = wb.create_sheet("Kurallar")
        k.append(["Konu kodu", "Anlamı"])
        for kod, ac in ayar["konular"].items():
            k.append([kod, ac])
        k.append([])
        for satir in [
            "1. Yorumda ŞİKÂYET edilen her konunun sütununa x koyun. Birden fazla olabilir.",
            "2. Şikâyet yoksa konu sütunlarını boş bırakın. Övgü konu değildir ('kargo hızlıydı' -> boş).",
            "3. 'duygu' sütununu listeden seçin: pozitif / notr / negatif.",
            "4. Puanı tahmin etmeye çalışmayın; yalnızca metne bakın.",
            "5. Diğer etiketleyiciyle konuşmadan, tek başınıza doldurun.",
            "6. Zaman yetmezse SIRAYLA en az ilk 150 satırı bitirin (iki kişi de aynı satırları).",
        ]:
            k.append([satir])
        k.column_dimensions["A"].width = 22
        k.column_dimensions["B"].width = 80
        wb.save(KOK / "data" / "gold" / f"etiketleyici_{n}.xlsx")
    print(f"{len(sec)} yorum -> data/gold/etiketleyici_1.xlsx, etiketleyici_2.xlsx")


def oku(yol) -> pd.DataFrame:
    ayar = ayarlari_yukle()
    konular = konu_kodlari(ayar)
    df = pd.read_excel(yol, sheet_name="Etiketleme", dtype=str).fillna("")
    df = df[df["duygu"].str.strip() != ""]  # doldurulmamış satırlar alınmaz
    isaretli = lambda v: str(v).strip().lower() not in ("", "0", "nan")  # noqa: E731
    df["konular"] = df.apply(lambda r: ";".join(k for k in konular if isaretli(r[k])), axis=1)
    df["duygu"] = df["duygu"].str.strip().str.lower().replace({"nötr": "notr"})
    return df[["kisa_id", "konular", "duygu"]]


def birlestir(y1, y2):
    from sklearn.metrics import cohen_kappa_score
    ayar = ayarlari_yukle()
    konular = konu_kodlari(ayar)
    a, b = oku(y1), oku(y2)
    m = a.merge(b, on="kisa_id", suffixes=("_1", "_2"))
    print(f"İki kişinin de doldurduğu yorum: {len(m)}")
    kappas = []
    for k in konular:
        x = m["konular_1"].str.split(";").map(lambda l: k in l).astype(int)
        y = m["konular_2"].str.split(";").map(lambda l: k in l).astype(int)
        if x.sum() + y.sum() > 0 and len(set(x) | set(y)) > 1:
            kappas.append(cohen_kappa_score(x, y))
    print(f"İnsanlar arası ortalama konu Kappa: {sum(kappas) / len(kappas):.2f}  "
          f"duygu Kappa: {cohen_kappa_score(m['duygu_1'], m['duygu_2']):.2f}")
    fark = m[(m["konular_1"] != m["konular_2"]) | (m["duygu_1"] != m["duygu_2"])].copy()
    fark["karar_konular"] = ""
    fark["karar_duygu"] = ""
    fark.to_csv(KOK / "data" / "gold" / "uyusmazliklar.csv", index=False, encoding="utf-8-sig")
    print(f"Uyuşmazlık: {len(fark)} yorum -> data/gold/uyusmazliklar.csv")
    ids = pd.read_csv(KOK / "data" / "gold" / "altin_idler.csv")
    altin = m.rename(columns={"konular_1": "konular", "duygu_1": "duygu"})[["kisa_id", "konular", "duygu"]]
    karar_yol = KOK / "data" / "gold" / "uyusmazliklar_cozulmus.csv"
    if karar_yol.exists():  # iki kişi uyuşmazlıkları konuşup 'karar_*' sütunlarını doldurduysa
        c = pd.read_csv(karar_yol, dtype=str).fillna("")
        c = c[c["karar_duygu"] != ""].set_index("kisa_id")
        altin = altin.set_index("kisa_id")
        altin.loc[c.index, "konular"] = c["karar_konular"]
        altin.loc[c.index, "duygu"] = c["karar_duygu"]
        altin = altin.reset_index()
        print(f"Çözülmüş uyuşmazlık uygulandı: {len(c)}")
    altin = altin.merge(ids, on="kisa_id")[["review_id", "kisa_id", "konular", "duygu"]]
    altin.to_csv(KOK / "data" / "gold" / "altin_standart.csv", index=False)
    print(f"Altın standart: {len(altin)} yorum -> data/gold/altin_standart.csv")


if __name__ == "__main__":
    if sys.argv[1] == "uret":
        uret()
    else:
        birlestir(sys.argv[2], sys.argv[3])
