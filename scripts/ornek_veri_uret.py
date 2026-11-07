"""data/sample/ altına SENTETİK örnek veri ve altın standart yazar.

    python scripts/ornek_veri_uret.py
"""
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK))
from src.ornek_veri import ornek_veri_uret  # noqa: E402

df = ornek_veri_uret(seed=42)
hedef = KOK / "data" / "sample"
hedef.mkdir(parents=True, exist_ok=True)
df.drop(columns=["gercek_konular", "gercek_duygu"]).to_csv(hedef / "ornek_yorumlar.csv", index=False)
(df[["review_id", "gercek_konular", "gercek_duygu"]]
 .rename(columns={"gercek_konular": "konular", "gercek_duygu": "duygu"})
 .sample(300, random_state=42)
 .to_csv(hedef / "ornek_altin_standart.csv", index=False))
print(f"{len(df)} satır -> {hedef}")
