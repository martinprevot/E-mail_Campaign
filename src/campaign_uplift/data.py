"""Loading of the MineThatData (Hillstrom) e-mail experiment.

64,000 customers who purchased in the last 12 months were randomly split
into three equal groups: a Men's merchandise e-mail, a Women's merchandise
e-mail, or no e-mail (control). Outcomes are measured over the following
two weeks: website visit, purchase (conversion) and dollars spent.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "hillstrom.csv"
RAW_URL = (
    "https://raw.githubusercontent.com/bscan/uplift/master/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"

CONTROL, MENS, WOMENS = "control", "mens_email", "womens_email"
ARMS = [CONTROL, MENS, WOMENS]
ARM_MAP = {"No E-Mail": CONTROL, "Mens E-Mail": MENS, "Womens E-Mail": WOMENS}
OUTCOMES = ["visit", "conversion", "spend"]
RAW_FEATURES = ["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]


def download(path: Path = RAW_CSV, url: str = RAW_URL) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"Unexpected checksum for {path}: {digest}")
    return path


def load(path: Path = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(download(path))
    df["arm"] = df["segment"].map(ARM_MAP)
    df["treated"] = (df["arm"] != CONTROL).astype(int)
    # Purchase history type, useful to read heterogeneous effects.
    df["buyer_type"] = (
        df["mens"].map({1: "H", 0: ""}) + df["womens"].map({1: "F", 0: ""})
    ).map({"H": "Homme uniquement", "F": "Femme uniquement", "HF": "Homme + Femme"})
    return df.drop(columns=["segment", "history_segment"])


def feature_matrix(df: pd.DataFrame, drop_first: bool = False) -> pd.DataFrame:
    """Pre-treatment covariates, one-hot encoded (no outcome, no arm).
    Use drop_first=True for linear regressions (avoids collinear dummies)."""
    X = pd.get_dummies(df[RAW_FEATURES], columns=["zip_code", "channel"], dtype=int, drop_first=drop_first)
    return X
