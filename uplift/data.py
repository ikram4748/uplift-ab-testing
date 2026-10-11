"""The Hillstrom MineThatData e-mail experiment (2008).

64,000 customers who bought in the last 12 months were randomly split into three equal groups: one got a
men's merchandise e-mail, one a women's merchandise e-mail, one no e-mail. Outcomes were tracked for two weeks:
visited the site, bought, and dollars spent. Amounts are in US dollars, as in the original data.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "hillstrom.csv"
URL = "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"

ARMS = {"No E-Mail": "control", "Mens E-Mail": "mens", "Womens E-Mail": "womens"}
TREATMENTS = ["mens", "womens"]
OUTCOMES = ["visit", "conversion", "spend"]
# Measured before the e-mails were sent, so they cannot be affected by the treatment.
PRE_TREATMENT = ["recency", "history", "bought_mens", "bought_womens", "newbie", "zip_code", "channel"]


def download(path: Path = RAW) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"{path} does not match the published file (sha256 {digest[:12]}...)")
    return path


def load(path: Path = RAW) -> pd.DataFrame:
    df = pd.read_csv(download(path))
    df["arm"] = df["segment"].map(ARMS)
    df["zip_code"] = df["zip_code"].replace({"Surburban": "Suburban"})     # typo in the original file
    df["bought_mens"] = df.pop("mens")       # "mens"/"womens" in the raw file = bought that category last year
    df["bought_womens"] = df.pop("womens")
    return df


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Model inputs: pre-treatment columns only, categories one-hot encoded."""
    x = df[["recency", "history", "bought_mens", "bought_womens", "newbie"]].astype(float)
    cats = pd.get_dummies(df[["zip_code", "channel"]], dtype=float, drop_first=True)   # no collinearity with an intercept
    return pd.concat([x, cats], axis=1)


def split(df: pd.DataFrame, seed: int, test_share: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Train/test indices, stratified by arm so every arm keeps its share in both parts."""
    rng = np.random.default_rng(seed)
    test = np.zeros(len(df), dtype=bool)
    for arm in df["arm"].unique():
        idx = np.flatnonzero(df["arm"].to_numpy() == arm)
        test[rng.choice(idx, int(round(len(idx) * test_share)), replace=False)] = True
    return np.flatnonzero(~test), np.flatnonzero(test)
