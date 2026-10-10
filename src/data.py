"""Load and clean the Kestrel warranty-claims pack.

Decisions (see README "Decisions log"):
- Resubmitted claims share a claim_id. All duplicate pairs are identical except
  submitted_at (gap 0-5 days), so we keep ONE row per claim: the first
  submission, because that is when a fraud check would have to run.
- is_fraud blank (CRM, still under investigation) -> label_status "undecided",
  excluded from supervised training. Legacy Zoho zeros are kept as 0 but tagged
  via `source` so we can test CRM-only training against all-history training.
- submitted_at is already IST; it is not converted (policy s9 UTC note applies
  only to legacy resolution events, which are not in this export).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT  # raw pack files sit in the repo root locally; they are git-ignored

POLICY_CHANGE = pd.Timestamp("2026-05-01")   # <Rs 2,000 auto-approved from here
CRM_START = pd.Timestamp("2025-10-01")
AUTO_APPROVE_LIMIT = 2000


def normalise_serial(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)


def load_raw():
    train = pd.read_csv(RAW / "train.csv")
    test = pd.read_csv(RAW / "test_unlabelled.csv")
    partners = pd.read_csv(RAW / "partners.csv", parse_dates=["onboarded_date"])
    products = pd.read_csv(RAW / "products.csv")
    return train, test, partners, products


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("submitted_at")
    n_sub = df.groupby("claim_id")["claim_id"].transform("size")
    out = df.drop_duplicates("claim_id", keep="first").copy()
    out["n_submissions"] = n_sub.loc[out.index].values
    return out


def load_claims() -> pd.DataFrame:
    """One row per claim (train + test), joined to partner and product data."""
    train, test, partners, products = load_raw()
    train["split"] = "train"
    test["split"] = "test"
    test["is_fraud"] = float("nan")
    df = pd.concat([train, test], ignore_index=True)
    df["submitted_at"] = pd.to_datetime(df["submitted_at"])
    df = dedupe(df)

    df["label_status"] = "known"
    df.loc[(df.split == "train") & df.is_fraud.isna(), "label_status"] = "undecided"
    df.loc[df.split == "test", "label_status"] = "unlabelled"
    df["is_fraud"] = df["is_fraud"].astype("Float64")

    df = df.merge(partners, on="partner_id", how="left", validate="m:1")
    df = df.merge(products, on="sku", how="left", validate="m:1")
    assert df.onboarded_date.notna().all() and df.list_price_inr.notna().all()

    df["serial_norm"] = normalise_serial(df.product_serial)
    return df.sort_values("submitted_at").reset_index(drop=True)
