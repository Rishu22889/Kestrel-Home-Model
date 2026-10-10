"""Step 1: clean, build features, print the data findings that drive modelling.

Run:  python -m src.step1_report
Writes data/processed/features.pkl (git-ignored) and reports/step1_findings.txt
"""
import io
from contextlib import redirect_stdout

import pandas as pd

from .data import ROOT, load_claims, load_raw
from .features import build_features


def rate(df, by):
    g = df.groupby(by, observed=True).is_fraud.agg(fraud_rate="mean", claims="size", frauds="sum")
    g["fraud_rate"] = (g.fraud_rate * 100).round(2)
    return g


def main():
    raw_train, raw_test, partners, _ = load_raw()
    df = build_features(load_claims())
    known = df[df.label_status == "known"].copy()
    known["is_fraud"] = known.is_fraud.astype(float)

    print("== Rows ==")
    print(f"train raw {len(raw_train)}, unique claims {raw_train.claim_id.nunique()}, "
          f"duplicate rows dropped {len(raw_train) - raw_train.claim_id.nunique()}")
    print(f"test raw {len(raw_test)}, unique {raw_test.claim_id.nunique()}")
    print(df.groupby(["split", "label_status"]).size().to_string())
    print(f"\nknown labels: {len(known)}, frauds {int(known.is_fraud.sum())}, "
          f"base rate {known.is_fraud.mean():.4f}")
    print(f"'flag nothing' accuracy = {1 - known.is_fraud.mean():.4f}  <- why accuracy is not the KPI")
    print(f"known fraud rupees {known.loc[known.is_fraud == 1, 'claim_amount_inr'].sum():,.0f}")

    print("\n== Labels by source (legacy zeros may hide undecided cases) ==")
    t = df[df.split == "train"].groupby("source").agg(
        claims=("claim_id", "size"),
        undecided=("label_status", lambda s: (s == "undecided").sum()),
        frauds=("is_fraud", "sum"))
    print(t.to_string())

    print("\n== Policy regime x claim size ==")
    print(rate(known, ["post_policy", "is_small"]).to_string())
    print("\ninspection share by regime:", df.groupby("post_policy").inspected.mean().round(3).to_dict())
    print("test share post-policy:", df[df.split == "test"].post_policy.mean(),
          " small:", round(df[df.split == "test"].is_small.mean(), 3))

    print("\n== Monthly ==")
    print(rate(known, "month").to_string())

    print("\n== New partners (age < 1 year at claim) ==")
    print(rate(known, "new_partner").to_string())
    by_p = known[known.new_partner == 1].groupby("partner_id").is_fraud.agg(["sum", "size"])
    print(f"new-partner outlets with claims: {len(by_p)}, with >=1 fraud: {(by_p['sum'] > 0).sum()}")
    top = known.groupby("partner_id").is_fraud.sum().sort_values(ascending=False)
    print(f"share of all fraud from top 10 outlets: {top.head(10).sum() / top.sum():.2%}")

    print("\n== Univariate signals (known labels) ==")
    known["amount_to_price_b"] = pd.cut(known.amount_to_price, [0, .1, .3, .6, 1.01])
    known["prior_claims_b"] = known.customer_prior_claims.clip(upper=3)
    known["partner_rate_b"] = pd.qcut(known.partner_fraud_rate_smoothed, 5, duplicates="drop")
    known["partner_30d_b"] = pd.qcut(known.partner_claims_30d, 4, duplicates="drop")
    known["serial_prior_b"] = known.serial_prior_claims.clip(upper=2)
    for col in ["amount_to_price_b", "prior_claims_b", "partner_rate_b", "partner_30d_b",
                "partner_type", "family", "photo", "has_note", "serial_prior_b",
                "serial_lowercase", "serial_hyphen", "serial_space", "n_submissions",
                "long_description"]:
        print(f"\n-- {col}")
        print(rate(known, col).to_string())

    print("\n== Inspector note is only present when inspected (availability check) ==")
    print(pd.crosstab(df.inspected, df.has_note).to_string())

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out / "features.pkl")
    print(f"\nsaved {len(df)} rows x {df.shape[1]} cols -> {out / 'features.pkl'}")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    text = buf.getvalue()
    print(text)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "step1_findings.txt").write_text(text)
