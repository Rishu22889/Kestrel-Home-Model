"""Train the final model on every known label and score the unlabelled claims.

Run:  python -m src.train_final
Writes
  models/final_model.joblib, models/meta.json   (git-ignored: derived from client data)
  predictions.csv                                (claim_id, score) - see README on s10
  reports/test_queue_summary.txt
"""
import json

import joblib
import pandas as pd

from .data import ROOT, load_raw
from .model import CANDIDATES, FINAL_MODEL, fit

CAPACITY = 40


def main():
    df = pd.read_pickle(ROOT / "data" / "processed" / "features.pkl")
    train = df[df.label_status == "known"]          # undecided excluded
    test = df[df.split == "test"]

    model = fit(FINAL_MODEL, train)
    test = test.assign(score=model.predict_proba(test)[:, 1])

    sample = pd.read_csv(ROOT / "sample_submission.csv")
    pred = sample[["claim_id"]].merge(test[["claim_id", "score"]], on="claim_id", how="left")
    assert len(pred) == 2252 and pred.claim_id.is_unique and pred.score.notna().all()
    pred["score"] = pred.score.round(6)
    pred.to_csv(ROOT / "predictions.csv", index=False)

    # Capacity is the hard rule: each month the desk gets that month's 40
    # highest-scored claims. The marker (median 40th score) is only a guide.
    month = test.submitted_at.dt.to_period("M")
    test = test.assign(month=month,
                       month_rank=test.groupby(month).score.rank(ascending=False, method="first"))
    test["in_queue"] = test.month_rank <= CAPACITY
    threshold = float(test[test.month_rank == CAPACITY].score.median())

    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    joblib.dump(model, models / "final_model.joblib")
    _, _, partners, products = load_raw()
    joblib.dump({"partners": partners, "products": products}, models / "lookups.joblib")
    meta = {
        "model": FINAL_MODEL, "post_weight": CANDIDATES[FINAL_MODEL][1],
        "train_rows": int(len(train)), "train_frauds": int(train.is_fraud.sum()),
        "train_until": str(train.submitted_at.max()), "review_threshold": threshold,
        "monthly_capacity": CAPACITY,
        "score_meaning": "Relative fraud-risk score from a logistic regression trained with "
                         "post-May rows weighted x5. Higher = review first. Not calibrated: "
                         "0.76 does not mean a 76% chance of fraud.",
    }
    (models / "meta.json").write_text(json.dumps(meta, indent=1))

    # what next quarter's queue looks like - the single source for memo/README counts
    q = test[test.in_queue]
    known = df[df.label_status == "known"]
    confirmed = sorted(known[(known.is_fraud == 1) & (known.post_policy == 1)].partner_id.unique())
    q_conf = q[q.partner_id.isin(confirmed)]
    rest = q[~q.partner_id.isin(confirmed)]
    new_rest = test[(test.new_partner == 1) & ~test.partner_id.isin(confirmed)]
    lines = [
        f"QUEUE RULE: top {CAPACITY} claims per month by score (hard cap).",
        f"score marker (median 40th score, guide only): {threshold:.4f}",
        f"claims above marker per month: {test[test.score >= threshold].groupby('month').size().to_dict()}",
        f"queue size per month: {q.groupby('month').size().to_dict()}  total {len(q)}",
        f"lowest queued score per month: {q.groupby('month').score.min().round(4).to_dict()}",
        "",
        f"queued claims from new outlets (<1y old at claim): {q.new_partner.sum()} of {len(q)} "
        f"({q.new_partner.mean():.0%}); new-outlet share of all test claims {test.new_partner.mean():.0%}",
        f"new outlets with test claims: {test[test.new_partner == 1].partner_id.nunique()}, "
        f"with >=1 queued claim: {q[q.new_partner == 1].partner_id.nunique()}",
        "",
        "Outlets with >=1 CONFIRMED fraud since 1 May 2026 (training labels): "
        f"{len(confirmed)} -> {', '.join(confirmed)}",
        f"  queued claims from them: {len(q_conf)} of {len(q)} ({len(q_conf) / len(q):.0%}), "
        f"from {q_conf.partner_id.nunique()} of these outlets",
        f"  per outlet: {q_conf.partner_id.value_counts().to_dict()}",
        f"queued claims from all other outlets: {len(rest)} from {rest.partner_id.nunique()} outlets; "
        f"claims-per-outlet distribution {rest.partner_id.value_counts().value_counts().sort_index().to_dict()}",
        f"new outlets WITHOUT confirmed fraud: {new_rest.partner_id.nunique()} outlets, "
        f"{len(new_rest)} test claims, {int(new_rest.index.isin(q.index).sum())} queued "
        f"({new_rest.index.isin(q.index).mean():.0%})",
        "",
        f"queued rupees: {q.claim_amount_inr.sum():,.0f}, median claim {q.claim_amount_inr.median():,.0f}",
        f"score distribution (all test): {test.score.describe().round(4).to_dict()}",
    ]
    text = "\n".join(lines)
    (ROOT / "reports" / "test_queue_summary.txt").write_text(text)
    meta["confirmed_fraud_outlets_post_may"] = confirmed
    (models / "meta.json").write_text(json.dumps(meta, indent=1))
    print(text)
    print(f"\nwrote predictions.csv ({len(pred)} rows) and models/")


if __name__ == "__main__":
    main()
