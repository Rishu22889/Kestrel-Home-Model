"""What the final model gets wrong in the strict walk-forward (June 2026, the
only post-May month where it had any post-May labels to learn from), plus the
pre-May months for comparison.

Run after src.backtest:  python -m src.error_analysis
"""
import pandas as pd

from .data import ROOT
from .model import FINAL_MODEL

COLS = ["claim_amount_inr", "amount_to_price", "customer_prior_claims", "new_partner",
        "partner_age_days", "is_small", "partner_prior_fraud", "partner_claims_30d"]


def main():
    s = pd.read_pickle(ROOT / "data" / "processed" / "walkforward_scores.pkl")
    f = pd.read_pickle(ROOT / "data" / "processed" / "features.pkl")
    d = f.loc[s.index].assign(score=s[FINAL_MODEL])
    d["month"] = d.submitted_at.dt.to_period("M")
    d["in_queue"] = d.groupby("month").score.rank(ascending=False, method="first") <= 40
    d["fraud"] = d.is_fraud.fillna(0).astype(float) == 1
    out = []
    for name, part in [("post-May (May+Jun)", d[d.post_policy == 1]),
                       ("pre-May (Oct-Apr)", d[d.post_policy == 0])]:
        missed = part[part.fraud & ~part.in_queue]
        false_pos = part[~part.fraud & part.in_queue]
        hits = part[part.fraud & part.in_queue]
        out += [f"\n===== {name} =====",
                f"queued {int(part.in_queue.sum())}, hits {len(hits)}, false alarms {len(false_pos)}, "
                f"missed frauds {len(missed)} (Rs {missed.claim_amount_inr.sum():,.0f})",
                "-- mean profile: hits / missed frauds / false alarms",
                pd.DataFrame({"hits": hits[COLS].mean(), "missed": missed[COLS].mean(),
                              "false_alarms": false_pos[COLS].mean()}).round(2).to_string(),
                "-- missed frauds by outlet (top 8):",
                missed.partner_id.value_counts().head(8).to_string(),
                "-- false alarms by outlet (top 8):",
                false_pos.partner_id.value_counts().head(8).to_string()]
    text = "\n".join(out)
    (ROOT / "reports" / "error_analysis.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
