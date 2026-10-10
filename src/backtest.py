"""Strict walk-forward backtest - the primary evidence.

Rules
- Prediction time = submitted_at. Models are refit at the start of every week
  (Mondays 00:00, plus the first day of the backtest).
- A past claim's label may be used only if submitted_at + LABEL_LAG_DAYS is
  before the refit time (proxy: the pack has no investigation-completion date).
  Undecided labels are never used for training.
- Partner-history features already obey the same lag (features.py).
- Each month the desk reviews the 40 highest-scored claims of that month.
  (Picking a month's top 40 needs the whole month; in operation a fixed score
  threshold would be used instead. Documented limitation.)

Money (ops policy s4), reported as separate lines:
  gross_avoided      = claim amounts of known frauds in the queue
  goodwill_known     = Rs 380 x known-genuine claims held
  goodwill_undecided = Rs 380 x undecided claims held (conservative: treated as genuine)
  net                = gross_avoided - goodwill_known - goodwill_undecided
  net_if_contact     = net - Rs 260 x 40; sensitivity only, applies only if every
                       review creates an extra service contact (policy does not say so)

Run:  python -m src.backtest
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .data import POLICY_CHANGE, ROOT
from .features import LABEL_LAG_DAYS
from .model import CANDIDATES, FINAL_MODEL, fit

CAPACITY = 40
GOODWILL = 380
CONTACT = 260
START, END = pd.Timestamp("2025-10-01"), pd.Timestamp("2026-07-01")
RULE = "rule_new_small_post+partner_history"


def load_train():
    df = pd.read_pickle(ROOT / "data" / "processed" / "features.pkl")
    return df[df.split == "train"].copy()


def rule_score(d):
    # transparent operational rule: new outlet + small claim after May first,
    # then the partner's (lagged) fraud record
    return (d.new_small_post + d.partner_fraud_rate_smoothed).values


def walk_forward_scores(df, names):
    """Score every claim in [START, END) with weekly refits under the label lag."""
    in_range = (df.submitted_at >= START) & (df.submitted_at < END)
    out = pd.DataFrame(np.nan, index=df.index[in_range], columns=names)
    refits = sorted({START, *pd.date_range(START, END, freq="W-MON", inclusive="left")})
    for t0, t1 in zip(refits, refits[1:] + [END]):
        idx = df.index[(df.submitted_at >= t0) & (df.submitted_at < t1)]
        if len(idx) == 0:
            continue
        avail = (df.label_status == "known") & (
            df.submitted_at + pd.Timedelta(days=LABEL_LAG_DAYS) < t0)
        tr = df[avail]
        for n in names:
            out.loc[idx, n] = fit(n, tr).predict_proba(df.loc[idx])[:, 1]
    return out


def queue(month_df, score):
    d = month_df.assign(_s=score)
    q = d.nlargest(CAPACITY, "_s")
    fraud = q.is_fraud.fillna(0).astype(float) == 1
    undec = q.label_status == "undecided"
    known = d[d.label_status == "known"]
    y = known.is_fraud.astype(int)
    gross = q.loc[fraud, "claim_amount_inr"].sum()
    gw_known = GOODWILL * int((~fraud & ~undec).sum())
    gw_undec = GOODWILL * int(undec.sum())
    return {
        "frauds_in_month": int(y.sum()),
        "fraud_rs_in_month": known.loc[y == 1, "claim_amount_inr"].sum(),
        "caught": int(fraud.sum()),
        "undecided_in_queue": int(undec.sum()),
        "precision_at_40": fraud.sum() / CAPACITY,
        "recall_at_40": fraud.sum() / y.sum() if y.sum() else np.nan,
        "gross_avoided": gross,
        "goodwill_known": gw_known,
        "goodwill_undecided": gw_undec,
        "net": gross - gw_known - gw_undec,
        "net_if_contact": gross - gw_known - gw_undec - CONTACT * CAPACITY,
        "pr_auc": average_precision_score(y, d.loc[known.index, "_s"]) if y.sum() else np.nan,
        # accuracy if exactly the 40 queued claims are called "fraud"
        "accuracy": float((known.index.isin(q.index) == (y == 1)).mean()),
    }


def monthly_table(df, scores):
    rng = np.random.default_rng(0)
    rows = []
    sub = df.loc[scores.index]
    for month, mdf in sub.groupby(sub.submitted_at.dt.to_period("M")):
        base = {"month": str(month),
                "regime": "post_policy" if month.start_time >= POLICY_CHANGE else "pre_policy"}
        known = mdf[mdf.label_status == "known"]
        # flags nothing: empty queue, catches nothing, holds no one
        rows.append({**base, "model": "always_not_fraud",
                     "frauds_in_month": int((known.is_fraud == 1).sum()),
                     "fraud_rs_in_month": known.loc[known.is_fraud == 1, "claim_amount_inr"].sum(),
                     "caught": 0, "undecided_in_queue": 0, "precision_at_40": 0.0,
                     "recall_at_40": 0.0, "gross_avoided": 0.0, "goodwill_known": 0.0,
                     "goodwill_undecided": 0.0, "net": 0.0, "net_if_contact": 0.0,
                     "pr_auc": np.nan, "accuracy": float((known.is_fraud == 0).mean())})
        rand = pd.DataFrame([queue(mdf, rng.random(len(mdf))) for _ in range(200)]).mean()
        rows.append({**base, "model": "random_40", **rand.to_dict()})
        rows.append({**base, "model": RULE, **queue(mdf, rule_score(mdf))})
        for n in scores.columns:
            rows.append({**base, "model": n, **queue(mdf, scores.loc[mdf.index, n].values)})
    return pd.DataFrame(rows)


def summary(monthly):
    s = monthly.groupby(["regime", "model"]).agg(
        months=("month", "nunique"), frauds=("frauds_in_month", "sum"), caught=("caught", "sum"),
        precision_at_40=("precision_at_40", "mean"), gross_avoided=("gross_avoided", "sum"),
        goodwill=("goodwill_known", "sum"), goodwill_undecided=("goodwill_undecided", "sum"),
        net=("net", "sum"), net_per_month=("net", "mean"),
        net_if_contact_per_month=("net_if_contact", "mean"), pr_auc=("pr_auc", "mean"),
        accuracy=("accuracy", "mean"))
    s["recall_at_40"] = s.caught / s.frauds
    s["net_per_review"] = s.net_per_month / CAPACITY
    return s.round(3).reset_index().sort_values(["regime", "net_per_month"], ascending=[True, False])


def relaxed_lag_diagnostic(df):
    """Secondary and weaker: June scored by a model that sees ALL labels before
    1 June (lag relaxed to 0). Shows what the model can do once May outcomes are
    in, which the strict backtest cannot show yet."""
    tr = df[(df.label_status == "known") & (df.submitted_at < "2026-06-01")]
    ev = df[df.submitted_at.dt.to_period("M") == pd.Period("2026-06")]
    rows = [{"model": RULE, **queue(ev, rule_score(ev))}]
    for n in CANDIDATES:
        rows.append({"model": n, **queue(ev, fit(n, tr).predict_proba(ev)[:, 1])})
    return pd.DataFrame(rows).round(3)


def bootstrap_ci(df, scores, regime, name=FINAL_MODEL, n=1000):
    """Resample claims within each month; 90% interval for monthly net and precision."""
    rng = np.random.default_rng(1)
    sub = df.loc[scores.index].assign(_s=scores[name])
    post = sub.submitted_at >= POLICY_CHANGE
    sub = sub[post] if regime == "post_policy" else sub[~post]
    months = [m for _, m in sub.groupby(sub.submitted_at.dt.to_period("M"))]
    nets, precs = [], []
    for _ in range(n):
        r = [queue(b, b._s.values) for b in (m.iloc[rng.integers(0, len(m), len(m))].reset_index(drop=True) for m in months)]
        nets.append(np.mean([x["net"] for x in r]))
        precs.append(np.mean([x["precision_at_40"] for x in r]))
    pct = lambda a: [round(float(v), 3) for v in np.percentile(a, [5, 50, 95])]
    return {"model": name, "regime": regime, "net_per_month_p5_p50_p95": pct(nets),
            "precision_at_40_p5_p50_p95": pct(precs)}


def main():
    df = load_train()
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    for old in ["backtest_monthly.csv", "backtest_summary.csv", "post_policy_folds.csv"]:
        (out / old).unlink(missing_ok=True)  # superseded experiments

    scores = walk_forward_scores(df, list(CANDIDATES))
    monthly = monthly_table(df, scores)
    summ = summary(monthly)
    diag = relaxed_lag_diagnostic(df)
    cis = [bootstrap_ci(df, scores, r) for r in ("pre_policy", "post_policy")]

    scores.join(df[["claim_id", "submitted_at", "claim_amount_inr", "is_fraud", "label_status"]]) \
        .to_pickle(ROOT / "data" / "processed" / "walkforward_scores.pkl")  # for error analysis
    monthly.round(3).to_csv(out / "walkforward_monthly.csv", index=False)
    summ.to_csv(out / "walkforward_summary.csv", index=False)
    diag.to_csv(out / "diagnostic_june_relaxed_lag.csv", index=False)
    pd.DataFrame(cis).to_json(out / "bootstrap_ci.json", orient="records", indent=1)

    pd.set_option("display.width", 250)
    cols = ["model", "months", "frauds", "caught", "precision_at_40", "recall_at_40", "gross_avoided",
            "goodwill", "goodwill_undecided", "net_per_month", "net_if_contact_per_month",
            "pr_auc", "accuracy"]
    for r in ["pre_policy", "post_policy"]:
        print(f"\n=== STRICT walk-forward, {r} ===")
        print(summ[summ.regime == r][cols].to_string(index=False))
    print("\n=== post-policy month by month ===")
    keep = [FINAL_MODEL, "logreg_baseline", RULE, "random_40"]
    m = monthly[(monthly.regime == "post_policy") & monthly.model.isin(keep)]
    print(m[["month", "model", "frauds_in_month", "caught", "gross_avoided", "net"]]
          .round(0).to_string(index=False))
    print("\n=== DIAGNOSTIC (lag relaxed): June given all labels before 1 June ===")
    print(diag[["model", "frauds_in_month", "caught", "precision_at_40", "gross_avoided",
                "goodwill_known", "goodwill_undecided", "net", "net_if_contact"]].to_string(index=False))
    print("\nbootstrap 90% intervals:", cis)


if __name__ == "__main__":
    main()
