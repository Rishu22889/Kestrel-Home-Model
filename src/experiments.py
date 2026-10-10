"""Exploratory experiments A-D (written down BEFORE they were run; run once).

These are exploratory: May-June outcomes were already seen while building the
frozen model, so nothing here is unbiased proof, and nothing here replaces
FINAL_MODEL. A winner becomes a *challenger* to be tested on Jul-Sep outcomes
once they mature.

Pre-registered design (no parameter below was changed after seeing results)
A  Feature: outlet's lagged, smoothed fraud rate on SMALL claims only
   (partner_small_fraud_rate_smoothed, prior 1% x 20 pseudo-claims) plus the
   count of resolved small claims and of small frauds.            -> lr_A_w5
B  Weighting of the final feature set: post-May rows x1, x2, x5 (= final),
   x10; and recency weighting 0.5 ** (age_days / 90) at each refit.
C  Hybrid: within each month, 0.5 x percentile rank of lr_A_w5
   + 0.5 x percentile rank of rule_A (new_small_post + small-claim fraud rate).
   Fixed 50/50, never adjusted.
D  Review strategy on the frozen final model's scores:
   explore_random_4   36 top-ranked + 4 random from the rest (mean of 200 draws)
   explore_estab_4    36 top-ranked + the 4 highest-scored claims from outlets
                      at least a year old that are not already queued
Same strict weekly walk-forward as src/backtest.py (Oct 2025 - Jun 2026,
30-day label lag, top 40 per month). Extra columns: catches at new vs
established outlets, and catches outside the five outlets with the most
post-May fraud (dependence check).

Run:  python -m src.experiments   -> reports/exploratory_*.csv
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .backtest import CAPACITY, CONTACT, END, GOODWILL, START, load_train, rule_score
from .data import POLICY_CHANGE, ROOT
from .features import LABEL_LAG_DAYS
from .model import FULL, logistic

A_FEATURES = ["partner_small_resolved", "partner_small_fraud", "partner_small_fraud_rate_smoothed"]
HALF_LIFE = 90
TOP5_POST_MAY = ["SP3160", "SP3232", "SP3318", "SP3129", "SP3319"]  # dependence check only


def _post(w):
    return lambda tr, t0: np.where(tr.post_policy == 1, w, 1.0)


def _recency(tr, t0):
    return 0.5 ** ((t0 - tr.submitted_at).dt.days / HALF_LIFE)


MODELS = {  # name -> (builder, weight function)
    "lr_w1": (lambda: logistic(), _post(1)),
    "lr_w2": (lambda: logistic(), _post(2)),
    "lr_w5 (final)": (lambda: logistic(), _post(5)),
    "lr_w10": (lambda: logistic(), _post(10)),
    "lr_recency_hl90": (lambda: logistic(), _recency),
    "lr_A_w5": (lambda: logistic(FULL + A_FEATURES), _post(5)),
}


def rule_a(d):
    return (d.new_small_post + d.partner_small_fraud_rate_smoothed).values


def walk_forward(df):
    in_range = (df.submitted_at >= START) & (df.submitted_at < END)
    out = pd.DataFrame(np.nan, index=df.index[in_range], columns=list(MODELS))
    refits = sorted({START, *pd.date_range(START, END, freq="W-MON", inclusive="left")})
    for t0, t1 in zip(refits, refits[1:] + [END]):
        idx = df.index[(df.submitted_at >= t0) & (df.submitted_at < t1)]
        tr = df[(df.label_status == "known")
                & (df.submitted_at + pd.Timedelta(days=LABEL_LAG_DAYS) < t0)]
        for name, (build, weight) in MODELS.items():
            m = build().fit(tr, tr.is_fraud.astype(int), clf__sample_weight=weight(tr, t0))
            out.loc[idx, name] = m.predict_proba(df.loc[idx])[:, 1]
    return out


def stats(mdf, sel, score=None):
    """Queue stats for a selected set of row labels within one month."""
    q = mdf.loc[sel]
    fraud = q.is_fraud.fillna(0).astype(float) == 1
    undec = q.label_status == "undecided"
    known = mdf[mdf.label_status == "known"]
    y = known.is_fraud.astype(int)
    kf = known[y == 1]
    gross = q.loc[fraud, "claim_amount_inr"].sum()
    net = gross - GOODWILL * int((~fraud).sum())
    hit = q[fraud]
    return {
        "frauds": int(y.sum()), "caught": int(fraud.sum()),
        "precision_at_40": fraud.sum() / CAPACITY,
        "gross_avoided": gross, "net": net, "net_if_contact": net - CONTACT * CAPACITY,
        "pr_auc": average_precision_score(y, score.loc[known.index]) if score is not None and y.sum() else np.nan,
        "frauds_new": int(kf.new_partner.sum()), "caught_new": int(hit.new_partner.sum()),
        "frauds_estab": int((kf.new_partner == 0).sum()), "caught_estab": int((hit.new_partner == 0).sum()),
        "frauds_outside_top5": int((~kf.partner_id.isin(TOP5_POST_MAY)).sum()),
        "caught_outside_top5": int((~hit.partner_id.isin(TOP5_POST_MAY)).sum()),
        "undecided_in_queue": int(undec.sum()),
    }


def monthly(df, scores):
    rng = np.random.default_rng(0)
    rows = []
    sub = df.loc[scores.index]
    for month, mdf in sub.groupby(sub.submitted_at.dt.to_period("M")):
        reg = "post_policy" if month.start_time >= POLICY_CHANGE else "pre_policy"
        cands = {n: pd.Series(scores.loc[mdf.index, n].values, index=mdf.index) for n in MODELS}
        cands["rule (existing)"] = pd.Series(rule_score(mdf), index=mdf.index)
        cands["rule_A"] = pd.Series(rule_a(mdf), index=mdf.index)
        cands["hybrid_A"] = 0.5 * cands["lr_A_w5"].rank(pct=True) + 0.5 * cands["rule_A"].rank(pct=True)
        for n, s in cands.items():
            rows.append({"month": str(month), "regime": reg, "candidate": n,
                         **stats(mdf, s.nlargest(CAPACITY).index, s)})
        # D: review strategies on the frozen model's scores
        base = cands["lr_w5 (final)"]
        top36 = base.nlargest(CAPACITY - 4).index
        rest = mdf.index.difference(top36)
        draws = [stats(mdf, top36.append(pd.Index(rng.choice(rest, 4, replace=False))))
                 for _ in range(200)]
        rows.append({"month": str(month), "regime": reg, "candidate": "D: 36 final + 4 random",
                     **pd.DataFrame(draws).mean().to_dict()})
        est = base.loc[rest][mdf.loc[rest, "new_partner"] == 0].nlargest(4).index
        rows.append({"month": str(month), "regime": reg, "candidate": "D: 36 final + 4 established",
                     **stats(mdf, top36.append(est))})
    return pd.DataFrame(rows)


def bootstrap(df, scores, n=500):
    """90% interval of post-May monthly net, resampling claims within month.
    Every candidate is scored on the same resample in each draw."""
    rng = np.random.default_rng(1)
    sub = df.loc[scores.index]
    sub = sub[sub.submitted_at >= POLICY_CHANGE]
    months = [m for _, m in sub.groupby(sub.submitted_at.dt.to_period("M"))]
    names = ["lr_w5 (final)", "lr_A_w5", "lr_recency_hl90", "hybrid_A", "rule_A"]
    nets = {k: [] for k in names}
    for _ in range(n):
        tot = dict.fromkeys(names, 0.0)
        for m in months:
            pick = rng.integers(0, len(m), len(m))
            b = m.iloc[pick].reset_index(drop=True)
            sc = {k: pd.Series(scores.loc[m.index, k].values[pick])
                  for k in ["lr_w5 (final)", "lr_A_w5", "lr_recency_hl90"]}
            sc["rule_A"] = pd.Series(rule_a(b))
            sc["hybrid_A"] = 0.5 * sc["lr_A_w5"].rank(pct=True) + 0.5 * sc["rule_A"].rank(pct=True)
            for k in names:
                tot[k] += stats(b, sc[k].nlargest(CAPACITY).index)["net"]
        for k in names:
            nets[k].append(tot[k] / len(months))
    return {k: [round(float(v)) for v in np.percentile(v, [5, 50, 95])] for k, v in nets.items()}


def save_challengers(df):
    """Freeze the challengers now: fit on every known label, score Jul-Sep.
    When those outcomes mature, compare against predictions.csv with no refits.
    Local only (data/ is git-ignored: claim ids are operational data)."""
    full = pd.read_pickle(ROOT / "data" / "processed" / "features.pkl")
    tr = full[full.label_status == "known"]
    te = full[full.split == "test"].copy()
    build, weight = MODELS["lr_A_w5"]
    te["lr_A_w5"] = build().fit(tr, tr.is_fraud.astype(int),
                                clf__sample_weight=weight(tr, None)).predict_proba(te)[:, 1]
    te["rule"] = rule_score(te)
    te["rule_A"] = rule_a(te)
    month = te.submitted_at.dt.to_period("M")
    te["hybrid_A"] = (0.5 * te.groupby(month).lr_A_w5.rank(pct=True)
                      + 0.5 * te.groupby(month).rule_A.rank(pct=True))
    te[["claim_id", "submitted_at", "lr_A_w5", "hybrid_A", "rule"]].to_csv(
        ROOT / "data" / "processed" / "challengers_test_scores.csv", index=False)


def main():
    df = load_train()
    scores = walk_forward(df)
    m = monthly(df, scores)
    agg = m.groupby(["regime", "candidate"]).agg(
        caught=("caught", "sum"), frauds=("frauds", "sum"), precision_at_40=("precision_at_40", "mean"),
        pr_auc=("pr_auc", "mean"), gross_avoided=("gross_avoided", "sum"),
        net_per_month=("net", "mean"), net_if_contact=("net_if_contact", "mean"),
        caught_new=("caught_new", "sum"), frauds_new=("frauds_new", "sum"),
        caught_estab=("caught_estab", "sum"), frauds_estab=("frauds_estab", "sum"),
        caught_outside_top5=("caught_outside_top5", "sum"),
        frauds_outside_top5=("frauds_outside_top5", "sum")).round(3).reset_index()
    agg["recall_at_40"] = (agg.caught / agg.frauds).round(3)
    ci = bootstrap(df, scores)
    agg["post_net_p5_p50_p95"] = agg.apply(
        lambda r: ci.get(r.candidate) if r.regime == "post_policy" else None, axis=1)
    out = ROOT / "reports"
    m.round(3).to_csv(out / "exploratory_monthly.csv", index=False)
    agg.to_csv(out / "exploratory_summary.csv", index=False)
    pd.set_option("display.width", 260)
    for reg in ["pre_policy", "post_policy"]:
        print(f"\n=== EXPLORATORY, strict walk-forward, {reg} ===")
        print(agg[agg.regime == reg].drop(columns="regime")
              .sort_values("net_per_month", ascending=False).to_string(index=False))
    save_challengers(df)
    print("\n=== post-May by month ===")
    pm = m[m.regime == "post_policy"]
    print(pm.pivot(index="candidate", columns="month", values="caught").join(
        pm.pivot(index="candidate", columns="month", values="net").add_prefix("net_")).round(0).to_string())


if __name__ == "__main__":
    main()
