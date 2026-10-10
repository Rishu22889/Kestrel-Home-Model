"""Feature engineering. Every feature uses only information available when the
claim is submitted (no future claims, no outcomes that were not yet decided).
"""
import numpy as np
import pandas as pd

from .data import AUTO_APPROVE_LIMIT, POLICY_CHANGE

# Investigation outcomes are not known instantly. A past claim's label only
# counts toward partner history once it is this many days old.
LABEL_LAG_DAYS = 30
PRIOR_STRENGTH = 20  # smoothing for partner fraud rate (pseudo-claims at base rate)

# Fixed prior for smoothing (not fitted on the data, so no future leakage).
PRIOR_RATE = 0.01

# Pre-payout feature set: only fields known when the partner submits the claim.
# Excluded on purpose (see README):
#   n_submissions      - counts later resubmissions -> future leakage
#   inspector_note/has_note - exist only for inspected claims (~21% post-May)
#   is_legacy          - system flag; test is 100% CRM, and it proxies time
#   serial_space       - looks like an export artefact (0.29% fraud), not behaviour
#   long_description   - 4 rows, no signal
NUMERIC = [
    "log_amount", "amount_to_price", "days_since_purchase",
    "warranty_used_frac", "customer_prior_claims", "list_price_inr",
    "partner_age_days", "partner_prior_claims", "partner_claims_30d",
    "partner_small_share_90d", "partner_fraud_rate_smoothed",
    "partner_prior_fraud", "serial_prior_claims", "serial_prior_partners",
    "submit_hour", "submit_dow",
]
BINARY = [
    "is_small", "post_policy", "small_post_policy", "photo", "new_partner",
    "new_small_post",
    "serial_lowercase", "serial_hyphen",
]
CATEGORICAL = ["family", "partner_type", "city"]
TEXT = "claim_description"
# Tested as an ablation only: sign-off is part of the submission, but since May
# it mostly encodes the policy (small claims are not inspected), not fraud.
OPTIONAL = ["inspected"]


def _partner_history(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling per-partner stats, computed strictly from earlier claims."""
    out = pd.DataFrame(index=df.index)
    known = (df.label_status == "known")
    fraud = (df.is_fraud.fillna(0) == 1) & known
    small = df.claim_amount_inr < AUTO_APPROVE_LIMIT

    cols = {k: np.zeros(len(df)) for k in
            ["prior", "c30", "small90", "n90", "lab", "fr", "slab", "sfr"]}
    for _, idx in df.groupby("partner_id").indices.items():
        t = df.submitted_at.values[idx].astype("datetime64[m]").astype(np.int64)
        k = known.values[idx].astype(float)
        f = fraud.values[idx].astype(float)
        sm = small.values[idx].astype(float)
        ck, cf, cs = np.cumsum(k), np.cumsum(f), np.cumsum(sm)
        csk, csf = np.cumsum(k * sm), np.cumsum(f * sm)  # resolved / fraud among small claims
        day = 24 * 60
        for j in range(len(idx)):
            cols["prior"][idx[j]] = j
            lo30 = np.searchsorted(t, t[j] - 30 * day)
            lo90 = np.searchsorted(t, t[j] - 90 * day)
            cols["c30"][idx[j]] = j - lo30
            n90 = j - lo90
            cols["n90"][idx[j]] = n90
            cols["small90"][idx[j]] = (cs[j - 1] - (cs[lo90 - 1] if lo90 else 0)) if n90 else 0
            # labels visible only for claims older than LABEL_LAG_DAYS
            hi = np.searchsorted(t, t[j] - LABEL_LAG_DAYS * day, side="right")
            cols["lab"][idx[j]] = ck[hi - 1] if hi else 0
            cols["fr"][idx[j]] = cf[hi - 1] if hi else 0
            cols["slab"][idx[j]] = csk[hi - 1] if hi else 0
            cols["sfr"][idx[j]] = csf[hi - 1] if hi else 0

    base = PRIOR_RATE
    out["partner_prior_claims"] = cols["prior"]
    out["partner_claims_30d"] = cols["c30"]
    out["partner_small_share_90d"] = np.where(
        cols["n90"] > 0, cols["small90"] / np.maximum(cols["n90"], 1), np.nan)
    out["partner_prior_fraud"] = cols["fr"]
    out["partner_fraud_rate_smoothed"] = (
        (cols["fr"] + PRIOR_STRENGTH * base) / (cols["lab"] + PRIOR_STRENGTH))
    # Experiment A (exploratory, not in the frozen model): the outlet's record on
    # SMALL claims only, same lag and smoothing. With 0 resolved small claims the
    # rate equals the prior - "no evidence", not "proven low risk"; the resolved
    # count is kept alongside so a model can tell the two apart.
    out["partner_small_resolved"] = cols["slab"]
    out["partner_small_fraud"] = cols["sfr"]
    out["partner_small_fraud_rate_smoothed"] = (
        (cols["sfr"] + PRIOR_STRENGTH * base) / (cols["slab"] + PRIOR_STRENGTH))
    return out


def _serial_history(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["serial_prior_claims"] = df.groupby("serial_norm").cumcount()
    seen, prior_partners = {}, np.zeros(len(df))
    for i, (s, p) in enumerate(zip(df.serial_norm.values, df.partner_id.values)):
        ps = seen.setdefault(s, set())
        prior_partners[i] = len(ps - {p})
        ps.add(p)
    out["serial_prior_partners"] = prior_partners
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df must be sorted by submitted_at (load_claims does this)."""
    assert df.submitted_at.is_monotonic_increasing
    f = df.copy()
    raw = f.product_serial.astype(str)

    f["log_amount"] = np.log1p(f.claim_amount_inr)
    f["amount_to_price"] = f.claim_amount_inr / f.list_price_inr
    f["warranty_used_frac"] = f.days_since_purchase / (f.warranty_months * 30.44)
    f["partner_age_days"] = (f.submitted_at - f.onboarded_date).dt.days
    f["new_partner"] = (f.partner_age_days < 365).astype(int)
    f["submit_hour"] = f.submitted_at.dt.hour
    f["submit_dow"] = f.submitted_at.dt.dayofweek
    f["month"] = f.submitted_at.dt.to_period("M").astype(str)

    f["is_small"] = (f.claim_amount_inr < AUTO_APPROVE_LIMIT).astype(int)
    f["post_policy"] = (f.submitted_at >= POLICY_CHANGE).astype(int)
    f["small_post_policy"] = f.is_small * f.post_policy
    # post-May fraud sits almost entirely in small, uninspected claims from
    # recently onboarded outlets; a linear model needs this spelled out
    f["new_small_post"] = f.new_partner * f.small_post_policy
    f["photo"] = (f.photo_attached == "Y").astype(int)
    f["inspected"] = (f.partner_inspected == "Y").astype(int)
    f["has_note"] = f.inspector_note.notna().astype(int)
    f["is_legacy"] = (f.source == "legacy_zoho").astype(int)

    f["serial_lowercase"] = raw.str.contains(r"[a-z]").astype(int)
    f["serial_hyphen"] = raw.str.contains("-").astype(int)
    f["serial_space"] = (raw != raw.str.strip()).astype(int)
    # a handful of descriptions carry long pasted text instead of a fault code
    f["long_description"] = (f.claim_description.str.len() > 40).astype(int)

    f["inspector_note"] = f.inspector_note.fillna("")
    f = pd.concat([f, _partner_history(f), _serial_history(f)], axis=1)
    return f
