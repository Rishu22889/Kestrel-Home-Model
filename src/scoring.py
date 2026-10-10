"""Score one claim with the saved final model and explain it in plain words.

Uses the same build_features() as training. History features (partner record,
serial reuse) are computed against the locally built claim history, restricted
to the claim's partner and serial - the only rows those features depend on.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import ROOT, normalise_serial
from .features import build_features

MODELS = ROOT / "models"
HISTORY = ROOT / "data" / "processed" / "features.pkl"
RAW_COLS = ["claim_id", "submitted_at", "partner_id", "sku", "product_serial",
            "days_since_purchase", "claim_amount_inr", "photo_attached",
            "partner_inspected", "claim_description", "inspector_note",
            "customer_prior_claims", "source"]


class NotReady(RuntimeError):
    pass


class Scorer:
    def __init__(self):
        missing = [p for p in [MODELS / "final_model.joblib", MODELS / "meta.json",
                               MODELS / "lookups.joblib", HISTORY] if not p.exists()]
        if missing:
            raise NotReady("Model files not found: " + ", ".join(str(m.relative_to(ROOT)) for m in missing)
                           + ". Put the data pack in the project folder and run `make train` "
                             "(see README).")
        self.model = joblib.load(MODELS / "final_model.joblib")
        self.meta = json.loads((MODELS / "meta.json").read_text())
        lk = joblib.load(MODELS / "lookups.joblib")
        self.partners = lk["partners"].set_index("partner_id")
        self.products = lk["products"].set_index("sku")
        self.history = pd.read_pickle(HISTORY)
        prep = self.model.named_steps["prep"]
        self.names = prep.get_feature_names_out()
        self.coef = self.model.named_steps["clf"].coef_[0]
        # every stored claim's score, to place a new claim in its month's queue
        h = self.history
        self.stored = h[["claim_id"]].assign(
            month=h.submitted_at.dt.to_period("M"), score=self.model.predict_proba(h)[:, 1])
        self.capacity = int(self.meta["monthly_capacity"])

    def _month_rank(self, claim_id, ts, p):
        """Rank of the claim among stored claims of the same calendar month
        (1 = highest). None when there are no stored claims for that month."""
        same = self.stored[(self.stored.month == ts.to_period("M")) & (self.stored.claim_id != claim_id)]
        if same.empty:
            return None, 0
        return int((same.score > p).sum()) + 1, len(same) + 1

    # ---------- features ----------
    def _features(self, claim: dict) -> pd.Series:
        c = {k: claim.get(k) for k in RAW_COLS}
        if c["partner_id"] not in self.partners.index:
            raise ValueError(f"Unknown partner_id {c['partner_id']!r}")
        if c["sku"] not in self.products.index:
            raise ValueError(f"Unknown sku {c['sku']!r}")
        c["claim_id"] = c["claim_id"] or "NEW"
        c["submitted_at"] = pd.Timestamp(c["submitted_at"] or pd.Timestamp.now())
        c["source"] = c["source"] or "crm"
        row = pd.DataFrame([c])
        row = row.join(self.partners, on="partner_id").join(self.products, on="sku")
        row["serial_norm"] = normalise_serial(row.product_serial)
        row["split"], row["label_status"], row["is_fraud"] = "api", "unlabelled", np.nan
        row["n_submissions"] = 1

        h = self.history
        h = h[((h.partner_id == c["partner_id"]) | (h.serial_norm == row.serial_norm.iloc[0]))
              & (h.claim_id != c["claim_id"])]
        both = pd.concat([h[row.columns], row], ignore_index=True)
        both = both.sort_values("submitted_at", kind="stable").reset_index(drop=True)
        feats = build_features(both)
        return feats[feats.split == "api"].iloc[0]

    # ---------- explanation ----------
    def _contributions(self, x: pd.DataFrame) -> pd.Series:
        z = self.model.named_steps["prep"].transform(x)[0]
        contrib = pd.Series(z * self.coef, index=self.names)
        # group one-hot / tf-idf columns back to their source field
        key = [n.split("__", 1)[1] for n in self.names]
        key = [k.split("_")[0] if n.startswith("cat__") else k for k, n in zip(key, self.names)]
        key = [k if not n.startswith("cat__") else next(c for c in ("family", "partner_type", "city")
               if n.startswith(f"cat__{c}")) for k, n in zip(key, self.names)]
        return contrib.groupby(key).sum()

    @staticmethod
    def _reason(field, f):
        """Plain-English reason for a risk-raising factor, or None when the
        wording would not be true for this claim. City, product family and
        partner type are never given as reasons: they describe groups, and
        the desk should not read a claim as risky because of where it is from."""
        amt, age = f.claim_amount_inr, int(f.partner_age_days)
        new = age < 365
        if field == "new_small_post" and f.new_small_post:
            return (f"Small claim (Rs {amt:,.0f}, under the Rs 2,000 no-inspection limit) from an "
                    f"outlet onboarded {age} days ago - the pattern behind most confirmed fraud "
                    f"since May 2026")
        if field in ("partner_prior_fraud", "partner_fraud_rate_smoothed") and f.partner_prior_fraud >= 1:
            return f"This outlet has {int(f.partner_prior_fraud)} confirmed fraud case(s) on earlier claims"
        if field == "amount_to_price" and f.amount_to_price >= 0.5:
            return f"Claim is {f.amount_to_price:.0%} of the product's list price"
        if field == "customer_prior_claims" and f.customer_prior_claims >= 1:
            return f"Customer has {int(f.customer_prior_claims)} earlier warranty claim(s)"
        if field == "partner_claims_30d" and f.partner_claims_30d >= 4:
            return f"Outlet submitted {int(f.partner_claims_30d)} claims in the previous 30 days"
        if field == "partner_small_share_90d" and f.partner_small_share_90d >= 0.7:
            return "Most of this outlet's recent claims are under Rs 2,000"
        if field in ("partner_age_days", "new_partner") and new:
            return f"Outlet onboarded only {age} days ago (limited track record)"
        if field in ("days_since_purchase", "warranty_used_frac") and f.days_since_purchase <= 90:
            return f"Claim made only {int(f.days_since_purchase)} days after purchase"
        if field in ("serial_prior_claims",) and f.serial_prior_claims >= 1:
            return f"This serial number appears on {int(f.serial_prior_claims)} earlier claim(s)"
        if field == "serial_prior_partners" and f.serial_prior_partners >= 1:
            return "This serial number was claimed earlier through a different outlet"
        if field == "photo" and f.photo == 0:
            return "No photo attached"
        if field in ("serial_hyphen", "serial_lowercase") and (f.serial_hyphen or f.serial_lowercase):
            return "Serial number typed in a non-standard format"
        return None

    def score(self, claim: dict) -> dict:
        f = self._features(claim)
        x = f.to_frame().T.infer_objects()
        p = float(self.model.predict_proba(x)[0, 1])
        thr = self.meta["review_threshold"]
        contrib = self._contributions(x).sort_values(ascending=False)
        reasons, seen = [], set()
        for field, v in contrib.items():
            if v < 0.2 or len(reasons) == 3:
                break
            r = self._reason(field, f)
            if r and r not in seen:
                reasons.append(r); seen.add(r)
        rank, n_month = self._month_rank(f.claim_id, f.submitted_at, p)
        # Capacity is the hard control: only a month's top 40 go to the desk.
        # The score marker is a guide for months with no stored claims yet.
        in_queue = rank <= self.capacity if rank is not None else p >= thr
        level = "high" if in_queue else "medium" if p >= thr / 4 else "low"
        if in_queue:
            rec = (f"Review queue: ranks {rank} of {n_month} claims this month (desk takes the top "
                   f"{self.capacity}). Hold payout until the desk has looked at it."
                   if rank is not None else
                   f"Above the review marker ({thr:.3f}); no other claims stored for this month, "
                   f"so confirm it is within the month's top {self.capacity} before holding payout.")
        elif level == "medium":
            rec = (f"Ranks {rank} of {n_month} this month, outside the top {self.capacity} the desk "
                   f"can review. Process normally." if rank is not None else
                   "Below the review marker. Process normally.")
        else:
            rec = "Process normally."
        return {
            "claim_id": f.claim_id,
            "risk_score": round(p, 4),
            "risk_level": level,
            "recommendation": rec,
            "reasons": reasons or ["No single factor stands out; score is close to the typical claim."],
            "context": {
                "rank_in_month": rank,
                "claims_in_month": n_month or None,
                "monthly_capacity": self.capacity,
                "score_marker": round(thr, 4),
                "outlet_onboarded_days_ago": int(f.partner_age_days),
                "outlet_confirmed_fraud_cases": int(f.partner_prior_fraud),
            },
            "note": "Relative risk score for ordering the review queue - not a probability and not "
                    "a fraud finding. Only an investigation can decide a claim.",
        }

    def example(self, claim_id: str) -> dict:
        h = self.history[self.history.claim_id == claim_id]
        if h.empty:
            raise KeyError(claim_id)
        r = h.iloc[0][RAW_COLS].to_dict()
        r["submitted_at"] = str(r["submitted_at"])
        r["inspector_note"] = r["inspector_note"] or None
        return r

    def example_ids(self, n=12):
        t = self.history[self.history.split == "test"]
        p = self.model.predict_proba(t)[:, 1]
        order = t.assign(p=p).sort_values("p", ascending=False)
        pick = pd.concat([order.head(n // 2), order.iloc[len(order) // 2: len(order) // 2 + n // 2]])
        return pick.claim_id.tolist()
