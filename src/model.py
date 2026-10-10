"""Candidate models. Deliberately few: 141 known frauds do not support a zoo.

FINAL_MODEL was fixed before the strict walk-forward backtest was run, so the
backtest is an honest check of it, not a tuning loop.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import BINARY, CATEGORICAL, NUMERIC, TEXT

FINAL_MODEL = "logreg_postpolicy_w5"
POST_WEIGHT = 5.0  # chosen once on the relaxed-lag June diagnostic; not re-tuned

PARTNER_HISTORY = ["partner_prior_claims", "partner_claims_30d", "partner_small_share_90d",
                   "partner_fraud_rate_smoothed", "partner_prior_fraud"]
PARTNER_AGE = ["partner_age_days", "new_partner"]
FULL = NUMERIC + BINARY


def drop(*cols):
    flat = {c for group in cols for c in (group if isinstance(group, list) else [group])}
    return [c for c in FULL if c not in flat]


def _prep(features, text=False):
    parts = [
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), features),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50,
                              sparse_output=False), CATEGORICAL),
    ]
    if text:
        parts.append(("txt", TfidfVectorizer(ngram_range=(1, 2), min_df=5), TEXT))
    return ColumnTransformer(parts, sparse_threshold=0)


def logistic(features=FULL, text=False):
    return Pipeline([("prep", _prep(features, text)),
                     ("clf", LogisticRegression(C=0.3, max_iter=2000))])


def gbm(features=FULL):
    return Pipeline([("prep", _prep(features)),
                     ("clf", HistGradientBoostingClassifier(
                         learning_rate=0.05, max_iter=200, max_depth=3, min_samples_leaf=40,
                         l2_regularization=1.0, random_state=0))])


# name -> (builder, weight on post-May rows)
CANDIDATES = {
    "logreg_baseline": (lambda: logistic(drop("new_small_post")), 1.0),
    "logreg_postpolicy_w1": (lambda: logistic(), 1.0),
    "logreg_postpolicy_w5": (lambda: logistic(), POST_WEIGHT),
    "logreg_postpolicy_w5_text": (lambda: logistic(text=True), POST_WEIGHT),
    "gbm_postpolicy_w5": (lambda: gbm(), POST_WEIGHT),
    # ablations of the final model
    "abl_no_partner_history": (lambda: logistic(drop(PARTNER_HISTORY)), POST_WEIGHT),
    "abl_no_new_small_post": (lambda: logistic(drop("new_small_post")), POST_WEIGHT),
    "abl_no_partner_age": (lambda: logistic(drop(PARTNER_AGE, "new_small_post")), POST_WEIGHT),
}


def fit(name, train_df):
    build, w = CANDIDATES[name]
    weights = np.where(train_df.post_policy == 1, w, 1.0)
    return build().fit(train_df, train_df.is_fraud.astype(int), clf__sample_weight=weights)
