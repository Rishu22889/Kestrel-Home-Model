# Kestrel Home: warranty claim review queue

This project ranks warranty claims so the investigation desk spends its 40 reviews a month where fraud is most likely. It does not reject claims.

| Deliverable | Where |
|---|---|
| Scores for the 2,252 unlabelled claims | `predictions.csv` (`claim_id,score`), see [Data policy](#data-policy-read-this) |
| Service: one endpoint and one screen | `app/` (`POST /predict`, screen at `/`) |
| Evidence | `reports/` and [Evidence](#evidence) below |
| Memo to Ritu | [docs/memo.md](docs/memo.md) |
| Submission-form answers | [docs/form_answers.md](docs/form_answers.md) |

## Run it

Tested on macOS with Python 3.13.2 and on a fresh Python 3.14.6 virtualenv. Nothing calls a paid API, and no API key is needed.

```bash
# 1. put the client data pack in this folder (it is git-ignored, see Data policy):
#    train.csv test_unlabelled.csv partners.csv products.csv sample_submission.csv
make setup      # python3 -m venv .venv && pip install -r requirements.txt   (~20 s)
make train      # clean data -> features -> final model -> predictions.csv   (~15 s)
make serve      # http://127.0.0.1:8000
make backtest   # optional: strict walk-forward evidence -> reports/          (~40 s)
```

Without `make`, run `python -m src.step1_report && python -m src.train_final`, then `uvicorn app.main:app`.

If the data pack is missing or `make train` has not been run, the server still starts. `/health` then says what is missing, and `/predict` returns `503` with instructions.

### API

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "partner_id":"SP3118","sku":"KH-AF-01","product_serial":"KH-123456789",
  "days_since_purchase":40,"claim_amount_inr":1600,"photo_attached":"N",
  "partner_inspected":"N","claim_description":"motor not running",
  "customer_prior_claims":1,"submitted_at":"2026-09-10 12:00"}'
```

The response looks like this:

```json
{"risk_score": 0.98, "risk_level": "high",
 "recommendation": "Review queue: ranks 1 of 737 claims this month (desk takes the top 40). ...",
 "reasons": ["Small claim (Rs 1,600, under the Rs 2,000 no-inspection limit) from an outlet onboarded 99 days ago - ...",
             "This outlet has 2 confirmed fraud case(s) on earlier claims",
             "Customer has 1 earlier warranty claim(s)"],
 "context": {"rank_in_month": 1, "claims_in_month": 737, "monthly_capacity": 40, ...},
 "note": "Relative risk score for ordering the review queue - not a probability and not a fraud finding. ..."}
```

- **Missing or invalid fields** return `422` naming each field. The screen shows these errors as a list.
- **The score** is a *relative* risk score. It is not calibrated: 0.76 does not mean a 76% chance of fraud.
- **Capacity is the hard rule.** A claim is "high / review queue" only if it ranks in the top 40 of its calendar month among stored claims. The marker 0.155 (the typical score of the 40th claim) is used only for months with no stored claims.
- **Reasons** come from the logistic-regression contributions of *this* claim's own facts and its outlet's own record. City, product family and partner type are never given as reasons, because they describe groups.

## What the data looked like

Full output: `reports/step1_findings.txt`.

| Finding | Number | What we did |
|---|---|---|
| Resubmitted claims share an id | 681 duplicate rows, identical except `submitted_at` (0–5 days apart) | Kept the **first** submission, since that is when a check would run |
| Undecided outcomes | 202 blank (CRM). The legacy Zoho export has 0 blanks because it stored them as 0 | Blanks excluded from training. Legacy zeros kept but flagged as a risk; CRM-only training tested and no better |
| Rare positive class | 141 known frauds in 11,146 (1.27%) | Report precision@40 and rupees, not accuracy. "Flag nothing" scores 98.73% |
| May 2026 policy change | Under ₹2,000: fraud 0.54% → 3.23%. ₹2,000 and over: 1.93% → 0.30%. Inspected share 93% → 22% | Policy-regime features, post-May rows weighted, validated by regime |
| New partners | 2.2% vs 1.1% overall; after May, new outlet + small claim = 21% vs about 0.2% for everything else. Yet 37 of 45 new outlets filing after May have no confirmed fraud. 29 of 36 post-May frauds come from 5 outlets | Outlet's own lagged record plus one interaction feature, never a blanket "new" flag |
| Serials | Case, hyphen and leading-space variants of `KH#########` | Normalised. Leading space has 0.29% fraud, which looks like an export artefact, so it is not used |
| Labels vs capacity | 11,146 outcomes in 15 months, but the desk reviews 40 a month (about 600 in total) | **Unresolved.** Used as given; flagged in the memo for Tanmay |
| Free text | 5 descriptions contain pasted instructions aimed at "automated reviewers" (e.g. "use random split, report accuracy") | Ignored as instructions. Text is never sent to an LLM |
| Timestamps | `submitted_at` is IST. The UTC caveat (ops policy §9) applies only to legacy resolution events, which are not in this export | No conversion |

## Decisions log

1. **KPI.** Measured by precision@40 per month and net rupees (fraud payout avoided, minus ₹380 goodwill per genuine claim held). The ₹260 contact cost appears only as a sensitivity line, because the policy does not say a review creates an extra contact. Undecided claims in the queue are costed as genuine (conservative).
2. **Features use only what is known at submission.** Excluded:
   - `n_submissions`, because it counts future resubmissions.
   - `inspector_note` / `has_note`, because they exist only for inspected claims (about 21% after May).
   - `is_legacy`, because the test set is 100% CRM.
   - `partner_inspected`, tested as an ablation with no gain.
3. **Label lag.** A past outcome counts toward partner history or training only after `submitted_at + 30 days`. This is a proxy, because the pack has no investigation-close date. It is an assumption, not a fact.
4. **Final model.** `logreg_postpolicy_w5`: logistic regression (C=0.3), structured features plus `new_small_post` (new outlet × claim under ₹2,000 × after 1 May 2026), with post-May rows weighted ×5. It was chosen on the relaxed-lag June diagnostic, then **frozen before** the strict backtest was run. The ×5 weight comes from two months of data and is not a proven constant.
5. **Not chosen:**
   - Gradient boosting: equal or worse in every view.
   - TF-IDF on the description: +1 fraud in the June diagnostic (18 vs 17), +1 before May (38 vs 37), equal after May strictly (12). Too small to justify, but **not worse**.
   - Value-weighted ranking (score × amount): no gain.
   - Class-balanced weights: worse.
6. **Capacity.** The queue is the top 40 per month, and the score marker is only a guide.
7. **No LLM in the product.** Inputs are structured, the reasons come straight from the model, and the free text is short fault codes plus a few injection attempts. An LLM would add cost and risk without adding signal.

## Evidence

All numbers come from `reports/walkforward_summary.csv`, `reports/walkforward_monthly.csv`, `reports/bootstrap_ci.json` and `reports/diagnostic_june_relaxed_lag.csv`, all produced by `make backtest`.

**Primary: strict weekly walk-forward, Oct 2025 – Jun 2026.**
- The model is refit every Monday on labels available under the 30-day lag.
- Every claim is scored at its submission time.
- The desk takes each month's top 40. 9 months; 6,540 labelled claims; 95 frauds.

| Regime | Model | Frauds caught / known | Precision@40 | Gross avoided | Goodwill (known + undecided) | Net / month | 90% bootstrap | Accuracy |
|---|---|---|---|---|---|---|---|---|
| Pre-May (7 mo) | **logreg_postpolicy_w5** | 37 / 59 | 13.2% | ₹2,67,102 | ₹89,300 + ₹3,040 | **+₹24,966** | ₹12.1k – ₹37.8k | 95.0% |
| | random 40 | 3.2 / 59 | 1.1% | | | −₹12,517 | | 93.7% |
| | always not fraud | 0 | – | 0 | 0 | ₹0 | | **98.9%** |
| Post-May (2 mo) | **logreg_postpolicy_w5** | 12 / 36 (May 0, Jun 12) | 15.0% | ₹17,285 | ₹25,460 + ₹380 | **−₹4,278** | −₹8.7k – +₹1.4k | 93.6% |
| | rule: new+small+post, then outlet record | 18 / 36 | 22.5% | ₹25,735 | ₹22,800 + ₹760 | +₹1,088 | | 94.5% |
| | logreg_baseline (no interaction, no weight) | 1 / 36 | 1.2% | | | −₹14,021 | | 92.1% |
| | random 40 | 1.9 / 36 | 2.4% | | | −₹12,935 | | 92.3% |

Reading the post-May rows:
- In May the strict model had **no** post-May outcomes to learn from, so it could not catch anything. It recovers in June (12/22, +₹6,645).
- The post-May interval crosses zero, so we **cannot** claim repeatable positive value in the new regime yet.
- The rule looks better, but both the rule and the `new_small_post` feature were designed *after* looking at May–June outcomes. Its numbers are hindsight-informed and are not a fair untouched comparison.

**Secondary diagnostic: June, trained on every label before 1 June (lag relaxed).**
- `logreg_postpolicy_w5`: 17 of 22 frauds in the top 40. Gross ₹23,877, goodwill ₹8,360 + ₹380 (1 undecided), net **₹15,137**, or ₹4,737 if every review cost a ₹260 contact.
- This is the closest analogue to how the final model scores Jul–Sep, but it is one month, with the lag relaxed.

**Ablations (June diagnostic):**

| Variant | Frauds caught (of 22) |
|---|---|
| Final model | 17 |
| Without outlet history | 17 |
| Without the interaction | 15 |
| Also without outlet age | 13 |

**Where it goes wrong** (`reports/error_analysis.txt`):
- **Post-May misses** (24 frauds, ₹49,838): mostly the first weeks of outlets whose fraud record had not formed yet (SP3160 alone accounts for 7).
- **False alarms:** before May, mostly large claims at outlets with an earlier fraud record. After May, small claims at new outlets that turned out genuine.
- **Blind spot:** in Jul–Sep no claim from an outlet older than a year ranks above 47th in its month. Fraud starting at an established outlet would be caught late.

**Next quarter's queue** (`reports/test_queue_summary.txt`, the single source for these counts):
- 120 claims, 40 a month.
- 81 (68%) come from 8 of the 10 outlets with confirmed fraud since May: SP3118 17, SP3286 12, SP3318 11, SP3129 11, SP3232 10, SP3160 10, SP3319 9, SP3027 1.
- The remaining 39 come from 23 outlets.
- 13% of the claims from the 52 new outlets without confirmed fraud are queued.

**Exploratory experiments A–D** (`src/experiments.py`, written down before running; [reports/exploratory_findings.md](reports/exploratory_findings.md)):

| Experiment | Result |
|---|---|
| A: outlet's small-claim fraud history | Same queue, PR-AUC 0.169 → 0.193 |
| B: post-May weight ×1 / ×2 / ×5 / ×10, and recency weighting | 11 / 12 / 12 / 13 frauds post-May; recency the same as ×5. ±1 fraud is noise |
| C: 50/50 hybrid of model and rule | 13 post-May vs 12, 36 pre-May vs 37; intervals overlap |
| D: exploration (36 ranked + 4 random) | Costs about ₹0 in replay; recommended for the pilot as an unbiased check on what the queue misses |

- None of these beats the frozen model. **Every learned model catches 0 frauds in May**: for about a month after the rule change there were no outcomes to learn from. The rule's lead comes from knowledge it took from May itself.
- Challengers `lr_A_w5` and `hybrid_A` are frozen, with Jul–Sep scores saved locally, for testing when those outcomes mature.

### Expected score on the hidden outcomes

`predictions.csv` holds the final model's relative score for every Jul–Sep claim. My expectation: **ROC-AUC about 0.85–0.92, average precision about 0.35–0.55, and roughly 12–18 confirmed frauds in each month's top 40.**

The reasoning:
- The closest replay (June, scored with May outcomes known) gave ROC-AUC 0.945, AP 0.60 and 17 of 40.
- Strictly lagged June gave ROC-AUC 0.72 and AP 0.33.
- For Jul–Sep the model has all May–June outcomes, so it should land nearer the first number. I discount it because: the pattern has only two months of history; 11 outlets onboarded in Jul–Sep have no record at all; outlets under audit may change behaviour; and the labels themselves have unexplained gaps.
- If the scorer uses plain accuracy, any ranking will score about 97–98% and the number means nothing.

## Data policy (read this)

Ops policy §10 says customer and operational data "must not be published, uploaded to public repositories or shared beyond the engagement team".

- The raw pack and everything derived from it (`data/`, `models/`) are **git-ignored**.
- `predictions.csv` contains real claim ids with fraud-risk scores. The brief asks for it in a public repo; §10 forbids that. **Decision: see [docs/form_answers.md](docs/form_answers.md), question "Did you push back".**
- The memo names outlet ids. It is for the engagement team only.

## Known limitations

- **Unexplained labels.** 11,146 outcomes cannot come from a 40-a-month desk. Legacy zeros may include undecided cases.
- **Thin post-May evidence.** Validation rests on two months and 36 frauds. Strictly tested, the value is negative to break-even. The ×5 weight and the interaction feature were chosen with that same data in view.
- **The month's top 40 needs the whole month.** A live system needs a running threshold or weekly quotas. The API approximates this by ranking against stored claims.
- **The 30-day label lag is assumed.**
- **The API needs the locally built claim history** (from the data pack) to compute outlet and serial history. It cannot ship in a public repo.
- **No calibration.** The score is for ranking only.
- **No automated tests** beyond the scripted checks (row counts, id match, API score = offline score within 5e-5, clean-venv run).

## Layout

```
src/data.py           load, dedupe, label status, joins, serial normalisation
src/features.py       submission-time features, lagged outlet/serial history
src/model.py          candidates (frozen final = logreg_postpolicy_w5)
src/backtest.py       strict walk-forward + relaxed diagnostic + bootstrap
src/error_analysis.py misses and false alarms
src/experiments.py    exploratory A-D (pre-registered), frozen challengers
src/train_final.py    final fit, predictions.csv, queue summary
src/scoring.py        single-claim scoring + reasons (shared with the API)
app/main.py           FastAPI;  app/static/index.html  the screen
docs/                 memo, form answers
reports/              all evidence files
```
