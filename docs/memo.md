# Warranty fraud: what the data says and what to do next week

**To:** Ritu Deshpande, Head of D2C Operations  
**Cc:** Farhan Sheikh, Meenal Joshi, Tanmay Kulkarni  
**Data:** 11,348 unique historical claims (Apr 2025–Jun 2026); 2,252 claims from Jul–Sep 2026 to score

## Decision
Run a three-month, human-in-the-loop pilot: review **36 highest-ranked claims plus four random claims from the remainder each month**, capped at 40. The random sample is intended to reveal patterns the model misses; it has not been shown to improve short-term financial returns. Audit the eight outlets below. The tool prioritises reviews; it must never reject a claim automatically.

## Why not accuracy above 97%?
A model that flags nothing achieves about **98.7% accuracy** because fraud is rare, yet catches no fraud. We therefore focus on precision@40, recall@40 and estimated net rupee impact.

## What changed in May
From 1 May 2026, claims under ₹2,000 could be auto-approved without inspection. The observed fraud rate among small claims rose from **0.54% to 3.23%**; for larger claims it fell from **1.93% to 0.30%**. In May–June, **29 of 36 confirmed frauds came from five outlets**. Of 45 new outlets filing claims then, 37 had no confirmed fraud in the supplied records. This supports targeted checks, not blanket suspicion of new partners; it does not prove the policy change alone caused the shift.

## Historical replay results
These are retrospective estimates, not guaranteed savings. The strict backtest ranked all 40 slots; the proposed 36-ranked-plus-4-random pilot is a different strategy.

| Period | Strict ranked-queue result | Net estimate/month | 90% range |
|---|---|---:|---:|
| Oct 2025–Apr 2026 | 37/59 known frauds caught; precision@40 13.2% | **+₹24,966** | ₹12.1k to ₹37.8k |
| May–Jun 2026 | 12/36 known frauds caught; precision@40 15.0% | **−₹4,278** | −₹8.5k to +₹1.0k |
| June relaxed-lag diagnostic only | 17 frauds among 40 reviewed | **+₹15,137**; ₹4,737 if each review adds a ₹260 contact | One month only |

The June diagnostic allows May outcomes earlier than the assumed 30-day lag, so it is not the strict post-policy estimate. The strict post-policy estimate is negative and its interval crosses zero; recurring positive value is not established. June's confirmed fraud value was about ₹45,700, versus about ₹47,600 per month before May; median fraud value fell from about ₹3,392 to ₹1,416. Reviewing the ₹2,000 auto-approval rule for outlets with a fraud record may matter more than further model tuning.

## July–September queue and next steps
The current queue summary reports 120 candidates across three months. **81 (68%)** are attributed to eight outlets with confirmed fraud since May: **SP3118, SP3286, SP3318, SP3129, SP3232, SP3160, SP3319, SP3027**. The other 39 are spread across 23 outlets. Of 52 new outlets without confirmed fraud in the supplied records, about 13% of their claims are queued. **Verify these counts against the final regenerated summary for the 36-ranked-plus-4-random strategy before circulation.**

1. **Meenal:** start the 36+4 pilot and record random selections; audit the eight outlets and consider inspection for their claims under ₹2,000 during the audit. Do not apply a blanket rule to new partners or automatically reject claims.
2. **Tanmay:** record investigation outcomes and closure dates; reconcile why 11,146 claims have outcomes when stated desk capacity is 40/month, and investigate possible undecided legacy zeros.
3. **Farhan:** report fraud found, payout prevented, genuine customers delayed (₹380 average goodwill), additional contacts if they actually occur, and false alarms by outlet each month.
4. **Ritu and the team:** review the pilot after three months and decide whether to continue the queue or change the auto-approval rule for outlets with confirmed fraud.

## Known blind spots
The model learns from recorded outcomes. With an assumed 30-day label lag, it may be slow to learn new patterns after a rule change; in May it caught 0 of 14 known frauds in strict replay. Agree a temporary review strategy for affected claims after future rule changes. The model also rarely ranks established outlets highly and has limited history for brand-new outlets; four random reviews are only a small sample. **Data caveat:** historical outcome counts do not reconcile with stated review capacity, so metrics and rupee estimates remain provisional until label generation and legacy labels are verified.
