# JPY Behavioral Hypothesis Spec v1

## Objective

Define a pre-specified behavioral hypothesis for retail FX sentiment that:

- is grounded in observed empirical structure
- avoids researcher degrees of freedom
- can be evaluated out-of-sample and reused across repositories (e.g. market-phase-ml)

---

## Hypothesis (v1)

Retail traders exhibit systematic, regime-dependent failure modes in FX markets.

These failure modes are strongest in JPY cross pairs and depend on:

- trend direction relative to positioning
- trend strength
- persistence of sentiment

---

## Universe restriction

pair_group == "JPY_cross"

---

## Persistence condition

abs_sentiment >= 70  
AND extreme_streak_70 >= 3

This defines persistent extreme sentiment.

---

## Trend definition

For horizon h ∈ {12, 48}:

- trend_h: past return over h bars
- trend_dir_h: sign(trend_h)
- trend_strength_h: abs(trend_h)

---

## Trend strength buckets

trend_strength_bucket ∈ {weak, medium, strong, extreme}

Defined using global quantiles (q = 4) on trend_strength_h.

These buckets are fixed once computed and must not be re-tuned.

---

## Alignment definition

trend_alignment_h = crowd_side × trend_dir_h

- -1 → fight trend (counter-trend positioning)
- +1 → follow trend (trend-following positioning)

---

## Behavioral regimes

### Regime A — Early reversal (fight trend)

Condition:

- trend_alignment_h == -1  
- trend_strength_bucket ∈ {medium, strong}

Interpretation:

Retail traders attempt to call reversals and enter too early.

Expected outcome:

- Positive contrarian returns  
- Peak signal in "strong" (not extreme)

---

### Regime B — Late chasing (follow trend)

Condition:

- trend_alignment_h == +1  
- trend_strength_bucket == "extreme"

Interpretation:

Retail traders join trends after large moves and enter too late.

Expected outcome:

- Positive contrarian returns  
- Peak signal in "extreme"

---

## Null regions

The hypothesis explicitly assumes no strong signal in:

- non_JPY pairs  
- weak trends  
- non-persistent sentiment  

These regions act as internal controls.

---

## Target variable

For horizon h:

contrarian_ret_h = -sign(net_sentiment) × ret_h

Interpretation:

- Positive → fading the crowd is profitable  
- Negative → the crowd is correct  

---

## Time semantics

- snapshot_time: corrected UTC timestamp of sentiment observation  
- entry_time: H1 bar open (UTC)  
- features are defined at entry_time  
- forward returns are computed from entry_close  

Execution assumptions (e.g. entering at next bar close) are handled downstream and are not part of this dataset.

---

## Fixed parameters (must not change)

HORIZONS: [12, 48]  
PERSISTENCE: extreme_streak_70 >= 3  
THRESHOLD: abs_sentiment >= 70  
TREND_BUCKETS: quantile-based (q = 4)  
PAIR_GROUP: JPY_cross  

These parameters are considered part of the hypothesis definition and must remain fixed for validation.

---

## Evaluation protocol

### Pre-registration

All rules in this document must be fixed before evaluating new or held-out data.

No parameter tuning is allowed after inspecting evaluation results.

---

### Metrics

Evaluate each regime using:

- mean contrarian return  
- median contrarian return  
- hit rate (fraction of positive outcomes)  
- standard deviation  

---

### Stability checks

Evaluate robustness across:

- rolling time windows  
- subperiods (e.g. yearly or quarterly)  
- individual JPY pairs  

---

### Out-of-sample validation

Use either:

- walk-forward validation  
OR  
- a fully held-out time period  

The evaluation dataset must not have been used during hypothesis formation.

---

## Expected structure (summary)

Within JPY crosses and persistent sentiment:

- fight_trend:
  - signal peaks at strong trend strength  

- follow_trend:
  - signal peaks at extreme trend strength  

- non-JPY pairs:
  - no consistent positive signal  

---

## Implementation mapping

### market-sentiment-ml

Available fields:

- trend_alignment_{h}b  
- trend_strength_bucket_{h}b  
- extreme_streak_70  
- pair_group  

---

### market-phase-ml (suggested inputs)

- trend_alignment  
- trend_strength_bucket  
- persistence flag  
- pair_group  

---

### Example derived signal (optional)

signal_regime = (
    pair_group == "JPY_cross"
    AND extreme_streak_70 >= 3
    AND (
        (trend_alignment == -1 AND trend_strength_bucket IN {"medium", "strong"})
        OR
        (trend_alignment == +1 AND trend_strength_bucket == "extreme")
    )
)

---

## Scope and limitations

This hypothesis:

- does not claim universal profitability  
- does not account for execution costs or slippage  
- does not represent a production-ready trading system  

It defines a testable behavioral structure.

---

## Notes

This specification represents the transition from exploratory analysis to a structured, testable hypothesis.

Any future extensions (e.g. volatility conditioning, regime overlays) should be introduced as new versions (v2, v3) rather than modifying this specification.