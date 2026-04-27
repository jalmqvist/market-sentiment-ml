# JPY Behavioral Hypothesis Spec v1 (ARCHIVED)

## Status

⚠️ This hypothesis is now **invalidated under corrected validation**.

It is retained for documentation purposes only.

---

## Original hypothesis

Retail traders exhibit systematic, regime-dependent failure modes in FX markets.

These were hypothesized to be strongest in JPY cross pairs and dependent on:

- trend alignment  
- trend strength  
- persistence of sentiment  

---

## What changed

After correcting the research pipeline:

- overlapping signals removed  
- strict walk-forward validation enforced  
- proper aggregation used  
- regime holdout testing applied  

👉 The hypothesized effects **do not persist**

---

## Result

The following do **not** produce robust signals:

- JPY cross restriction  
- persistence conditioning  
- trend alignment  
- trend strength  
- volatility regime  

All effects collapse to:

- mean ≈ 0  
- Sharpe ≈ 0  

---

## Interpretation

The original hypothesis was influenced by:

- clustering in specific time periods  
- pair concentration (e.g. SGDJPY, CHFJPY)  
- overlapping trade exposure  
- in-sample bias  

---

## Conclusion

> The JPY-specific behavioral hypothesis is **not supported** under strict validation.

---

## Forward direction

The project now moves to:

### Regime v2 — Behavioral modeling

Focus shifts to:

- crowd dynamics (not price regimes)  
- persistence, acceleration, saturation  
- cross-pair generalization  

---

## Note

This document remains as a record of the research process:

- exploratory finding  
- hypothesis formalization  
- rigorous invalidation  

This progression is intentional and reflects correct scientific workflow.