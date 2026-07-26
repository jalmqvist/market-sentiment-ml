# DL vs ABM Reconciliation — Regime-Dependent Predictability

## Purpose

This note reconciles two parallel research tracks:

- **Deep Learning (DL):** empirical detection of predictive signal
- **Agent-Based Modeling (ABM):** behavioral explanation of sentiment structure

The goal is to understand whether both approaches describe the same underlying
mechanism. The document now covers two reconciliation layers:

- **Layer 1 (original):** T/V regime-dependent predictability
- **Layer 2 (July 2026):** Reactive-JPY lifecycle-dependent predictability

---

## Layer 1 — T/V Regime Reconciliation

### DL Findings (Empirical)

Controlled experiments (fixed horizon, no grid search) show:

- Weak predictive signal (F1 ≈ 0.25–0.50)
- Stable across MLP and LSTM
- Strong dependence on market regime

#### Regime hierarchy

| Regime                | Signal    |
| --------------------- | --------- |
| LVTF (low-vol trend)  | strongest |
| HVR (high-vol range)  | moderate  |
| LVR (low-vol range)   | unstable  |
| HVTF (high-vol trend) | weak      |

**Key observation:** regime effects dominate pair effects.

### ABM Findings (Behavioral)

A minimal agent-based model reproduces sentiment when:

- agents accumulate positions over time
- agents exhibit inertia (resist switching)
- agents reinforce positions when aligned with price

This produces persistent sentiment imbalance, clustering, and path dependence.

**Limitation:** the model fails on JPY, CHF, and macro/flow-driven markets.

### Reconciliation

The apparent mismatch — ABM assumes trend → accumulation → signal, but DL
shows signal exists only in *some* trending regimes — resolves as:

> **trend + stability → accumulation → predictive signal**
> **trend + high volatility → breakdown**

Predictive signal requires both directional structure (trend) and temporal
persistence (stability / low volatility). When stability is absent, sentiment
reacts rather than accumulates and predictive structure disappears.

### Conceptual Model (Layer 1)

```text
market regime → governs stability
    stability + trend → enables accumulation
    accumulation → produces sentiment structure
    sentiment structure → weak predictive signal
```

### Open Questions (Layer 1)

- What behavioral mechanism links volatility to accumulation breakdown?
- How does persistence vary across regimes?
- Why are some pairs structurally different (JPY, CHF)?
- Can ABM reproduce DL signal magnitudes?

------

## Layer 2 — Reactive-JPY Lifecycle Reconciliation

*Added July 2026 following completion of ABM Stages 3 and 4.*

### Motivation

The Layer 1 reconciliation applies to the T/V Behavioral Surface. The Reactive-JPY Behavioral Surface (frozen July 2026) operates on a different scientific object: **consensus lifecycle structure** rather than instantaneous regime. Layer 2 records the mechanistic explanation for why the Reactive-JPY predictive gradient exists.

The key empirical constraint (F-007) is that Reactive-JPY and T/V Behavioral Surfaces produce substantially different prediction artifacts — they encode distinct predictive representations, not alternative parameterisations of a common signal. Any mechanistic explanation must account for this independence.

### DL Finding (Reactive-JPY)

Both MLP and LSTM architectures consistently show (F-003, confirmed):

> **JPY_CONSENSUS_MATURING is the most predictive state**

The ordering MATURING > ENTRY > MATURE holds across architectures and forward horizons, with the 24-bar horizon showing the strongest gradient.

This finding is stable, reproducible, and cannot be recovered from instantaneous T/V regime classification alone (F-007).

### The Lifecycle Hypothesis

The gradient is interpretable as a consequence of **lifecycle position** within an endogenously generated crowd dynamic:

- Episodes that have survived early dissolution have selected for the most entrenched crowd positions.
- The rising dissolution hazard in the maturation zone means an increasing fraction of the crowd is positioned against imminent reversal pressure.
- This creates a structurally predictable contrarian setup that is invisible to instantaneous classifiers but recoverable from lifecycle position.

### Evidence for the Endogenous Interpretation

Two independent lines of evidence support this interpretation:

**1. News-shock null (F-008, confirmed)**

Scheduled JPY news events (high + medium impact, 6,379 events, 2007–2026) show no meaningful association with consensus episode exits. REVERSAL and THRESHOLD exit types have near-identical news proximity distributions, both at or below the Poisson base rate for the observed news calendar density. Observable macro shocks are not the primary dissolution trigger.

**2. ABM mechanistic reproduction (Stages 3 and 4)**

The ABM reproduces the MATURING > ENTRY > MATURE gradient without any predictive training, purely from persistence + decay + volatility-conditioned shock dynamics:

- *Stage 3 shock sweep:* vol_t80_f30_cd10 produces FULL H4 on EUR-JPY and GBP-JPY (20 seeds each). USD-JPY excluded from primary test surface due to structural MATURE cell limitation (n=54).
- *Stage 4 robustness sweep:* FULL H4 on EUR-JPY + GBP-JPY is stable across the fraction dimension (f=0.30, 0.40, 0.50 all produce 2/3 FULL). cd10 is confirmed as a cooldown optimum (inverted-U gradient; over-shocking at cd5 degrades the MATURING signal by disrupting lifecycle completion).

The gradient is generated by the mechanism, not by fitting to the empirical outcome.

### Mechanistic Constraint: Cooldown and Lifecycle Integrity

The Stage 4 finding that cd5 degrades H4 provides a mechanistic constraint with predictive content:

> Any shock mechanism whose mean inter-shock interval falls below the mature_boundary (24 bars) will disrupt lifecycle completion and degrade the MATURING signal.

At cd=10 with volatility threshold=0.80, the mean inter-shock interval is approximately 22 bars — near but above the critical boundary. This is the operative regime. At cd=5 (~13 bars) the boundary is violated and the gradient collapses on USD-JPY (|r|MATURING drops from 0.079 to 0.012).

### Known Structural Limitation: H3 Frequency Gap

The ABM overproduces short episodes relative to the empirical BSVE targets:

- Simulated freq/1k = 58–70 vs empirical target 45–56
- Simulated median duration ≈ 3 bars vs empirical target ≈ 4 bars
- Reversal gradient (rev_young > rev_mature) = 0 across all tested configurations

This gap persists across all shock parameter combinations tested in Stages 3 and 4. The reversal gradient failure is interpretable: at median duration ~3 bars, most simulated episodes exit before reaching young_boundary (8 bars), so no episode accumulates enough history for the hazard-rate comparison to operate. This is a hard structural property of the persistence + decay + shock mechanism at the current calibrated point.

The H3 gap does **not** invalidate the H4 finding. The predictive gradient test (H4) operates on the distributional alignment of ABM sentiment with BSVE-labelled state windows, not on the episode extractor output. The two tests measure different scientific objects.

Resolution of the H3 gap is deferred to Stage 5.2 (episode extractor sensitivity analysis — see roadmap).

### Updated Conceptual Model (Layer 2)

market regime (trend/vol) → governs shock frequency
    shock + price direction → episode formation (rapid, directional)
    persistence + anchoring → episode sustaining (characteristic duration)
    decay (vol-conditioned) → episode dissolution (hazard-rate structure)
        ↓
    consensus lifecycle position → weak predictive signal
        (MATURING most predictive: entrenched crowd near dissolution threshold)

This model is consistent with:

- F-007 (T/V and Reactive-JPY independence: they describe different layers)
- F-008 (news-shock null: dissolution is endogenous)
- F-003 (MATURING most predictive: lifecycle position, not news proximity)
- Stage 3/4 ABM results (gradient reproduced mechanistically)

### Three-Level Success Criteria

Mechanistic success is defined at three cumulative levels:

| Level                         | Criterion                                                    | Status                                                       |
| ----------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **L1 — Episode Statistics**   | Simulated reversal_rate_young > reversal_rate_mature; freq/dur in plausible range | **NOT MET** — H3 freq/dur gap structural at current calibration |
| **L2 — Hazard Structure**     | Simulated hazard curve matches empirical shape qualitatively | **NOT TESTED** — pending L1 resolution                       |
| **L3 — Predictive Structure** | ABM sentiment shows MATURING > ENTRY > MATURE forward-return gradient | **MET** — EUR-JPY + GBP-JPY FULL, stable across Stage 4 robustness sweep |

Note: L3 is met before L1/L2. This is not a contradiction — L3 tests distributional alignment of ABM sentiment with BSVE state windows (a coarser test), while L1/L2 test the episode extraction directly (a finer test requiring correct short-duration episode structure). Closing the H3 gap (Stage 5.2) is the path to L1/L2.

### What Has Been Learned (Layer 2)

- The Reactive-JPY predictive gradient is mechanistically reproducible from first principles without predictive training.
- The gradient is endogenous to crowd lifecycle dynamics, not driven by exogenous news shocks.
- Cooldown is the critical shock parameter; fraction is insensitive in the tested range.
- The shock mechanism is a necessary (not sufficient) condition for H4: the gradient collapses without shocks, and is disrupted by over-shocking.
- USD-JPY is structurally limited as an H4 test surface due to small MATURE cell size (n=54); EUR-JPY + GBP-JPY form the reliable test surface.
- The H3 freq/duration gap is a structural property of the current calibrated point, not a parameter tuning problem.

------

## Combined Conclusion

Deep learning and ABM now describe different layers of the same system at two distinct levels of behavioral organisation:

**Layer 1 (T/V):**

> Predictability in retail sentiment is **conditional, regime-dependent, and structurally constrained** by the interplay of trend direction and market stability.

**Layer 2 (Reactive-JPY):**

> The lifecycle position of a consensus crowd episode determines its predictive content. Maturation selects for the most entrenched positions and places the crowd on the rising limb of the dissolution hazard curve — a structurally predictable setup that is invisible to instantaneous classifiers but reproduced by the persistence + decay + shock mechanism.

DL detects *where* the signal exists. ABM explains *why* it exists at that lifecycle position and not others. F-007 (surface independence) is explained by the two layers operating on genuinely different mechanistic objects: instantaneous market state (T/V) vs crowd lifecycle position (Reactive-JPY).

------

*Document version: 2026-07-26* *Next review: After Stage 5.2 (episode extractor sensitivity) completion*
