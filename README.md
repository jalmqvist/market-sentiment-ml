# FX Retail Sentiment — Behavioral Signal Research

A quantitative research project studying whether retail FX sentiment contains predictive signal.

------

## Executive Summary

**Current status: no validated standalone or incremental alpha.**

Extensive experimentation and strict out-of-sample validation have produced a clear and well-supported result:

- Raw retail FX sentiment = noise
- Pipeline-based signals (V20–V21) were invalidated (pipeline artifacts)
- Price-based signals show stable predictive power (~0.14 Sharpe)
- Sentiment provides **no incremental value beyond price** under tested formulations

This is a meaningful and informative research outcome. The project now focuses on understanding structural relationships and exploring alternative modeling approaches.

------

## Key Findings

| Component               | Status                       |
| ----------------------- | ---------------------------- |
| Raw sentiment           | ❌ No edge (noise)            |
| Pipeline signals V20–21 | ❌ Invalid (artifact)         |
| Price-based signals     | ✅ Stable (~0.14 Sharpe)      |
| Additive sentiment      | ❌ Degrades performance (V28) |
| Conditional sentiment   | ❌ No effect (V29)            |

------

### Agent-Based Modeling — Behavioral Interpretation

We developed an agent-based model (ABM) to explain how retail FX sentiment
structure emerges from trader behavior.

The goal was not prediction, but explanation:

> What behavioral rules generate the sentiment we observe?

---

### Key Insight

Retail sentiment behaves as a **path-dependent accumulation process**.

A minimal model reproduces real-world sentiment when:

- traders build positions over time (accumulation)
- traders resist switching positions (inertia)
- traders reinforce positions when aligned with price (asymmetry)

This produces:

- persistent crowd imbalance
- clustered sentiment regimes
- realistic sentiment magnitude

Crucially, this occurs **without generating predictive signal**.

---

### New Finding: Regime Dependence

Multi-pair validation shows that this mechanism is **not universal**.

The model reproduces sentiment structure in:

- EUR, GBP, NZD pairs
- some AUD/NZD crosses

But fails in:

- JPY pairs
- CHF pairs
- some CAD pairs

---

### Interpretation

This indicates that retail sentiment depends on **market regime**:

- **Trend-dominated markets** → accumulation dynamics dominate → model works
- **Macro / flow-driven markets** → regime switching dominates → model fails

In other words:

> Retail sentiment is structurally different across currency regimes

---

### Implication

This refines earlier conclusions:

- sentiment has no global predictive power
- sentiment reflects **accumulated positioning**, not expectations

But:

> predictive signal may exist **conditionally**, within specific regimes

This is an open direction for further research.

---

### Status

- Single-pair validation: complete (EUR/USD)
- Multi-pair validation: complete (majors + crosses)
- Regime classification: identified
- Conditional predictability: open question

---

## Deep Learning Experiments (MLP)

A series of controlled deep learning experiments were conducted to test whether nonlinear models can extract predictive signal from sentiment.

### Setup

- Model: minimal 2-layer MLP
- Inputs:
  - price features (trend, returns)
  - sentiment features
  - volatility features (rolling std)
- Strict validation:
  - chronological split
  - no leakage (train-only normalization)
  - independent evaluation (shift / shuffle tests)

### Experiments

1. **price_only**
2. **price + sentiment**
3. **price + volatility**
4. **price + volatility + sentiment**

### Results

- price_only → small, stable signal (~0.01–0.02 Sharpe)
- price + sentiment → no meaningful improvement (noise increases)
- price + volatility → degrades performance
- price + volatility + sentiment → no recovery

### Conclusion

> Nonlinear models do not uncover hidden predictive structure in sentiment.

This reinforces earlier findings:

- sentiment is not predictive
- sentiment does not add incremental value
- sentiment does not become useful under volatility regimes

### Interpretation

Sentiment likely:

- reflects accumulated positioning driven by price dynamics
- reacts to market conditions
- does not lead price in a predictive way

### Next Step

→ Sequence models (LSTM) to test temporal dependencies.

---

### Deep Learning Experiments (LSTM)

To test whether temporal dependencies contain predictive signal, a sequence modeling approach was implemented.

#### Setup

- Model: minimal LSTM (single layer)
- Input: rolling sequences (length = 24)
- Strict validation:
  - per-pair sequence construction
  - global chronological ordering
  - train-only normalization
  - evaluation via baseline / shift / shuffle

#### Results

- price_only → no stable predictive signal
- price + sentiment → no improvement (slight degradation)
- no temporal structure recovered

#### Conclusion

> Sequence models do not reveal predictive information in sentiment or price under current formulation.

This is consistent with prior results:

- no standalone signal
- no additive signal
- no conditional signal
- no temporal signal

---

## Incremental Value Testing (V28–V29)

A focused experiment series was conducted to answer:

> **Does sentiment provide incremental predictive value beyond price?**

### V28 — Additive Model

- Signal: `price + β * sentiment`
- Result: **Destroys predictive signal (Sharpe → ~0)**

Interpretation:

> Sentiment introduces noise when combined linearly with price.

------

### V29 — Conditional / Regime Filter

- Signal: price remains dominant
- Sentiment only applied in extreme regimes

Result:

- **No measurable difference vs price-only signal**

Interpretation:

> Sentiment does not improve price signals, even in extreme conditions (under this formulation).

------

### Conclusion from V28–V29

- No standalone predictive power
- No additive contribution
- No simple conditional effect

> Sentiment is **informationally redundant with price** in this setting.

------

## Refined Interpretation

The experiments suggest:

- sentiment reflects or reacts to price behavior
- it does not lead price in a usable way
- any apparent edge disappears under causal validation

This reframes the problem:

> not “does sentiment predict returns?”, but
> “what structural role does sentiment play in the market?”

------

## Research Direction

Three parallel research tracks are now active:

### 1. Hypothesis Testing (statistical)

Systematic testing of falsifiable behavioral hypotheses. Most results are negative — this is expected and valuable.

------

### 2. Deep Learning (sequence modeling)

Exploring whether nonlinear sequence models (LSTM, Transformer) can extract higher-order structure not captured by linear methods.

------

### 3. Agent-Based Modeling (behavioral simulation)

Simulating retail crowd behavior to understand how sentiment forms and why it may lack predictive power.

------

## Repo Structure

```
market-sentiment-ml/
├── research/
│   ├── signal_discovery/        # V1–V29 (archived exploration)
│   ├── raw_validation/          # ground truth + diagnostics
│   ├── hypothesis_experiments/  # clean, modern experiments
│   ├── analysis/                # descriptive / exploratory analysis
│   ├── utils/                   # helper scripts
│   ├── deep_learning/
│   ├── abm/
├── docs/
│   ├── archive/
│   └── RESEARCH_STRATEGY.md
├── scripts/
├── pipeline/
├── evaluation/
├── utils/
└── tests/
```

------

## Philosophy

**Negative results are valuable.**

> Clearly ruling out entire classes of models significantly narrows the search space.

------

**Validation outside the pipeline is mandatory.**

> All hypotheses must first pass minimal, causal validation before further development.

------

**Causality over complexity.**

> Simpler models with strict causality are preferred over complex pipelines prone to artifacts.

------

## Signal Discovery Archive (V1–V29)

Early exploratory experiments are preserved in:

```
research/signal_discovery/
```

These experiments:

- explored a wide range of behavioral signal constructions
- revealed multiple failure modes (leakage, selection bias, instability)
- helped define the current validation standards

They should be interpreted as:

> a map of explored ideas — not validated signals

------

## Further reading

- `RESEARCH_STATE.md` — current experiment status
- `docs/RESEARCH_STRATEGY.md` — research philosophy
