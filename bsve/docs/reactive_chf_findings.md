> **Post-leakage verification (Dataset 1.6.1).** Following correction of the volatility leakage identified in Dataset 1.5.1, Studies -1 through 0D were repeated using Dataset 1.6.1. Although numerical volatility thresholds changed substantially due to the corrected volatility calculation, the principal qualitative findings were reproduced. In particular, the inverse relationship between volatility and persistence, the concentration of ultra-persistent episodes within low-volatility environments, and the absence of a strong direct volatility–outcome relationship all remained present. The exploratory conclusions therefore appear robust to the leakage correction.

---

### CHF Null Hypothesis (Pre-Study)

**H0:** Volatility context does not predict crowd-state persistence duration or crowd-failure probability in CHF pairs within the 2019–2026 sentiment-data window.

Studies 0A–0C are designed to test whether volatility contains information about persistence behavior and subsequent crowd outcomes. The Volatility-Conditioned Persistence ontology is therefore treated as a hypothesis rather than an assumption.

If H0 cannot be rejected after Studies 0A–0C, the Volatility-Conditioned Persistence ontology should be abandoned. In that case, either an alternative CHF behavioral hypothesis should be formulated or the CHF environment should be classified as having no stable behavioral ontology within the available dataset.

The purpose of these studies is hypothesis testing rather than ontology construction. Ontology development will proceed only if exploratory evidence demonstrates that volatility context is behaviorally informative.

---

## Phase 0: Exploratory Ontology Discovery

The Reactive-CHF program begins with exploratory studies rather than ontology construction.

This differs from the Reactive-JPY sequence. In Reactive-JPY, the maturity ontology was calibrated first and outcome discovery followed. The outcome studies subsequently changed the interpretation of the states and identified crowd-failure probability as the primary behavioral outcome family.

The CHF hypothesis is structurally different. Rather than focusing on episode lifecycle, it proposes that volatility context may influence how crowd states form, persist, and eventually fail. Before constructing a volatility-conditioned ontology, it is therefore necessary to establish whether volatility contains meaningful behavioral information at all.

The purpose of Studies 0A–0C is to determine whether volatility provides meaningful behavioral organization within CHF markets. In particular, the studies evaluate whether volatility acts as (i) a primary behavioral variable, (ii) a conditioning variable governing persistence dynamics, (iii) a proxy for another underlying mechanism, or (iv) a variable with little independent behavioral relevance. The objective is therefore not to establish a volatility ontology a priori, but to determine how volatility participates in the observed behavioral organization.

Importantly, these studies are exploratory but falsifiable. The objective is not to confirm the existence of a CHF ontology. The objective is to determine whether the data supports one.

------

## Study -1: Volatility Measure Discovery (Completed)

### Purpose

Before evaluating volatility-conditioned persistence, it was necessary to determine whether volatility itself exhibited any meaningful behavioral relationship to the persistence features that originally caused CHF pairs to be classified as reactive.

This study therefore preceded Studies 0A–0C and served as an exploratory assessment of volatility representations and their relationship to crowd persistence.

### Data

The study used the 2019–2026 CHF subset of the master research dataset.

Pairs examined:

- EURCHF
- USDCHF

Volatility measures examined:

- vol_12b (12-bar rolling return volatility)
- vol_48b (48-bar rolling return volatility)

At this stage, volatility was treated as a continuous variable rather than a regime variable.

### Study -1A: Volatility Distribution Analysis

The first objective was to determine whether CHF volatility naturally separated into distinct regimes.

Results:

- Both volatility measures exhibited largely unimodal distributions.
- No clear evidence of naturally occurring low-, medium-, and high-volatility clusters was observed.
- The distributions were right-skewed, with elevated-volatility tail events but no obvious multimodal structure.
- EURCHF and USDCHF displayed similar distributional shapes, although USDCHF exhibited consistently higher volatility levels.

Interpretation:

The results did not support the existence of obvious volatility regimes based solely on distributional structure.

However, a unimodal distribution does not imply behavioral irrelevance. Volatility may still contain predictive information even if natural regime boundaries are not visually apparent.

![volatility_distributions](/home/almqvist/Documents/PycharmProjects/market-sentiment-ml/bsve/calibration_artifacts/plots/volatility_distributions.png)

> **Figure -1A. CHF volatility distributions (2019–2026).** Distribution of rolling volatility measures used during the CHF exploratory studies. Both the 12-bar (`vol_12b`) and 48-bar (`vol_48b`) measures exhibit largely unimodal, right-skewed distributions. Elevated-volatility observations occur as a continuous tail rather than as clearly separated clusters. This suggests that volatility environments in CHF are not naturally partitioned into discrete low-, medium-, and high-volatility regimes based on distributional structure alone. Notice the volatility range difference between the two plots.

---

**Figure -1A** summarizes the distributional structure of the volatility measures examined during Study -1. Contrary to the original intuition behind a volatility-conditioned ontology, neither measure exhibits obvious multimodal behavior. Instead, volatility appears to vary along a largely continuous spectrum with an extended high-volatility tail. This result weakens the argument for defining CHF states directly from volatility buckets alone. At the same time, the absence of discrete volatility clusters does not imply behavioral irrelevance. Subsequent analyses demonstrated that persistence behavior varies substantially across this continuous volatility spectrum, suggesting that volatility may still influence crowd dynamics even in the absence of naturally separated volatility regimes.

---

### Study -1B: Volatility vs Persistence

The second objective was to determine whether volatility was related to the persistence features that originally motivated the CHF classification.

Persistence variables examined:

- side_streak
- extreme_streak_70
- extreme_streak_80

Results:

- A strong inverse relationship was observed between volatility and persistence.
- The relationship was strongest for vol_48b.
- Low-volatility observations exhibited substantially longer side streaks than high-volatility observations.
- The effect survived pair decomposition and was observed independently in both EURCHF and USDCHF.
- The relationship appeared nonlinear and threshold-like rather than purely linear.

The most stable signal was observed in side_streak. Extreme-state persistence displayed the same broad direction but with greater noise.
![eurchf_vol48b_sidestreak_median_extremestreak70_vol48b_percentile_bin](/home/almqvist/Documents/PycharmProjects/market-sentiment-ml/bsve/calibration_artifacts/plots/eurchf_vol48b_sidestreak_median_extremestreak70_vol48b_percentile_bin.png)

> **Figure -1B. Volatility versus persistence in EURCHF.**
>
> Scatterplot showing the relationship between 48-bar rolling volatility (`vol_48b`) and crowd persistence (`side_streak`) in EURCHF observations from 2019–2026. Each point represents a sentiment observation. The x-axis measures how long the crowd has maintained the same directional positioning, while the y-axis measures recent realized market volatility.
>
> The relationship is strongly nonlinear. High-volatility environments are associated exclusively with short-lived crowd states, whereas very persistent crowd states occur only within a narrow low-volatility corridor. Importantly, low volatility does not guarantee persistence; many low-volatility observations still exhibit short streak lengths. This suggests that low volatility may be a necessary but not sufficient condition for extended crowd persistence.
>
> Several visually distinct clusters correspond to long-lived crowd episodes that remained active within relatively stable volatility environments. These observations provide preliminary support for the Volatility-Conditioned Persistence hypothesis and motivated the subsequent CHF ontology studies.

---

**Figure -1B** provides a more direct view of the volatility–persistence relationship than percentile-bin analysis. Rather than imposing volatility buckets, the scatterplot shows the raw relationship between volatility and crowd-state duration. The absence of highly persistent crowd states at elevated volatility levels suggests that volatility acts as a constraint on persistence. The figure therefore supports the interpretation that volatility may influence the conditions under which long-lived CHF crowd states can emerge.

---

### **USDCHF Mid-Volatility Persistence Anomaly**

Unlike EURCHF, which exhibits a broadly monotonic decline in persistence as volatility increases, USDCHF displays a secondary persistence peak at intermediate volatility levels (approximately volatility percentile bins 7–8 in the vol_48b analysis).

This feature was not anticipated by the original CHF hypothesis and remains unexplained at this stage. Several interpretations remain possible:

- The USDCHF volatility–persistence relationship may differ structurally from EURCHF, reflecting the dual influence of CHF safe-haven dynamics and USD macro cycles on USDCHF positioning.
- USD-driven trend environments may sustain crowd states at intermediate volatility levels through a different mechanism than the low-volatility equilibrium persistence observed in EURCHF.
- The feature may reflect concentration of a small number of unusually long-lived episodes at a specific volatility level.
- The feature may be a statistical artifact that disappears under alternative volatility definitions or finer bin resolution.

At present there is insufficient evidence to distinguish between these explanations. The anomaly therefore remains an open question rather than a confirmed finding.

If the effect survives into Studies 0B and 0C -- particularly if it appears in outcome-conditioned analyses -- it will be treated as evidence that EURCHF and USDCHF do not share a single volatility-conditioned persistence ontology and may require separate ontological treatment. This would represent a significant structural finding for the CHF research program.

---

### Preliminary Interpretation

The findings suggest that volatility context is behaviorally relevant in CHF pairs.

Importantly, the volatility distributions themselves appeared largely continuous, while persistence behavior displayed threshold-like structure. This implies that regime-like behavior may emerge from the relationship between volatility and persistence rather than from discrete volatility clusters.

At present, the strongest candidate explanatory variable is vol_48b.

This does not establish a CHF ontology. However, it provides preliminary evidence that the Volatility-Conditioned Persistence hypothesis survives initial exploratory testing and warrants continuation into Studies 0A–0C.

---

### Volatility Measure Selection

Study -1 evaluated two volatility representations: **vol_12b** and **vol_48b**.

Both measures produced broadly consistent results, but the relationship between volatility and persistence was substantially clearer for **vol_48b**. Across pair-level analyses, volatility-binned persistence statistics, and volatility-versus-persistence scatterplots, the vol_48b measure consistently produced:

- Stronger persistence relationships.
- Cleaner threshold behavior.
- Reduced noise.
- More interpretable behavioral structure.

By comparison, vol_12b exhibited greater short-term variability and weaker persistence signals, making interpretation less stable.

Accordingly, **vol_48b will be treated as the primary volatility measure for all subsequent CHF studies.** The vol_12b measure will be retained as a robustness check and sensitivity analysis but will not be used as the primary ontology variable unless later studies provide evidence that it contains additional behavioral information.

This decision reflects empirical performance rather than theoretical preference and may be revisited if future analyses identify superior volatility representations.

---

### Methodological Note

An exploratory attempt was made to introduce ATR-based and Kaufman-style volatility measures.

This effort revealed that the master sentiment dataset consists of irregularly sampled sentiment observations rather than continuous hourly price bars. ATR-derived measures therefore cannot be computed directly within the sentiment dataset and require reconstruction from the underlying continuous OHLC price series.

This limitation does not affect the validity of the existing vol_12b and vol_48b features, which were generated during dataset construction.

---

## Study 0A: Behavioral Partition Discovery (Completed)

### Question

Can the continuous CHF volatility spectrum be partitioned into behaviorally distinct regions?

### Result

Yes.

Although CHF volatility distributions are largely unimodal, persistence behavior varies strongly across the volatility spectrum.

Behavioral structure emerges from the volatility–persistence relationship rather than from discrete volatility clusters.

Persistence duration is concentrated in low-volatility environments and declines substantially as volatility increases.

This finding rejects the idea that volatility must exhibit natural statistical clustering in order to produce meaningful behavioral partitions.

---

### Duration Percentiles Across Volatility Regimes

Episode durations were evaluated across 50 volatility percentile bins.

Median persistence remained short across most volatility environments.

However, upper-tail persistence exhibited dramatic volatility dependence.

The strongest effect appeared in the P95–P99 duration statistics.

Several low-volatility regions produced extremely long episodes exceeding 600 bars.

Equivalent episodes were largely absent in elevated-volatility environments.

This indicates that volatility primarily influences the persistence tail rather than the typical episode.

![fig1](/home/almqvist/Documents/PycharmProjects/market-sentiment-ml/analysis.chf/study_0a_v2/fig1.png)

---

### Volatility-Persistence Structure

Hexbin analysis confirmed a strong inverse relationship between volatility and persistence.

The persistence envelope contracts rapidly as volatility increases.

High-volatility environments almost never produce long-duration episodes.

By contrast, low-volatility environments produce both short and extremely long persistence episodes.

The relationship is therefore asymmetric:

- low volatility permits persistence
- high volatility suppresses persistence

Low volatility appears necessary but not sufficient for ultra-persistent crowd behavior.

![fig2](/home/almqvist/Documents/PycharmProjects/market-sentiment-ml/analysis.chf/study_0a_v2/fig2.png)

------

## Study 0B: Volatility vs Persistence (Completed)

### Question

Does volatility context contain information about crowd-state persistence duration?

### Result

Strong evidence supports the hypothesis.

Persistence duration varies systematically with volatility context in both EURCHF and USDCHF.

Multiple independent analyses produced the same qualitative result:

- duration percentile analysis
- hexbin analysis
- survival analysis
- tail-distribution analysis

All indicate that persistence is strongly conditioned on volatility regime.

------

### Survival Analysis by Volatility Quintile

Survival curves were estimated separately within volatility quintiles.

Across both EURCHF and USDCHF:

- low-volatility episodes survive substantially longer
- high-volatility episodes decay rapidly
- ultra-long persistence becomes increasingly rare as volatility rises

The result is visible in every volatility quintile and is consistent across both CHF pairs.

The analysis demonstrates that volatility contains information not only about expected duration but about the entire persistence distribution.

![fig3](/home/almqvist/Documents/PycharmProjects/market-sentiment-ml/analysis.chf/study_0a_v2/fig3.png)

---

### Persistence Tail Analysis

Complementary CCDF analysis revealed that the longest persistence episodes are concentrated almost entirely within the lowest volatility quintile.

Episodes exceeding 100 bars occur predominantly in the lowest-volatility environment.

Higher-volatility regimes exhibit much thinner persistence tails.

This suggests that volatility primarily controls the probability of extreme persistence rather than median persistence alone.

![fig4](/home/almqvist/Documents/PycharmProjects/market-sentiment-ml/analysis.chf/study_0a_v2/fig4.png)

---

## Study 0A/0B Results Summary (Completed)

Studies 0A and 0B were conducted jointly because the same episode-level analyses simultaneously addressed both partition discovery and persistence-mechanism validation.

The analyses used a newly constructed episode dataset derived from sentiment-state persistence episodes across EURCHF and USDCHF between 2019 and 2026.

A total of 658 persistence episodes were identified:

- EURCHF: 417 episodes
- USDCHF: 241 episodes

Episode duration was defined as the number of consecutive observations during which crowd direction remained unchanged.

Volatility context was measured using vol_48b evaluated at episode initiation.

---

### Data Coverage Audit (Post-Hoc)

Following completion of Studies 0A and 0B, a coverage audit of the underlying sentiment archive identified two substantial interruptions in data collection:

- 2024-08-23 to 2024-10-03 (41 days)
- 2024-10-31 to 2025-05-09 (190 days)

The outages originated in the sentiment collection process rather than in episode construction or master-dataset generation.

A subsequent audit confirmed that the dataset-construction pipeline does not forward-fill sentiment states across missing periods. Missing sentiment snapshots therefore do not create synthetic persistence episodes. Instead, the outages reduce observational coverage during the affected intervals.

The outages nevertheless overlap a period during which sentiment distributions appear to shift substantially. Following the 2024–2025 outage, CHF-cross sentiment becomes overwhelmingly crowd-long, while several JPY control pairs become overwhelmingly crowd-short. Because the transition period itself is largely unobserved, it is not currently possible to determine whether this reflects:

- a genuine change in crowd positioning,
- a change in the sentiment provider's methodology,
- a change in source population composition, or
- some combination of these factors.

Accordingly, persistence findings that depend heavily on the post-2024 sample should be interpreted with appropriate caution until additional historical sentiment data becomes available.

---

## Study 0C: Volatility vs Crowd Failure (Completed)

### Question

Does volatility context predict crowd-failure probability?

### Result

No meaningful evidence was found that volatility regime directly predicts crowd-failure outcomes in CHF pairs.

Failure rates remained broadly stable across volatility quintiles at both 24-bar and 48-bar evaluation horizons. Direct comparisons between the lowest-volatility (Q1) and highest-volatility (Q5) environments produced only small differences and no statistically significant effects.

24-bar horizon:

- Q1 failure rate: 50.2%
- Q5 failure rate: 51.6%
- Fisher p-value: 0.55

48-bar horizon:

- Q1 failure rate: 49.1%
- Q5 failure rate: 51.6%
- Fisher p-value: 0.27

These results provide no support for a direct volatility → crowd-failure mechanism.

### Persistence-Based Outcome Discovery

Although volatility did not predict crowd failure directly, a previously unanticipated relationship emerged when outcomes were conditioned on episode duration.

Episodes were grouped by persistence duration:

| Duration Bucket | Failure Rate (24b) |
| --------------- | ------------------ |
| 1–2 bars        | 50.1%              |
| 3–5 bars        | 37.4%              |
| 6–10 bars       | 48.4%              |
| 11–20 bars      | 50.0%              |
| 21–50 bars      | 61.9%              |

The strongest effect occurred in the 3–5 bar duration bucket.

Episodes lasting approximately 3–5 bars exhibited substantially lower crowd-failure rates than very short 1–2 bar episodes.

### Statistical Assessment

A direct comparison was performed between:

- Very Short episodes (1–2 bars)
- Short episodes (3–5 bars)

Results:

- 1–2 bar failure rate: 50.1%
- 3–5 bar failure rate: 37.4%
- Difference: −12.8 percentage points
- Relative risk: 0.75
- Fisher p-value: 0.036
- Chi-square p-value: 0.037

The effect therefore survives simple contingency-table testing.

### Pair Robustness

The same directional relationship was observed independently in both CHF pairs.

24-bar horizon:

| Pair   | 1–2 bars | 3–5 bars |
| ------ | -------- | -------- |
| EURCHF | 51.1%    | 35.6%    |
| USDCHF | 48.1%    | 40.6%    |

48-bar horizon:

| Pair   | 1–2 bars | 3–5 bars |
| ------ | -------- | -------- |
| EURCHF | 51.5%    | 37.3%    |
| USDCHF | 47.3%    | 37.5%    |

Although sample sizes remain limited, both pairs exhibit the same qualitative pattern.

### Interpretation

Study 0C rejects the hypothesis that volatility directly predicts crowd-failure probability.

The evidence instead supports a mediation structure:

```
volatility
        ↓
  persistence
        ↓
   outcomes
```

> The post-leakage replication strengthened this interpretation. While the volatility–persistence relationship remained qualitatively unchanged, direct volatility–outcome relationships remained weak across repeated analyses. This suggests that volatility primarily provides behavioral context within which persistence develops, rather than acting as an independent determinant of crowd success or failure.

Volatility strongly influences persistence duration (Studies 0A–0B), but does not appear to exert an independent influence on crowd-failure rates.

The primary outcome discovery from Study 0C is therefore not a volatility effect but a persistence-duration effect.

Specifically, episodes that survive the initial 1–2 bar phase but terminate within approximately 3–5 bars exhibit unusually low crowd-failure rates relative to both shorter and longer-lived episodes.

At present this finding should be treated as an exploratory outcome family rather than as a confirmed ontological state. The CHF sample contains only two pairs and the result has not yet been observed outside the CHF environment.

------

## Current Status

### **Post-Leakage Replication (Dataset 1.6.1)**

> Following correction of the volatility leakage identified in Dataset 1.5.1, Studies -1 through 0D were repeated using Dataset 1.6.1. Although the corrected volatility calculation substantially changed the numerical volatility scale and regime calibration, the principal qualitative findings were reproduced. In particular:
>
> - the inverse volatility–persistence relationship remained present;
> - ultra-persistent episodes remained concentrated in low-volatility environments;
> - `vol_48b` continued to outperform `vol_12b`;
> - little evidence was found for a direct volatility–crowd-failure relationship;
> - persistence remained more strongly associated with behavioral outcomes than volatility itself.
>
> The exploratory interpretation of Reactive-CHF therefore appears robust to the leakage correction, although all volatility thresholds reported in earlier analyses should be regarded as superseded by Dataset 1.6.1.

Studies -1, 0A, 0B and 0C have now been completed.

The evidence consistently supports a strong empirical relationship between volatility context and persistence dynamics.

Key findings include:

- CHF volatility distributions are largely unimodal.
- Behavioral partitions nevertheless emerge through nonlinear persistence behavior.
- Low-volatility environments support substantially longer crowd-state persistence.
- Ultra-long persistence episodes are concentrated almost exclusively within the lowest volatility quintile.
- The effect is observed independently in both EURCHF and USDCHF.
- Multiple independent analyses support a strong volatility → persistence relationship.
- No meaningful evidence was found that volatility directly predicts crowd-failure probability.
- The strongest outcome-family discovery is a persistence-duration effect, in which episodes lasting approximately 3–5 bars exhibit substantially lower crowd-failure rates than very short 1–2 bar episodes. Follow-up temporal diagnostics showed that these 3–5 bar episodes occurred across multiple years and in both CHF pairs rather than being confined to a single historical interval, reducing concern that the effect reflects a localized historical artifact. Nevertheless, the finding remains exploratory pending replication on additional data.

Taken together, the results suggest that volatility primarily influences CHF behavior through persistence rather than through direct effects on crowd outcomes.

The original CHF null hypothesis can therefore be rejected in its strongest form. Volatility contains meaningful behavioral information, but that information appears to operate primarily through persistence dynamics rather than through direct outcome prediction.

The principal remaining questions concern ontology synthesis, external validation, and the relationship between CHF persistence behavior and the broader BSVE framework.

---

## Additional Caveat: Late-2024 Sentiment Discontinuity

A post-hoc audit of the underlying sentiment archive identified two substantial interruptions in data collection spanning late 2024 and early 2025. These outages occur near the end of the available observation window and overlap a period during which CHF-cross sentiment becomes almost exclusively crowd-long. Similar changes were observed in several JPY control pairs, suggesting that the phenomenon is not unique to CHF. 

Follow-up diagnostics (Study 0D) were conducted to determine whether the available data support the existence of a discrete CHF behavioral transition. The analyses did **not** provide convincing evidence for such a transition. Although the final negative-sentiment observations across most CHF pairs occur within a narrow interval in late October 2024, the transition itself falls largely within the period of missing data and is therefore not directly observed. The available evidence cannot distinguish between a genuine market-wide behavioral change, a change in the sentiment provider or source population, or other explanations. 

An important consequence is that the final persistence episode observed in each CHF pair remains active until the end of the dataset. From a statistical perspective these episodes should therefore be regarded as **right-censored** rather than completed persistence episodes. Their observed durations represent lower bounds rather than true completed durations.

Importantly, this limitation does **not** materially affect the principal conclusions of the CHF exploratory studies. The volatility–persistence relationship identified in Studies -1, 0A and 0B is present throughout the observed sample and was reproduced after correction of the volatility leakage in Dataset 1.6.1. Likewise, Study 0C continued to find little evidence for a direct volatility–crowd-failure relationship while identifying persistence duration as the more informative behavioral variable. The uncertainty introduced by the late-2024 discontinuity therefore concerns the interpretation of the final persistence episodes rather than the broader exploratory conclusions regarding volatility and persistence.

Accordingly, future work should avoid interpreting the prolonged post-2024 crowd-long state as evidence for a permanent CHF behavioral regime. Instead, it should be treated as an unresolved observational limitation until additional historical sentiment data or independent sentiment sources become available.

---

## Decision Logic

The purpose of Studies 0A–0C is not to prove that a volatility ontology exists.

The purpose is to determine whether one is justified.

Completed:

✓ Study -1
✓ Study 0A
✓ Study 0B
✓ Study 0C

---

## **Revised Working Interpretation**

The current evidence suggests that volatility primarily organizes the environments in which persistent crowd states emerge rather than defining behavioral states directly. Low-volatility environments consistently permit unusually long persistence episodes, while elevated-volatility environments suppress the upper tail of the persistence distribution. Crowd outcomes appear more closely related to persistence than to volatility itself, indicating that persistence may mediate much of the observable behavioral effect. Consequently, the present evidence favors interpreting volatility as contextual information governing persistence dynamics rather than as the primary ontological variable.

---

## Frozen Findings (June 2026)

The following findings are considered sufficiently stable to carry forward:

1. CHF volatility distributions are largely unimodal and do not naturally partition into discrete volatility regimes.
2. Despite the absence of natural clustering, volatility contains substantial information about crowd-state persistence.
3. Low-volatility environments consistently support substantially longer persistence tails than elevated-volatility environments.ates, while high-volatility environments strongly suppress long-duration persistence.
4. Ultra-long persistence episodes (>100 bars) occur almost exclusively within the lowest volatility environments.
5. No meaningful evidence was found that volatility directly predicts crowd-failure probability.
6. The strongest outcome-family discovery is a persistence-duration effect in which 3–5 bar episodes exhibit substantially lower crowd-failure rates than very short 1–2 bar episodes. 
7. Current evidence supports a volatility → persistence relationship more strongly than a volatility → crowd-failure relationship.

Current evidence consistently supports treating volatility as a persistence-conditioning variable rather than a direct outcome variable within any future CHF behavioral representation.

---

