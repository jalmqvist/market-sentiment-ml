# Persistent Family Findings

## Overview

Persistent-family currency pairs exhibit a distinctive persistence structure characterized by many short-lived sentiment episodes together with a small population of exceptionally long-lived states. Although approximately half of all episodes terminate within two observations and roughly seventy percent terminate within five observations, the persistence distribution contains a pronounced heavy tail extending beyond 500 bars. The defining characteristic of the Persistent family is therefore not unusually long average persistence, but the repeated emergence of rare, highly persistent sentiment states.

Figures 1–5 examine these long-lived episodes from complementary perspectives by analysing regime transitions, persistence distributions and within-episode dynamics.

------

## Major findings

### 1. Long-lived episodes are associated with volatility transitions rather than static regimes

![04_behavioral_transition_classes](/home/almqvist/Documents/PycharmProjects/market-sentiment-ml/analysis.persistent/output/04_behavioral_transition_classes.png)

> **Figure 1. Behavioral transition classes.** Episodes are grouped according to whether trend and/or volatility differ between the first and last regime of the episode. The upper panel shows that nearly 80% of episodes begin and end in the same behavioral class. The lower panel reveals that these common episodes are *less* likely to become long-lived, whereas episodes involving volatility changes are enriched by approximately 3–4× among episodes lasting more than 100 bars. The figure introduces the central finding of this study: persistence is associated with **regime evolution** rather than remaining within a single market state.

The strongest behavioral distinction emerges when episodes are grouped according to how their start and end regimes differ (**Figure 1**).

Episodes that preserve both trend and volatility ("No change") dominate the dataset, accounting for nearly 80% of all observations. However, these episodes are underrepresented among the longest-lived persistence events. In contrast, episodes that preserve trend while changing volatility occur approximately three times more frequently among 100+ bar episodes than expected from their overall frequency. Episodes changing both trend and volatility are almost four times more common among extreme persistence events despite representing less than two percent of all episodes.

This indicates that persistence is associated with regime evolution rather than remaining within a single market state.

------

### 2. Transition families reveal where extreme persistence develops

![01_transition_summary](/home/almqvist/Documents/PycharmProjects/market-sentiment-ml/analysis.persistent/output/01_transition_summary.png)

> **Figure 2. Episode transition families.** The behavioral classes from Figure 1 are resolved into individual regime-to-regime transitions. While LVTF→LVTF and HVTF→HVTF dominate the overall population, the strongest enrichment among long-lived episodes occurs for transitions connecting low- and high-volatility trend-following regimes, particularly LVTF→HVTF. This identifies the specific trajectories responsible for the enrichment observed in Figure 1.

**Figure 2** resolves the behavioral classes into individual regime transitions.

Most episodes remain within the same regime, particularly LVTF→LVTF and HVTF→HVTF transitions. However, these dominant transition families do not explain the persistence tail.

Instead, the largest enrichment among extreme-duration episodes occurs for LVTF→HVTF transitions, which are more than four times as likely to appear among 100+ bar episodes than expected from their overall frequency. HVTF→LVTF transitions also exhibit substantial enrichment, while transitions involving volatility changes without altering trend consistently outperform expectation.

Taken together, these results indicate that persistence preferentially develops during sustained trend-following behavior accompanied by changing volatility conditions.

------

### 3. Regime evolution follows a highly structured state space

![02_internal_transition_matrix](/home/almqvist/Documents/PycharmProjects/market-sentiment-ml/analysis.persistent/output/02_internal_transition_matrix.png)

> **Figure 3. Within-episode regime transition probabilities.** Heatmap of transition probabilities between consecutive regime observations within persistent sentiment episodes. Trend-following regimes exhibit strong self-persistence (>85%), whereas range regimes preferentially transition into trend-following states instead of remaining in range. The figure demonstrates that persistent episodes evolve within a highly structured state space rather than through random regime switching.

**Figure 3** shows that within-episode transitions are far from random.

Both trend-following regimes exhibit strong self-persistence, with more than 85% of transitions remaining within the same regime. Range regimes are considerably less stable, frequently transitioning into trend-following environments rather than remaining unchanged.

The dominant off-diagonal transitions are:

- LVR → LVTF
- HVR → HVTF

whereas direct transitions between low- and high-volatility range regimes occur only rarely.

This suggests that trend formation represents the principal direction of regime evolution during persistent sentiment episodes.

------

### 4. Timing analysis shows that transitions occur throughout an episode

![03_transition_timing](/home/almqvist/Documents/PycharmProjects/market-sentiment-ml/analysis.persistent/output/03_transition_timing.png)

> **Figure 4. Timing of regime transitions within an episode.** Histograms showing where each transition type occurs during the normalized lifetime of an episode. Each transition family is normalized independently to facilitate comparison despite different sample sizes. Neither volatility nor trend transitions are confined to episode initiation or termination; instead, they occur throughout the episode, indicating continuous structural evolution rather than a sequence of discrete phases.

**Figure 4** examines when regime transitions occur during an episode.

Normalizing each transition family independently reveals that neither volatility changes nor trend changes are concentrated exclusively near episode initiation or termination. Instead, transitions occur throughout the episode lifecycle, with mild clustering around the middle of long episodes.

This suggests that persistent sentiment episodes evolve continuously rather than progressing through a sequence of clearly separated phases.

------

### 5. Persistence depends on regime evolution

![05_duration_by_behavior](/home/almqvist/Documents/PycharmProjects/market-sentiment-ml/analysis.persistent/output/05_duration_by_behavior.png)

> **Figure 5. Persistence distributions by behavioral transition class.** Distribution of episode durations grouped by their overall start/end transition class. Episodes involving volatility changes exhibit substantially longer durations than episodes with no behavioral change, while episodes changing both trend and volatility display the longest median persistence despite representing only a small fraction of all observations. Together with Figures 1–4, this supports the interpretation that long persistence is linked to the path an episode follows through regime space rather than to any individual regime.

**Figure 5** compares persistence across the four behavioral transition classes.

Episodes preserving both trend and volatility exhibit the shortest durations. Episodes involving volatility changes while preserving trend have substantially longer persistence distributions, while transitions changing both trend and volatility display the longest median durations despite being relatively uncommon.

These results reinforce the conclusion that persistence depends more strongly on the trajectory through regime space than on the market state at any individual point in time.

------

## Interpretation

Across all analyses a consistent picture emerges.

Persistent-family pairs spend most of their time in relatively short-lived sentiment episodes that terminate without substantial structural change. Extreme persistence arises from a much smaller subset of episodes whose market environment evolves while maintaining directional structure. Rather than remaining fixed within a single regime, these episodes progressively traverse the Volatility–Trend state space, most commonly through transitions connecting low- and high-volatility trend-following regimes.

The evidence therefore suggests that persistence should be viewed as an evolving process rather than as a property of any single regime classification.

------

## Next research directions

The current analysis identifies **how** persistent episodes evolve but does not yet explain **why** they evolve differently from ordinary episodes.

The next stage of the project should investigate the mechanisms underlying these trajectories. Promising directions include:

- comparing Persistent and Reactive families using the same transition framework;
- measuring regime occupancy as a continuous process rather than only analyzing transitions;
- constructing episode-level state machines to identify common persistence pathways;
- quantifying transition entropy and regime stability throughout an episode;
- examining whether specific macroeconomic or market events preferentially trigger volatility-preserving or volatility-changing trajectories.

These analyses will determine whether the transition dynamics identified here represent a unique behavioral signature of Persistent-family currency pairs or a more general property of sentiment evolution.