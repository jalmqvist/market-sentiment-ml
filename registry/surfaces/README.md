# Behavioral Surface Registry

## Purpose

The Behavioral Surface Registry is the project's durable scientific memory.

Each file in this directory is the authoritative scientific record for one
Behavioral Surface, accumulating evidence across many experiments.

Individual experiment reports answer:

> **What happened during this experiment?**

The Registry answers:

> **What do we currently believe about this Behavioral Surface, based on all
> accumulated evidence?**

---

## Registry Lifecycle

Each Behavioral Surface progresses independently through the research
programme.

```
Characterization
      ↓
Predictive Validation
      ↓
Trading Validation
      ↓
Integrated
```

or

```
Characterization
      ↓
Predictive Validation
      ↓
Retired
```

Lifecycle stage records research progress rather than scientific quality.

A Behavioral Surface that ultimately fails predictive validation is retained.
Its registry entry transitions to `Retired` together with the supporting
evidence.  Negative results remain durable scientific outputs.

---

## Promotion Workflow

Promotion is an **explicit manual step**.

Completed experiment reports do **not** automatically update the registry.

To promote an experiment into the registry:

```bash
python analysis/registry/promote.py \
  --surface reactive_jpy \
  --experiments analysis/output/exp_2026_01_01 \
  --author "your.name" \
  --recommendation "Repeat characterization with additional training." \
  --scientific-interest medium \
  --scientific-confidence low \
  --notes "Initial characterization.  Entropy high across all states."
```

Promotion appends supporting evidence, updates the current scientific
assessment, records the author, and appends a timestamped entry to the
promotion history.

See `analysis/registry/promote.py --help` for full usage.

---

## Registry Summary

To generate a concise human-readable summary of all registered surfaces:

```bash
python analysis/registry/high_score.py
```

The summary focuses on scientific status rather than numeric ranking.

---

## Registry Schema

Each `<surface_id>.yaml` file follows this schema:

```yaml
surface_id: <str>
  Unique identifier matching the surface_id used in BSVE artifacts.

ontology_version: <str>
  Current ontology version (e.g. "v1").

lifecycle_stage: <str>
  One of: Characterization, Predictive Validation, Trading Validation,
          Integrated, Retired.

current_status: <str>
  One of: active, retired, suspended.

scientific_interest: <str>
  One of: low, medium, high.
  Human-authored assessment of how important or potentially novel findings
  would be if confirmed.  Never computed automatically.

scientific_confidence: <str>
  One of: low, medium, high.
  Human-authored assessment of how strongly current evidence supports the
  accumulated findings.  Never computed automatically.

current_recommendation: <str>
  Current recommended next research step.  Examples:
    - Repeat characterization
    - Compare against previous Behavioral Surfaces
    - Proceed to walk-forward validation
    - Archive surface
    - Await additional evidence

supporting_experiments: <list>
  Promoted experiment directory references.  Each entry contains:
    experiment_dir: path to the experiment output directory
    promoted_at: ISO 8601 timestamp
    promoted_by: author identifier
    notes: human description of the evidence promoted

promotion_history: <list>
  Full history of promotions.  Each entry records:
    promoted_at: ISO 8601 timestamp
    promoted_by: author identifier
    lifecycle_stage: stage at time of promotion
    scientific_interest: interest at time of promotion
    scientific_confidence: confidence at time of promotion
    recommendation: recommendation at time of promotion
    experiments_added: list of experiment dirs added in this promotion
    notes: description

stage1:
  Stage 1 (Behavioral Characterization) evidence and summary.

stage2:
  Stage 2 (Predictive Validation) evidence and summary.
  Present even when empty (status: not_started).

stage3:
  Stage 3 (Trading Validation) evidence and summary.
  Present even when empty (status: not_started).
```

---

## Design Principles

### Scientific judgments, not measurements

The registry stores scientific judgments derived from accumulated evidence
rather than experimental measurements.

Experiment reports remain the authoritative source of measurements.

The registry records the current scientific interpretation of that evidence.

### Human interpretation remains authoritative

Scientific Interest and Scientific Confidence are always human-authored.

Each assessment records the author, timestamp, and supporting experiments.

These assessments are informed by experiment evidence but are never computed
automatically.

### One Behavioral Surface, one scientific record

Each Behavioral Surface owns exactly one registry entry.

Evidence accumulates.  Interpretation evolves.  History is preserved.

### The registry is an index

The registry references experiment reports.  It never duplicates measurements
or report content.  Reports remain responsible for experimental evidence.

The registry records the project's current working scientific interpretation.

---

## Distinction from Experiment Reports

| Aspect | Experiment Report | Registry Entry |
|--------|-------------------|----------------|
| Location | `analysis/output/<experiment_id>/report.md` | `registry/surfaces/<surface_id>.yaml` |
| Updated by | Experiment framework (automatic) | Promotion workflow (explicit manual) |
| Content | Raw metrics, diagnostics, individual findings | Accumulated scientific interpretation |
| Scope | Single experiment | All experiments for a surface |
| Answers | What happened in this experiment? | What do we currently believe? |
| Preserved | Per-experiment | Cross-experiment |
