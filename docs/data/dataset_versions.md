# Dataset Version History

## 1.5.1 — Entry-bar deduplication correction

Date:
2026-06-15

Changes:
- deduplicate rare `(pair, entry_time)` collisions created when multiple sentiment snapshots align to the same hourly entry bar
- retain the latest `snapshot_time` when such collisions occur
- add fail-fast validation to guarantee `(pair, entry_time)` uniqueness before export

Scientific impact:
- preserves the dataset contract expected by BSVE and other downstream consumers
- corrects rare alignment artifacts without changing feature definitions, merge semantics, or schemas

---

## 1.5.0 — Contract-layer generation

Date:
2026-05-21

Changes:
- formal DL artifact schema v2
- explicit timestamp semantics
- causal availability timestamps
- producer/consumer validation layer
- contract metadata standardization
- fail-fast validation semantics

Scientific impact:
- first fully contract-validated generation
- suitable for post-audit MPML experiments

---

## 1.4.0 — Causality-hardened generation

Changes:
- missingness semantics formalization
- sparse-observed-only semantics
- improved provenance
- availability controls

Known limitations:
- timestamp semantic ambiguity between
  prediction generation and prediction availability

---

## 1.3.x — Pre-audit generation

Characteristics:
- exploratory semantics
- implicit availability assumptions
- pre-contract architecture
