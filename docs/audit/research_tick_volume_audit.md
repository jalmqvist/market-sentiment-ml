# Tick-volume usage and HST decoding impact

This audit covers tick-volume decoding, propagation, and downstream usage.

## Summary

`entry_tick_volume` is carried from the FX bar input into the master research
dataset and retained in datasets augmented with Behavioral Surface columns.
The checked-in `train.py` and `train_lstm.py` entry points use local feature lists
that omit it. The canonical feature registry and generic dataset-loader API can
include it, but the follow-up static audit found no executable checked-in callers
of either `get_features()` API. Likewise, the BSVE adapter exposes volume in its
default structural-column list, but no checked-in production caller was found
using that default list. The standard Behavioral Surface and inspected ABM paths
select other inputs and do not consume tick volume.

The legacy converter used to generate the existing CSV lineage interprets
the 8-byte tick-volume field in a 60-byte HST v401 record as a `double`. The
correct decoding is a signed little-endian 64-bit integer:
`struct.unpack("<q", r[40:48])[0]`. The supplied v2 converter applies this
volume correction, but it is not yet a fully validated reference converter:
its v401 header parser retains the legacy incorrect offsets, and its heuristic
record-size detection tries 44-byte records before 60-byte records rather than
selecting the layout from the HST version and validating the body length. Those
issues should be resolved before full conversion. The intended volume-only
change should leave OHLC and timestamp serialization unchanged. The converter
is external to this repository; this audit records the supplied code review
and expected downstream effects, not a completed corpus-level conversion test.

Scope: repository revision `c3ac5cb56a4e07c27add66376db4a68a26600f69`,
plus the supplied external converter. This is a static audit of checked-in
code, not an inventory of external callers, historical runs, custom plugins,
or generated datasets.

## Data path

1. The input schema describes source `tick_volume` as a broker/platform-specific
   tick-activity measure, not centralized FX volume
   ([`INPUT_SCHEMA.md`](../../INPUT_SCHEMA.md#L175-L179)).
2. The FX dataset builder requires the raw `tick_volume` column, parses it as
   numeric `volume`, then maps the matched entry bar's volume to
   `entry_tick_volume`
   ([source loading](../../scripts/build_fx_sentiment_dataset.py#L218-L247),
   [entry-bar mapping](../../scripts/build_fx_sentiment_dataset.py#L268-L329)).
3. The output schema identifies `entry_tick_volume` as the aligned entry-bar
   activity measure ([`OUTPUT_SCHEMA.md`](../../OUTPUT_SCHEMA.md#L187-L195)).
4. Behavioral augmentation left-joins only its behavioral columns onto the
   base dataset, preserving its rows and original fields. Thus it retains
   `entry_tick_volume`; it does not recalculate it
   ([join](../../bsve/dataset_augmentation.py#L227-L277)).

## Feature definitions, APIs, and observed consumers

The following table distinguishes executable consumers from fields or APIs that merely expose volume.

| Component / API | Finding / status |
|---|---|
| `research/deep_learning/train.py` | Its local `price_trend` and `trend_vol_only` feature lists omit `entry_tick_volume`; the selected numeric columns form the model input. |
| `research/deep_learning/train_lstm.py` | Its local feature lists likewise omit the volume column; sequences are built from those selected columns. |
| `research/deep_learning/feature_sets.py` | Includes `entry_tick_volume` in `PRICE_FEATURES`; derived registry sets such as `price_only` and `price_sentiment` include it. |
| `research/deep_learning/dataset_loader.py` | `get_features()` uses the canonical registry and converts its feature matrix to `float32`. Consequently, with the current subnormal values, the volume feature will underflow to zero in this path; the corrected counts remain representable. |
| `bsve/adapters/dataset_adapter.py` | Includes `entry_tick_volume` in the default `structural_columns`. |
| `bsve/validation/outcome_labeling.py` | The inspected market-outcome path explicitly requests `entry_close` and `vol_48b`, not tick volume. |
| Behavioral Surface variants | Preserve the base dataset's volume column through the left join; no behavior-surface-specific volume transformation was found. |

The training scripts' local feature sets do not use the canonical registry.
Therefore merely selecting a behavioral dataset variant does not make tick
volume an input to those scripts. This is a finding about the checked-in code,
not proof that no external script, notebook, custom plugin, or historical run
used volume.

## Feature API and call-site inventory

- [`dataset_loader.get_features()`](../../research/deep_learning/dataset_loader.py#L119-L170)
  has no executable call sites in the checked-in repository. Its docstring
  and [`docs/data/DATASET.md`](../data/DATASET.md#L113-L129) show example
  calls with `price_sentiment`. It defaults to that set, rejects missing
  feature columns, drops rows missing any selected feature or target, and
  converts the result to `float32`.
- The distinct
  [`MasterResearchDatasetAdapter.get_features()`](../../bsve/adapters/dataset_adapter.py#L122-L131)
  accepts an arbitrary list of feature names and returns those columns with
  pair and timestamp. It has no checked-in call sites. It can expose volume
  if a caller explicitly requests it, but its existence does not imply volume
  is consumed by BSVE.
- The [registry](../../research/deep_learning/feature_sets.py#L30-L38) includes
  volume in `PRICE_FEATURES`. Its four derived sets containing volume are
  `price_only`, `price_sentiment`, `price_vol`, and `price_vol_sentiment`;
  the three `price_trend*` registry sets do not include it
  ([definitions](../../research/deep_learning/feature_sets.py#L58-L63),
  [mapping](../../research/deep_learning/feature_sets.py#L118-L125)).
  Aside from these definitions, `PRICE_FEATURES` appears in the
  [`lstm_dataset.py` usage example](../../research/deep_learning/lstm_dataset.py#L13-L21),
  not in an executable checked-in call. That module's generic sequence builder
  can include volume if a caller passes it.
- [`run_dl_experiment.sh`](../../scripts/run_dl_experiment.sh#L13-L31) and
  [`run_lstm_experiment.sh`](../../scripts/run_lstm_experiment.sh#L10-L25)
  pass `price_only` and `price_sentiment` to the two training scripts. Those
  names are **not** accepted by the scripts' local registries; if execution
  reaches feature-set validation, each invocation raises `ValueError` rather
  than training with the canonical sets. The scripts can return earlier for
  too few rows or fail for other prerequisites. These wrappers therefore are
  not evidence that tick volume was used in a successful run.

## Behavioral and agent-state trace

- `structural_columns` is read by
  [`get_structural_observations()`](../../bsve/adapters/dataset_adapter.py#L160-L171)
  only when no explicit column list is supplied; missing selected columns are
  silently omitted. The only checked-in production call passes an explicit
  list: `entry_close` and a threshold column, normally `vol_48b`
  ([outcome labeling](../../bsve/validation/outcome_labeling.py#L20-L22)).
  A test explicitly requests `atr_pct`
  ([test](../../tests/test_bsve_dataset_adapter.py#L42-L49)).
  No checked-in consumer calls it with the default list, so the presence of
  volume in that list alone does not establish a production use.
- **Configurable exception:** outcome labeling permits
  [`--threshold-column`](../../bsve/validation/outcome_labeling.py#L264-L270).
  Setting it to `entry_tick_volume` would feed the column into the
  [success threshold](../../bsve/validation/outcome_labeling.py#L90-L104)
  and affect outcome availability and labels. That is not the default and is
  dimensionally inappropriate: the CLI requires a threshold in return units,
  whereas tick volume is a count. No such checked-in invocation was found.
- The standard [surface generator](../../bsve/state_machine/rule_based.py#L62-L91)
  selects only pair/time, `net_sentiment`, and `crowd_side` before passing
  observations to the engine. Reactive-JPY consensus and classification use
  sentiment and maturity
  ([plugin](../../bsve/state_machine/plugins/reactive_jpy.py#L31-L38),
  [classifier](../../bsve/state_machine/plugins/reactive_jpy.py#L71-L96));
  Persistent uses sentiment depth, crowd side, prior history and gap state
  ([plugin](../../bsve/state_machine/plugins/persistent.py#L121-L170)).
  Their checked-in calibration adapters likewise request sentiment columns,
  not volume
  ([Reactive-JPY](../../bsve/calibration/jpy_maturity_calibration.py#L748-L758),
  [Persistent](../../bsve/calibration/persistent_calibration.py#L138-L152)).
  The engine assigns episode boundaries and maturity from those observations
  and plugin results ([engine](../../bsve/state_machine/engine.py#L99-L170)).
  No direct tick-volume input to these default classifications, episode
  construction or surface calculations was found.
- In the checked-in ABM path,
  [`run_abm.py`](../../research/abm/run_abm.py#L185-L198) selects entry close,
  entry time and real sentiment; the
  [simulation](../../research/abm/simulation.py#L150-L167) updates agents
  using price history, aggregate sentiment and price-derived volatility.
  No direct entry-volume input to agent state or the inspected episode
  filtering paths was found.

The generic engine's `process_observation()` passes an observation dictionary
to plugins, but the standard `generate_behavioral_surface()` function first
[selects only four input columns](../../bsve/state_machine/engine.py#L200-L233).
Direct engine callers, custom plugins, adapter methods and other configurable
APIs could use volume; the negative findings above concern checked-in default
paths, not all possible configurations.

## Dataset-builder operations

The canonical FX builder requires the volume column, coerces it to numeric,
and carries it through the entry-bar join and output. It does not use its
value in matching, coverage-based universe selection, derived features,
targets, or standard row filtering; missing volume alone does not discard an
input bar or master row
([input cleaning](../../scripts/build_fx_sentiment_dataset.py#L223-L245),
[valid rows and universes](../../scripts/build_fx_sentiment_dataset.py#L688-L704),
[core/extended split](../../scripts/build_fx_sentiment_dataset.py#L804-L806)).
The standard `build_dataset.py` calls the volatility helper, which calculates
price volatility from `entry_close`, not tick volume
([helper](../../scripts/build_dataset_vol.py#L39-L98)).

**Exception:** running `scripts/build_dataset_vol.py` as a standalone
builder overwrites its core dataset using `master.dropna()` across *all*
columns ([code](../../scripts/build_dataset_vol.py#L124-L144)). Missing
`entry_tick_volume` can therefore remove rows from that core output. The
normal `build_dataset.py` path imports only the volatility helper and does
not execute that standalone filtering. Similarly, registry-based
`get_features()` can drop rows whose selected volume is missing before its
`float32` conversion.

These are missingness/row-selection dependencies, distinct from dependence on
the numerical tick-volume value. A volume-only decoder correction is not
expected to change row retention if the legacy and corrected versions have the
same missing-value mask, but that equivalence must be measured on the full
converted corpus rather than assumed. Record and compare per-pair missingness
masks and retained row identifiers for any historical path that used either
filtering operation.

## Estimated effect of correcting the decoder

The five supplied sample records change from values around `4.16e-321` to
`1.08e-320` to integer counts from 842 to 2,190. The corrected sample counts
have a mean of 1,382.6. These five rows illustrate the scale and type of change;
they are not enough to estimate corpus-wide volume distributions, missingness,
or model-performance deltas.

| Area | Expected impact |
|---|---|
| Raw CSV and master dataset | `tick_volume` / `entry_tick_volume` becomes a usable count instead of a subnormal float. In the shown data path, the OHLC and timestamp fields are unaffected by this decoding change. |
| `train.py` and `train_lstm.py` | No direct change to their feature matrices or metrics while their current feature lists continue to exclude volume. |
| Canonical registry-based feature extraction | Material change: `get_features()` currently casts inputs to `float32`, where the supplied subnormal values become zero. Correct counts survive that conversion and provide a nonconstant feature. |
| Models or analyses that consume volume without normalization | Potentially material input and prediction changes; the field changes from effectively zero-valued to counts in the thousands in the example. The exact model effect cannot be inferred without identifying the experiment and rerunning it. |
| BSVE structural access and behavioral variants | The adapter can expose the corrected structural value, and augmented dataset files will carry it forward. Default outcome labeling does not use it; an explicitly configured threshold column can. |

This is not simply a harmless unit rescaling for all consumers: the current
feature-loader path casts to `float32`, under which the supplied subnormal
values collapse to zero. A downstream normalization performed only after that
cast cannot recover their variation. Conversely, consumers that read the CSV
as `float64` and standardize before narrowing may preserve the relative values
of these particular subnormal encodings; results depend on the exact
preprocessing path.

## Rebuild and validation implications

Fixing the external converter does not update existing CSVs or datasets. To
realize the corrected feature, re-convert the source `.hst` files, then rebuild
the affected master research datasets and rerun Behavioral Surface augmentation
for any variants that need to stay aligned with the rebuilt base data. Existing
trained models and exported prediction artifacts are unchanged until their
training data is rebuilt and the experiments are rerun.

A corpus-level impact estimate or model-performance claim requires the full
reconverted inputs and controlled before/after runs with identical dataset
versions, variants, feature sets, splits, and seeds. In particular, correcting
the decoder alone should have no direct metric effect on the current
`train.py`/`train_lstm.py` paths unless their feature selections are changed.

## Historical research impact and provisional migration assessment

The following decisions are supported by the checked-in code audit. They are
provisional with respect to historical execution because this static audit does
not inspect every run manifest, external launcher, notebook, custom plugin, or
historical generated input.

| Research/component | Provisional assessment | Migration treatment |
|---|---|---|
| Standard Reactive-JPY and Persistent surface generation | Tick volume is not selected by the standard surface-generation path or the inspected plugins and calibration adapters. | Do not reconstruct or recalibrate surfaces solely because the base dataset's volume values are corrected. |
| Behavioral Surface augmentation | The augmentation left-joins behavioral columns and preserves the base dataset's other columns; it does not recalculate the volume field. | Regenerate augmented dataset exports when needed to align them with the corrected base dataset; do not rerun surface discovery absent a demonstrated upstream dependency. |
| Inspected ABM path | The checked-in path selects entry close/time and sentiment; agent updates use price history, aggregate sentiment, and price-derived volatility. | No ABM rerun is indicated by this volume audit alone. Verify the actual historical configuration if it differs from the checked-in path. |
| `train.py` and `train_lstm.py` | Their checked-in local feature lists omit `entry_tick_volume`. | No volume-driven retraining is indicated if the historical run used these same code paths and feature selections. Confirm run manifests/configurations before declaring individual historical runs unaffected. |
| Canonical registry-based DL extraction | The registry includes volume in `price_only`, `price_sentiment`, `price_vol`, and `price_vol_sentiment`; the generic loader casts to `float32`, where the legacy subnormal values become zero. No executable checked-in caller was found. | Treat as a latent capability, not an established historical dependency. Investigate any external or historical caller before deciding on reruns. Do not silently add volume to existing model specifications. |
| Standalone `build_dataset_vol.py` | Its all-column `dropna()` can make row retention depend on volume missingness. | Check whether this standalone path was used historically and compare missingness masks and retained row IDs. |
| Default BSVE outcome labeling | The inspected production call requests `entry_close` and the configured threshold column, normally `vol_48b`; tick volume is not the default. | No rerun is indicated for the default configuration. Check historical CLI arguments for the configurable threshold-column exception. |
| MPML | Not covered by this MSML/BSVE audit. | Keep MPML impact unresolved until its own D1-to-strategy dependency audit is complete. |

### Provisional conclusions

1. The standard checked-in Behavioral Surface generation, calibration, and
   inspected ABM paths are independent of tick volume. Existing lifecycle
   findings should not be reopened solely because the source volume field is
   corrected.
2. The main checked-in MLP/LSTM training scripts exclude tick volume from their
   local feature lists. This supports retaining results from runs that used
   those exact paths and configurations, subject to input/provenance checks.
3. The canonical feature registry and generic loader can consume volume, but
   no executable checked-in caller was found. The experiment wrapper scripts
   pass registry names that the training scripts' local registries do not
   accept; they are therefore not evidence of successful historical runs using
   those registry feature sets.
4. Data artifacts that carry the field should be rebuilt into a separate,
   versioned corrected lineage. Rebuilding an export is not the same as
   reconstructing a behavioral surface, retraining a model, or rerunning a
   benchmark.
5. No conclusion about MPML can be drawn from this audit.

## Audit boundaries and evidence required for final sign-off

This audit describes checked-in code at repository revision
`c3ac5cb56a4e07c27add66376db4a68a26600f69`. It does not establish what code was
used by every historical experiment or whether external callers consumed the
exposed APIs. In particular, absence of a checked-in call site is not proof of
absence of external use.

Before final migration sign-off:

- Inspect historical run manifests, command lines, and saved feature lists for
  the DL and standalone volatility-builder paths.
- Compare legacy and corrected volume missingness masks, row identifiers, and
  all non-volume dataset columns.
- Verify that the corrected conversion changes only the intended volume field
  (and separately validate the converter's v401 header parsing and record-size
  selection).
- Compare actual feature matrices, targets, splits, and upstream artifact
  fingerprints for any historical experiment whose dependency remains uncertain.
- Complete a separate MPML audit covering D1 ingestion, indicators, phase
  detection, strategy selection, execution, and portfolio calculations.
