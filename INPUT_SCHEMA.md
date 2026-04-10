# Input Schema

This document describes the expected local input files for the FX retail sentiment research pipeline.

The repository does **not** include raw data. Users are expected to supply their own local input files in the formats described below.

---

## Directory structure

Expected local layout:

```text
data/
├── input/
│   ├── fx/
│   └── sentiment/
└── output/
```

- `data/input/fx/` contains hourly FX CSV files, one file per pair

- `data/input/sentiment/` contains sentiment snapshot CSV files, one file per scrape
- `data/output/` is used for generated research datasets and diagnostics

## 1. Sentiment input files

### Location

```
data/input/sentiment/
```

### File naming

Expected format:

```
YYYY_MM_DD_HHMM.csv
```

Examples:

```
2019_03_26_0842.csv
2021_11_05_1201.csv
2024_07_18_1800.csv
```

The filename timestamp is used as the **snapshot timestamp** for all rows in that file.

### Expected columns

Each sentiment CSV is expected to contain the following columns:

- `pair`
- `perc`
- `direction`
- `time`

Some files may also contain an extra unnamed index column, which the pipeline ignores.

### Example

```
"","pair","perc","direction","time"
"1","aud-sgd","94","long",2019-03-26 08:39:45
"2","eur-nok","92","short",2019-03-26 08:40:34
"3","aud-nzd","91","long",2019-03-26 08:39:42
```

### Column definitions

#### `pair`

FX pair identifier.

Expected normalized form:

- `eur-usd`
- `usd-jpy`
- `gbp-chf`

The pipeline normalizes common separators such as `/`, `_`, and `-`.

#### `perc`

Percentage of visible retail crowd on one side of the market.

Example:

- `72`

#### `direction`

Side of crowd positioning.

Expected values:

- `long`
- `short`

#### `time`

Row-level scrape timestamp from the source.

This is parsed and preserved, but the pipeline uses the **filename timestamp** as the canonical snapshot time for the file.

### Notes

- Each file is treated as one market snapshot
- All rows in the file inherit the same `snapshot_time`
- Snapshot timestamps may require timezone normalization before merge

## 2. FX hourly market data files

### Location

```
data/input/fx/
```

### File naming

One hourly CSV per pair.

Examples of supported styles:

```
EURUSD_H1.csv
USDJPY_H1.csv
eurusd.csv
GBPCHF60.csv
```

The pipeline extracts the 6-letter symbol from the filename and converts it to normalized pair format:

- `EURUSD` → `eur-usd`
- `USDJPY` → `usd-jpy`

### Expected columns

Each FX CSV is expected to contain:

- `time_utc`
- `open`
- `high`
- `low`
- `close`
- `tick_volume`

### Example

```
time_utc,open,high,low,close,tick_volume
2019-03-26 08:00:00,1.12820,1.12880,1.12790,1.12840,1532
2019-03-26 09:00:00,1.12840,1.12860,1.12750,1.12780,1458
2019-03-26 10:00:00,1.12780,1.12810,1.12710,1.12750,1499
```

### Column definitions

#### `time_utc`

Hourly bar timestamp.

Must be parseable as a datetime and sortable within each pair.

#### `open`, `high`, `low`, `close`

Hourly OHLC values.

Expected to be numeric.

#### `tick_volume`

Broker/platform-specific tick activity measure.

This is preserved by the pipeline as a proxy for market activity, but it should not be treated as centralized FX volume.

### Notes

- The pipeline expects one row per hourly bar
- Duplicate timestamps within a pair are dropped during load
- Files with incomplete or truncated history reduce usable coverage for that pair

## 3. Timezone expectations

The current research setup assumes:

- sentiment snapshot timestamps correspond to `UTC+2`
- FX hourly timestamps correspond to `UTC+1`

The pipeline shifts sentiment snapshot timestamps by **-1 hour** before merging.

This assumption should be validated whenever new source data is introduced.

## 4. Pair normalization

The pipeline normalizes pairs to lowercase with `-` separator:

- `EURUSD` → `eur-usd`
- `eur/usd` → `eur-usd`
- `eur_usd` → `eur-usd`

Consistent pair naming across sentiment and price inputs is required for successful matching.

## 5. Output expectation

After loading and alignment, the pipeline produces a master event-level dataset with columns such as:

- `snapshot_time`
- `pair`
- `net_sentiment`
- `abs_sentiment`
- `sentiment_change`
- `side_streak`
- `extreme_streak_70`
- `entry_time`
- `entry_close`
- `ret_1b`, `ret_2b`, `ret_4b`, `ret_6b`, `ret_12b`, `ret_24b`, `ret_48b`
- `contrarian_ret_1b`, `contrarian_ret_2b`, `contrarian_ret_4b`, `contrarian_ret_6b`, `contrarian_ret_12b`, `contrarian_ret_24b`, `contrarian_ret_48b`

## 6. Minimal validation checklist

Before running the pipeline on a new dataset, verify:

- sentiment files follow `YYYY_MM_DD_HHMM.csv`
- sentiment files contain `pair`, `perc`, `direction`, `time`
- FX files contain `time_utc`, `open`, `high`, `low`, `close`, `tick_volume`
- pair naming can be inferred from FX filenames
- timestamps are parsable
- timezone assumptions are still correct
- raw data is available locally and used in compliance with its source terms, licenses, and any applicable contractual restrictions