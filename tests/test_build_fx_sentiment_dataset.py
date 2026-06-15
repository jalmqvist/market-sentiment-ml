from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
for _p in [str(_REPO_ROOT), str(_SCRIPTS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from build_fx_sentiment_dataset import deduplicate_pair_entry_rows


def test_deduplicate_pair_entry_rows_keeps_latest_snapshot():
    entry_time = pd.Timestamp("2024-01-01 08:00:00")
    df = pd.DataFrame(
        {
            "pair": ["eur-usd", "eur-usd", "usd-jpy"],
            "entry_time": [entry_time, entry_time, entry_time],
            "snapshot_time": [
                pd.Timestamp("2024-01-01 07:42:00"),
                pd.Timestamp("2024-01-01 07:57:00"),
                pd.Timestamp("2024-01-01 07:30:00"),
            ],
            "entry_close": [1.10, 1.11, 145.0],
        }
    )

    deduped = deduplicate_pair_entry_rows(df)

    assert deduped.duplicated(subset=["pair", "entry_time"]).sum() == 0
    eur_usd_row = deduped.loc[deduped["pair"] == "eur-usd"].iloc[0]
    assert eur_usd_row["snapshot_time"] == pd.Timestamp("2024-01-01 07:57:00")
    assert eur_usd_row["entry_close"] == 1.11
