"""
Reference walk-forward implementation copied from market-phase-ml.

Purpose
-------
Behavioral Walk-forward Validation (PR7) shall use the identical fold
construction algorithm as MPML so that predictive and trading validation
evaluate identical historical windows.

This module intentionally mirrors the MPML implementation.

Changes should normally be made in MPML first and then synchronized here.
"""
import pandas as pd

#======
# fold generator helper functions:
#=====
def _find_index_pos(dt_index: pd.DatetimeIndex, dt: pd.Timestamp) -> int:
    """
    Return integer position of the last index value <= dt.
    Raises if dt is earlier than the first index value.
    """
    pos = dt_index.searchsorted(dt, side="right") - 1
    if pos < 0:
        raise ValueError(f"Date {dt} is before start of series {dt_index[0]}")
    return int(pos)


def _window_diagnostics(
    *,
    train_start_ts: pd.Timestamp,
    train_end_ts: pd.Timestamp,
    test_start_ts: pd.Timestamp,
    test_end_ts: pd.Timestamp,
) -> dict:
    """Return causal diagnostics for one train/test window."""
    train_start_ts = pd.Timestamp(train_start_ts)
    train_end_ts = pd.Timestamp(train_end_ts)
    test_start_ts = pd.Timestamp(test_start_ts)
    test_end_ts = pd.Timestamp(test_end_ts)

    assert train_start_ts <= train_end_ts, (
        "train_start_ts must be <= train_end_ts "
        f"({train_start_ts} !<= {train_end_ts})"
    )
    assert test_start_ts <= test_end_ts, (
        "test_start_ts must be <= test_end_ts "
        f"({test_start_ts} !<= {test_end_ts})"
    )
    assert train_end_ts < test_start_ts, (
        "train_end_ts must be < test_start_ts "
        f"({train_end_ts} !< {test_start_ts})"
    )

    # Normalize to calendar days because fold diagnostics are reported in day
    # units and some callers may provide non-midnight timestamps.
    train_start_day = train_start_ts.normalize()
    train_end_day = train_end_ts.normalize()
    test_start_day = test_start_ts.normalize()
    test_end_day = test_end_ts.normalize()

    overlap_start = max(train_start_day, test_start_day)
    overlap_end = min(train_end_day, test_end_day)
    overlap_days = (
        int((overlap_end - overlap_start).days + 1)
        if overlap_start <= overlap_end
        else 0
    )
    gap_days = int((test_start_day - train_end_day).days)
    return {
        "train_start_ts": train_start_ts,
        "train_end_ts": train_end_ts,
        "test_start_ts": test_start_ts,
        "test_end_ts": test_end_ts,
        "gap_days": gap_days,
        "overlap_days": overlap_days,
    }


def _print_window_diagnostics(prefix: str, **diag) -> None:
    print(
        f"{prefix} "
        f"train={diag['train_start_ts']} -> {diag['train_end_ts']} "
        f"test={diag['test_start_ts']} -> {diag['test_end_ts']} "
        f"gap_days={diag['gap_days']} overlap_days={diag['overlap_days']}"
    )


def generate_walkforward_folds_by_pos(
    dates: pd.DatetimeIndex,
    train_years: int = 7,
    test_months: int = 6,
    step_months: int = 6,
) -> list[dict]:
    """
    Walk-forward folds using date boundaries but converted to integer positions.

    Expanding window:
      - train_start fixed at first date
      - train_end advances by step_months
      - test window length = test_months
    """
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    if not dates.is_monotonic_increasing:
        dates = dates.sort_values()
    start = pd.Timestamp(dates.min())
    end = pd.Timestamp(dates.max())

    # We'll walk train_end forward in time using DateOffset,
    # then map boundaries to integer positions.
    train_start_dt = start
    train_end_dt = train_start_dt + pd.DateOffset(years=train_years)

    folds = []
    fold_id = 0

    while True:
        test_start_dt = train_end_dt + pd.Timedelta(days=1)
        test_end_dt = test_start_dt + pd.DateOffset(months=test_months)

        if test_start_dt >= end:
            break
        if test_end_dt > end:
            test_end_dt = end

        # Convert to positions (snap to nearest available bar <= boundary)
        train_start_pos = 0
        train_end_pos = _find_index_pos(dates, train_end_dt)

        # ----------------------------------------------------------
        # Causal fold boundary:
        # test must begin strictly AFTER the final train bar.
        #
        # Using calendar-based lookup for both train_end_dt and
        # test_start_dt can collapse onto the same trading bar when
        # the index is sparse/non-uniform.
        #
        # Therefore test_start_pos is defined positionally.
        # ----------------------------------------------------------
        test_start_pos = train_end_pos + 1

        if test_start_pos >= len(dates):
            break

        test_end_pos = _find_index_pos(dates, test_end_dt)

        if test_end_pos <= test_start_pos:
            break

        train_start_ts = dates[train_start_pos]
        train_end_ts = dates[train_end_pos]
        test_start_ts = dates[test_start_pos]
        test_end_ts = dates[test_end_pos]
        window_diag = _window_diagnostics(
            train_start_ts=train_start_ts,
            train_end_ts=train_end_ts,
            test_start_ts=test_start_ts,
            test_end_ts=test_end_ts,
        )

        folds.append({
            "fold": fold_id,
            "train_start_pos": train_start_pos,
            "train_end_pos": train_end_pos,
            "test_start_pos": test_start_pos,
            "test_end_pos": test_end_pos,
            "train_start_dt": str(train_start_ts.date()),
            "train_end_dt": str(train_end_ts.date()),
            "test_start_dt": str(test_start_ts.date()),
            "test_end_dt": str(test_end_ts.date()),
            "gap_days": window_diag["gap_days"],
            "overlap_days": window_diag["overlap_days"],
        })
        fold_id += 1

        # expanding window: move train_end forward
        train_end_dt = train_end_dt + pd.DateOffset(months=step_months)

        if train_end_dt >= end:
            break

    return folds
    
def walkforward_signature(
    train_years: int,
    test_months: int,
    step_months: int,
) -> dict:
    """
    Canonical description of the walk-forward protocol.

    Useful for experiment manifests and reproducibility.
    """
    return {
        "train_years": train_years,
        "test_months": test_months,
        "step_months": step_months,
        "protocol": "mpml_reference_v1",
    }
