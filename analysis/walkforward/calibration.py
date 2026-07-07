from __future__ import annotations

import numpy as np


def compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if y_true.size == 0:
        return {
            "calibration_ece": float("nan"),
            "calibration_abs_error_mean": float("nan"),
            "calibration_bins": int(n_bins),
        }, []

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)

    curve_rows: list[dict[str, float]] = []
    ece = 0.0
    abs_errors: list[float] = []

    for i in range(n_bins):
        mask = bin_ids == i
        count = int(mask.sum())
        if count == 0:
            curve_rows.append(
                {
                    "bin": float(i),
                    "bin_left": float(bins[i]),
                    "bin_right": float(bins[i + 1]),
                    "count": 0.0,
                    "mean_pred": float("nan"),
                    "observed_freq": float("nan"),
                    "abs_gap": float("nan"),
                }
            )
            continue

        mean_pred = float(y_prob[mask].mean())
        observed_freq = float(y_true[mask].mean())
        abs_gap = abs(mean_pred - observed_freq)

        curve_rows.append(
            {
                "bin": float(i),
                "bin_left": float(bins[i]),
                "bin_right": float(bins[i + 1]),
                "count": float(count),
                "mean_pred": mean_pred,
                "observed_freq": observed_freq,
                "abs_gap": float(abs_gap),
            }
        )
        ece += abs_gap * (count / y_true.size)
        abs_errors.append(abs_gap)

    summary = {
        "calibration_ece": float(ece),
        "calibration_abs_error_mean": float(np.mean(abs_errors)) if abs_errors else float("nan"),
        "calibration_bins": int(n_bins),
    }
    return summary, curve_rows
