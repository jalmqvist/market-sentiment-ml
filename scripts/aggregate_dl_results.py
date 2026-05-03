import re
from pathlib import Path
import pandas as pd

LOG_DIR = Path("logs")


# ---------------------------
# Parse CONFIG line
# ---------------------------
def parse_config_line(line):
    try:
        line = line.split("CONFIG |")[-1].strip()
        pattern = r"(\w+)=([^\s]+)"
        matches = dict(re.findall(pattern, line))

        return {
            "model": matches.get("model", "").upper(),
            "pair": matches.get("pair"),
            "regime": matches.get("regime"),
            "horizon": int(matches.get("horizon", 0)),
            "quantile": float(matches.get("quantile", 0)),
        }
    except Exception:
        return None


# ---------------------------
# Parse metrics block
# ---------------------------
def parse_metrics_block(lines, i):
    metrics = {}

    for j in range(i + 1, min(i + 10, len(lines))):
        line = lines[j].strip()

        if ":" not in line:
            continue

        # Strip logging prefix
        if "|" in line:
            line = line.split("|")[-1].strip()

        if ":" not in line:
            continue

        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()

        try:
            val = float(val)
        except:
            continue

        metrics[key] = val

    return metrics if metrics else None


# ---------------------------
# Parse all logs
# ---------------------------
def parse_logs():
    rows = []

    for file in LOG_DIR.glob("*.log"):
        with open(file, "r") as f:
            lines = f.readlines()

        config = None

        for i, line in enumerate(lines):

            if "CONFIG |" in line:
                config = parse_config_line(line)

            if "=== Test metrics ===" in line:
                metrics = parse_metrics_block(lines, i)

                if metrics and config:
                    rows.append({**config, **metrics})

    return pd.DataFrame(rows)


# ---------------------------
# Score
# ---------------------------
def compute_score(df):
    df = df.copy()
    df["score"] = (
        0.4 * df["f1"]
        + 0.3 * df["precision"]
        + 0.3 * df["recall"]
    )
    return df


# ---------------------------
# Remove collapsed models
# ---------------------------
def filter_collapsed(df):
    return df[
        (df["precision"] > 0)
        | (df["recall"] > 0)
    ]


# ---------------------------
# Weighted heatmap (NEW)
# ---------------------------
def weighted_heatmap(df):
    df = df.copy()

    df["weighted_f1"] = df["f1"] * df["n"]

    grouped = (
        df
        .groupby(["pair", "regime"])
        .agg(
            total_n=("n", "sum"),
            weighted_f1=("weighted_f1", "sum"),
        )
    )

    grouped["f1"] = grouped["weighted_f1"] / grouped["total_n"]

    return grouped["f1"].unstack().round(3)


# ---------------------------
# Main
# ---------------------------
def main():
    df = parse_logs()

    if df.empty:
        print("❌ No results found")
        return

    df = compute_score(df)
    df_valid = filter_collapsed(df)

    print("\n=== PAIR × REGIME (WEIGHTED F1) ===\n")
    print(weighted_heatmap(df_valid))

    print("\n=== TOP RESULTS (NON-COLLAPSED) ===\n")
    print(
        df_valid
        .sort_values("score", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    print("\n=== DIAGNOSTICS ===")
    print(f"Total runs: {len(df)}")
    print(f"Valid runs: {len(df_valid)}")


if __name__ == "__main__":
    main()