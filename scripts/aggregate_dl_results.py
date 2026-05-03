import ast
import re
from pathlib import Path
import pandas as pd


LOG_DIR = Path("logs")


def clean_np(val):
    """Convert np.float64(x) → x"""
    if isinstance(val, str) and "np.float64" in val:
        return float(val.split("(")[1].rstrip(")"))
    return val


def parse_metrics_dict(line):
    """
    Extract dict from:
    test metrics: {...}
    """
    try:
        dict_str = line.split("test metrics:")[1].strip()

        # Replace np.float64(...) with plain float
        dict_str = re.sub(r"np\.float64\((.*?)\)", r"\1", dict_str)

        return ast.literal_eval(dict_str)
    except Exception:
        return None


def parse_logs():
    rows = []

    for file in LOG_DIR.glob("*.log"):
        with open(file, "r") as f:
            lines = f.readlines()

        model = None
        config = None

        for line in lines:

            # Model
            if "=== MLP Training ===" in line:
                model = "MLP"
            elif "=== LSTM Training ===" in line:
                model = "LSTM"

            # Config
            if "dataset_version" in line and "{" in line:
                try:
                    config = ast.literal_eval(line.split("|")[-1].strip())
                except:
                    continue

            # Metrics
            if "test metrics:" in line:
                metrics = parse_metrics_dict(line)

                if metrics and config:
                    row = {
                        "model": model,
                        "feature_set": config.get("feature_set"),
                        "horizon": config.get("target_horizon"),
                        "quantile": config.get("label_quantile"),
                        "regime": config.get("regime"),
                        **metrics,
                    }
                    rows.append(row)

    return pd.DataFrame(rows)


def compute_score(df):
    df = df.copy()
    df["score"] = (
        0.4 * df["f1"]
        + 0.3 * df["precision"]
        + 0.3 * df["recall"]
    )
    return df


def filter_collapsed(df):
    return df[
        (df["precision"] > 0)
        | (df["recall"] > 0)
    ]


def main():
    df = parse_logs()

    if df.empty:
        print("❌ No results found (unexpected now)")
        return

    df = compute_score(df)
    df_valid = filter_collapsed(df)

    print("\n=== TOP RESULTS (NON-COLLAPSED) ===\n")
    print(
        df_valid.sort_values("score", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    print("\n=== DIAGNOSTICS ===")
    print(f"Total runs: {len(df)}")
    print(f"Collapsed runs: {len(df) - len(df_valid)}")

    print("\n=== BEST PER (model, feature_set, regime) ===\n")
    print(
        df_valid.sort_values("score", ascending=False)
        .groupby(["model", "feature_set", "regime"])
        .head(1)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()