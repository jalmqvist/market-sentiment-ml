import re
import pandas as pd
from pathlib import Path

LOG_DIR = Path("logs")


def parse_log_file(path: Path):
    """
    Extract metrics + metadata from a log file.
    """

    text = path.read_text()

    def extract(pattern, default=None, cast=float):
        m = re.search(pattern, text)
        if not m:
            return default
        return cast(m.group(1))

    # --- metadata ---
    model = "LSTM" if "lstm" in path.name else "MLP"

    feature_set = extract(r"feature_set': '([^']+)'", default="unknown", cast=str)
    horizon = extract(r"target_horizon': (\d+)", default=None, cast=int)

    # --- metrics ---
    accuracy = extract(r"'accuracy': ([0-9\.]+)")
    precision = extract(r"'precision': ([0-9\.]+)")
    recall = extract(r"'recall': ([0-9\.]+)")
    f1 = extract(r"'f1': ([0-9\.]+)")
    sharpe = extract(r"'sharpe': ([0-9\.\-]+)")
    n = extract(r"'n': (\d+)", cast=int)

    return {
        "model": model,
        "feature_set": feature_set,
        "horizon": horizon,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sharpe": sharpe,
        "n": n,
        "file": path.name,
    }


def main():
    rows = []

    for file in LOG_DIR.glob("*.log"):
        try:
            rows.append(parse_log_file(file))
        except Exception as e:
            print(f"Skipping {file}: {e}")

    if not rows:
        print("No logs found.")
        return

    df = pd.DataFrame(rows)

    # --- drop incomplete rows ---
    df = df.dropna(subset=["accuracy", "precision", "recall", "f1"])

    # --- detect collapsed models ---
    df["collapsed"] = (df["precision"] == 0) & (df["recall"] == 0)

    # --- remove collapsed ---
    df_clean = df[~df["collapsed"]].copy()

    if df_clean.empty:
        print("\n⚠️ All models collapsed (predicting single class).")
        print("This means NO SIGNAL at current setup.\n")
        return

    # --- ranking score ---
    # prioritize predictive quality + tradability
    df_clean["score"] = (
        df_clean["f1"].fillna(0) * 0.6 +
        df_clean["sharpe"].fillna(0) * 0.4
    )

    # --- sort ---
    df_clean = df_clean.sort_values("score", ascending=False)

    print("\n=== TOP RESULTS (NON-COLLAPSED) ===\n")
    print(
        df_clean[
            [
                "model",
                "feature_set",
                "horizon",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "sharpe",
                "n",
                "score",
            ]
        ].head(15).to_string(index=False)
    )

    # --- diagnostics ---
    total = len(df)
    collapsed = df["collapsed"].sum()

    print("\n=== DIAGNOSTICS ===")
    print(f"Total runs: {total}")
    print(f"Collapsed runs: {collapsed} ({collapsed/total:.1%})")

    # --- best by group ---
    print("\n=== BEST PER (model, feature_set) ===\n")
    best = (
        df_clean.sort_values("score", ascending=False)
        .groupby(["model", "feature_set"])
        .first()
        .reset_index()
    )

    print(
        best[
            [
                "model",
                "feature_set",
                "horizon",
                "precision",
                "recall",
                "f1",
                "sharpe",
                "score",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
