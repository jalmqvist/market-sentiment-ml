from __future__ import annotations

import pandas as pd

from bsve.adapters.dataset_adapter import (
    DatasetAdapterConfig,
    MasterResearchDatasetAdapter,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pair": ["USDJPY", "EURJPY", "USDCHF"],
            "entry_time": pd.date_range("2024-01-01", periods=3, freq="h"),
            "pair_family": ["reactive_jpy", "reactive_jpy", "reactive_chf"],
            "net_sentiment": [10.0, -15.0, 5.0],
            "abs_sentiment": [10.0, 15.0, 5.0],
            "crowd_side": [1, -1, 1],
            "sentiment_change": [0.0, -25.0, 20.0],
            "atr_pct": [0.2, 0.3, 0.4],
        }
    )


def test_adapter_normalizes_pairs_and_exposes_families() -> None:
    adapter = MasterResearchDatasetAdapter(_sample_df())

    assert adapter.get_pair_family("usdjpy") == "reactive_jpy"
    assert adapter.get_pair_family("EURJPY") == "reactive_jpy"
    assert adapter.get_pairs_for_family("reactive_jpy") == ["eur-jpy", "usd-jpy"]


def test_adapter_feature_access_and_filters() -> None:
    adapter = MasterResearchDatasetAdapter(_sample_df())

    feature = adapter.get_feature("net_sentiment", pairs=["USDJPY"])
    assert list(feature.columns) == ["pair", "entry_time", "net_sentiment"]
    assert feature["pair"].tolist() == ["usd-jpy"]


def test_adapter_sentiment_and_structural_observations() -> None:
    adapter = MasterResearchDatasetAdapter(_sample_df())

    sentiment = adapter.get_sentiment_observations()
    structural = adapter.get_structural_observations(columns=["atr_pct"])

    assert "net_sentiment" in sentiment.columns
    assert "atr_pct" in structural.columns


def test_adapter_loads_from_csv_artifact(tmp_path) -> None:
    path = tmp_path / "master_research_dataset.csv"
    _sample_df().to_csv(path, index=False)

    adapter = MasterResearchDatasetAdapter.from_artifact(path)
    assert len(adapter.dataset) == 3


def test_adapter_resolves_feature_aliases() -> None:
    config = DatasetAdapterConfig(feature_aliases={"sentiment": "net_sentiment"})
    adapter = MasterResearchDatasetAdapter(_sample_df(), config=config)

    out = adapter.get_feature("sentiment")
    assert "net_sentiment" in out.columns
