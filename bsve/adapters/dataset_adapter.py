"""Ontology-agnostic master research dataset adapter layer for BSVE."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class DatasetAdapterConfig:
    """Configuration for normalized access into a master research dataset artifact."""

    pair_col: str = "pair"
    timestamp_col: str = "entry_time"
    pair_family_col: str = "pair_family"
    sentiment_columns: tuple[str, ...] = (
        "net_sentiment",
        "abs_sentiment",
        "crowd_side",
        "sentiment_change",
    )
    structural_columns: tuple[str, ...] = (
        "entry_open",
        "entry_high",
        "entry_low",
        "entry_close",
        "entry_tick_volume",
        "atr_pct",
        "trend_strength_12",
        "trend_strength_48",
    )
    feature_aliases: Mapping[str, str] = field(default_factory=dict)


class MasterResearchDatasetAdapter:
    """Adapter that exposes normalized, ontology-agnostic dataset access."""

    def __init__(
        self,
        dataset: pd.DataFrame,
        *,
        config: DatasetAdapterConfig | None = None,
        pair_family_membership: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config or DatasetAdapterConfig()
        self._dataset = dataset.copy()
        self._validate_core_columns()
        self._dataset[self.config.pair_col] = self._dataset[self.config.pair_col].map(
            self.normalize_pair
        )
        self._dataset[self.config.timestamp_col] = pd.to_datetime(
            self._dataset[self.config.timestamp_col], errors="coerce"
        )
        if self._dataset[self.config.timestamp_col].isna().any():
            raise ValueError(
                f"column '{self.config.timestamp_col}' contains non-datetime values"
            )

        self._pair_family_membership = self._build_pair_family_membership(
            pair_family_membership
        )

    @classmethod
    def from_artifact(
        cls,
        artifact_path: str | Path,
        *,
        config: DatasetAdapterConfig | None = None,
        pair_family_membership: Mapping[str, str] | None = None,
    ) -> "MasterResearchDatasetAdapter":
        """Load a master research dataset artifact from CSV or parquet."""
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"artifact path does not exist: {path}")

        if path.suffix.lower() == ".parquet":
            dataset = pd.read_parquet(path)
        else:
            dataset = pd.read_csv(path)

        return cls(
            dataset,
            config=config,
            pair_family_membership=pair_family_membership,
        )

    @staticmethod
    def normalize_pair(pair: str) -> str:
        """Normalize pair names to lowercase dash style (e.g., EURUSD -> eur-usd)."""
        p = str(pair).strip()
        if len(p) == 6 and p.isalpha():
            return f"{p[:3].lower()}-{p[3:].lower()}"
        p = p.replace("/", "-").replace("_", "-")
        return p.lower()

    @property
    def dataset(self) -> pd.DataFrame:
        """Return a defensive copy of the normalized dataset."""
        return self._dataset.copy()

    def resolve_feature_column(self, feature_name: str) -> str:
        """Resolve aliases to physical column names and validate existence."""
        column = self.config.feature_aliases.get(feature_name, feature_name)
        if column not in self._dataset.columns:
            raise KeyError(f"feature '{feature_name}' resolves to missing column '{column}'")
        return column

    def get_feature(
        self,
        feature_name: str,
        *,
        pairs: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Return normalized access to one feature with pair/timestamp keys."""
        column = self.resolve_feature_column(feature_name)
        frame = self._filter_pairs(pairs)
        return frame[[self.config.pair_col, self.config.timestamp_col, column]].copy()

    def get_features(
        self,
        feature_names: Iterable[str],
        *,
        pairs: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Return normalized access to multiple feature columns."""
        cols = [self.resolve_feature_column(name) for name in feature_names]
        frame = self._filter_pairs(pairs)
        return frame[[self.config.pair_col, self.config.timestamp_col, *cols]].copy()

    def get_pair_family_membership(self) -> dict[str, str]:
        """Return normalized pair -> family membership mapping."""
        return dict(self._pair_family_membership)

    def get_pair_family(self, pair: str) -> str | None:
        """Return the family label for one pair, if available."""
        return self._pair_family_membership.get(self.normalize_pair(pair))

    def get_pairs_for_family(self, family: str) -> list[str]:
        """Return pairs belonging to a given family label."""
        fam = str(family).strip().lower()
        pairs = [p for p, f in self._pair_family_membership.items() if str(f).lower() == fam]
        return sorted(pairs)

    def get_sentiment_observations(
        self,
        *,
        pairs: Sequence[str] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Return sentiment observations exposed by adapter configuration."""
        selected = list(columns) if columns else list(self.config.sentiment_columns)
        existing = [c for c in selected if c in self._dataset.columns]
        frame = self._filter_pairs(pairs)
        base_cols = [self.config.pair_col, self.config.timestamp_col]
        return frame[base_cols + existing].copy()

    def get_structural_observations(
        self,
        *,
        pairs: Sequence[str] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Return structural observations exposed by adapter configuration."""
        selected = list(columns) if columns else list(self.config.structural_columns)
        existing = [c for c in selected if c in self._dataset.columns]
        frame = self._filter_pairs(pairs)
        base_cols = [self.config.pair_col, self.config.timestamp_col]
        return frame[base_cols + existing].copy()

    def _validate_core_columns(self) -> None:
        required = {self.config.pair_col, self.config.timestamp_col}
        missing = required.difference(self._dataset.columns)
        if missing:
            raise ValueError(f"dataset missing required adapter columns: {sorted(missing)}")

    def _build_pair_family_membership(
        self,
        explicit_mapping: Mapping[str, str] | None,
    ) -> dict[str, str]:
        mapping: dict[str, str] = {}
        if self.config.pair_family_col in self._dataset.columns:
            pair_col = self.config.pair_col
            fam_col = self.config.pair_family_col
            unique = self._dataset[[pair_col, fam_col]].dropna(subset=[pair_col, fam_col])
            for pair, family in unique.itertuples(index=False):
                mapping[self.normalize_pair(pair)] = str(family)

        if explicit_mapping:
            for pair, family in explicit_mapping.items():
                mapping[self.normalize_pair(pair)] = str(family)

        return mapping

    def _filter_pairs(self, pairs: Sequence[str] | None) -> pd.DataFrame:
        if not pairs:
            return self._dataset
        normalized = {self.normalize_pair(p) for p in pairs}
        return self._dataset[self._dataset[self.config.pair_col].isin(normalized)]
