"""
Data-source abstraction.

The pipeline consumes per-agent daily time series through the `DataSource`
interface. Today the only implementation is `LocalFileSource`, which reads the
AGWA-derived .bot/.human/.chrome/.pothuman files already on disk. When richer
features are re-derived from the full 490 GB Zenodo MySQL/TokuDB dump (calendar
timestamps, full RFC 7231 status codes incl. 403/429), implement `DBSource`
against the same interface and the rest of the pipeline is unchanged.

An `Agent` bundles the parsed frame with metadata. `features` is indexed by the
integer `day` column (days since first observation), with one row per observed
day. `active` marks days with >0 hits (agents are intermittent; most behavioral
analysis runs on active days only).
"""
from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

import config


@dataclass
class Agent:
    agent_id: str           # filename stem, == AGWA a_id
    label: str              # "bot" | "human" | "chrome" | "pothuman"
    features: pd.DataFrame  # columns == config.COLUMNS, one row per observed day
    meta: dict = field(default_factory=dict)

    @property
    def active(self) -> pd.DataFrame:
        """Rows on which the agent made at least one request."""
        return self.features[self.features["hits"] > 0]

    @property
    def n_active(self) -> int:
        return int((self.features["hits"] > 0).sum())

    @property
    def lifespan_days(self) -> int:
        if self.features.empty:
            return 0
        return int(self.features["day"].iloc[-1] - self.features["day"].iloc[0] + 1)


def _read_agent_file(path: Path) -> pd.DataFrame | None:
    """Robustly parse one per-agent TSV.

    The data files have no header (line 1 is already day 0), but we detect and
    skip a stray header row if present. Returns None if the file is unusable.
    """
    try:
        df = pd.read_csv(path, sep="\t", header=None, engine="c")
    except Exception:
        return None
    if df.shape[1] < len(config.COLUMNS):
        return None
    # Keep only the expected columns (defensive against trailing empties).
    df = df.iloc[:, : len(config.COLUMNS)]
    df.columns = config.COLUMNS
    # Coerce to numeric; a header row becomes NaN and is dropped.
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if df.empty:
        return None
    df = df.sort_values("day").reset_index(drop=True)
    return df


class DataSource:
    """Interface: iterate agents, optionally filtered by class label."""

    def agents(self, labels: list[str] | None = None) -> Iterator[Agent]:
        raise NotImplementedError

    def load_all(self, labels: list[str] | None = None) -> list[Agent]:
        return list(self.agents(labels))


class LocalFileSource(DataSource):
    """Reads the AGWA-derived per-agent files from `config.DATA_DIR`."""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = Path(data_dir or config.DATA_DIR)
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data dir not found: {self.data_dir}. "
                f"Set AGWA_DATA_DIR to the folder holding the .bot/.human/.chrome files."
            )

    def agents(self, labels: list[str] | None = None) -> Iterator[Agent]:
        labels = labels or list(config.CLASSES)
        for label in labels:
            pattern = config.CLASSES[label]
            for fp in sorted(glob.glob(str(self.data_dir / pattern))):
                path = Path(fp)
                df = _read_agent_file(path)
                if df is None:
                    continue
                yield Agent(
                    agent_id=path.stem,
                    label=label,
                    features=df,
                    meta={"path": str(path), "n_rows": len(df)},
                )


class DBSource(DataSource):
    """Placeholder for re-derivation from the full Zenodo AGWA database.

    Implement against the same `Agent` contract to unlock calendar-aligned time,
    full RFC 7231 status codes (403/429 = explicit block / rate-limit), and the
    raw robots.txt access log. The downstream analyses need no changes.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "DBSource is a stub. Re-derive per-agent series from the Zenodo dump "
            "(10.5281/zenodo.14497695) and yield Agent objects with the same columns "
            "(plus optional p403/p429 and a calendar `date` column)."
        )


def get_source() -> DataSource:
    """Factory: returns the active data source (local files for now)."""
    return LocalFileSource()
