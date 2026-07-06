"""CBG-level demographic sampling helpers for synthetic population generation.

The generator still builds household structure from the existing county-level
tables, but this module lets age/sex come from block-group B01001 sex-by-age
counts when the SafeGraph Open Census bundle is available.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class SexAgeBucket:
    field: str
    sex: str
    min_age: int
    max_age: int


B01001_SEX_AGE_BUCKETS = [
    SexAgeBucket("B01001e3", "M", 0, 4),
    SexAgeBucket("B01001e4", "M", 5, 9),
    SexAgeBucket("B01001e5", "M", 10, 14),
    SexAgeBucket("B01001e6", "M", 15, 17),
    SexAgeBucket("B01001e7", "M", 18, 19),
    SexAgeBucket("B01001e8", "M", 20, 20),
    SexAgeBucket("B01001e9", "M", 21, 21),
    SexAgeBucket("B01001e10", "M", 22, 24),
    SexAgeBucket("B01001e11", "M", 25, 29),
    SexAgeBucket("B01001e12", "M", 30, 34),
    SexAgeBucket("B01001e13", "M", 35, 39),
    SexAgeBucket("B01001e14", "M", 40, 44),
    SexAgeBucket("B01001e15", "M", 45, 49),
    SexAgeBucket("B01001e16", "M", 50, 54),
    SexAgeBucket("B01001e17", "M", 55, 59),
    SexAgeBucket("B01001e18", "M", 60, 61),
    SexAgeBucket("B01001e19", "M", 62, 64),
    SexAgeBucket("B01001e20", "M", 65, 66),
    SexAgeBucket("B01001e21", "M", 67, 69),
    SexAgeBucket("B01001e22", "M", 70, 74),
    SexAgeBucket("B01001e23", "M", 75, 79),
    SexAgeBucket("B01001e24", "M", 80, 84),
    SexAgeBucket("B01001e25", "M", 85, 100),
    SexAgeBucket("B01001e27", "F", 0, 4),
    SexAgeBucket("B01001e28", "F", 5, 9),
    SexAgeBucket("B01001e29", "F", 10, 14),
    SexAgeBucket("B01001e30", "F", 15, 17),
    SexAgeBucket("B01001e31", "F", 18, 19),
    SexAgeBucket("B01001e32", "F", 20, 20),
    SexAgeBucket("B01001e33", "F", 21, 21),
    SexAgeBucket("B01001e34", "F", 22, 24),
    SexAgeBucket("B01001e35", "F", 25, 29),
    SexAgeBucket("B01001e36", "F", 30, 34),
    SexAgeBucket("B01001e37", "F", 35, 39),
    SexAgeBucket("B01001e38", "F", 40, 44),
    SexAgeBucket("B01001e39", "F", 45, 49),
    SexAgeBucket("B01001e40", "F", 50, 54),
    SexAgeBucket("B01001e41", "F", 55, 59),
    SexAgeBucket("B01001e42", "F", 60, 61),
    SexAgeBucket("B01001e43", "F", 62, 64),
    SexAgeBucket("B01001e44", "F", 65, 66),
    SexAgeBucket("B01001e45", "F", 67, 69),
    SexAgeBucket("B01001e46", "F", 70, 74),
    SexAgeBucket("B01001e47", "F", 75, 79),
    SexAgeBucket("B01001e48", "F", 80, 84),
    SexAgeBucket("B01001e49", "F", 85, 100),
]


class CbgSexAgeSampler:
    def __init__(self, bucket_counts_by_cbg: dict[str, list[tuple[SexAgeBucket, float]]], source_path: Path):
        self.bucket_counts_by_cbg = bucket_counts_by_cbg
        self.source_path = source_path

    @classmethod
    def load_default(cls, cbgs: Iterable[str]) -> Optional["CbgSexAgeSampler"]:
        path = cls._default_path()
        if path is None:
            return None
        return cls.load(path, cbgs=cbgs)

    @classmethod
    def load(cls, path: os.PathLike, cbgs: Optional[Iterable[str]] = None) -> Optional["CbgSexAgeSampler"]:
        path = Path(path)
        if not path.exists():
            return None

        wanted_cbgs = {cls._normalize_cbg(cbg) for cbg in (cbgs or [])}
        wanted_cbgs.discard("")
        fields = {bucket.field for bucket in B01001_SEX_AGE_BUCKETS}
        usecols = {"census_block_group", *fields}
        df = pd.read_csv(
            path,
            dtype={"census_block_group": str},
            usecols=lambda col: col in usecols,
        )
        if "census_block_group" not in df.columns:
            return None
        df["census_block_group"] = df["census_block_group"].map(cls._normalize_cbg)
        if wanted_cbgs:
            df = df[df["census_block_group"].isin(wanted_cbgs)]
        if df.empty:
            return None

        bucket_counts_by_cbg: dict[str, list[tuple[SexAgeBucket, float]]] = {}
        for _, row in df.iterrows():
            cbg = row["census_block_group"]
            if not cbg:
                continue
            counts: list[tuple[SexAgeBucket, float]] = []
            for bucket in B01001_SEX_AGE_BUCKETS:
                if bucket.field not in df.columns:
                    continue
                count = pd.to_numeric(row.get(bucket.field), errors="coerce")
                if pd.notna(count) and float(count) > 0:
                    counts.append((bucket, float(count)))
            if counts:
                bucket_counts_by_cbg[cbg] = counts

        if not bucket_counts_by_cbg:
            return None
        return cls(bucket_counts_by_cbg, path)

    @staticmethod
    def _default_path() -> Optional[Path]:
        env_path = os.getenv("DELINEO_CBG_DEMOGRAPHICS_PATH")
        if env_path:
            return Path(env_path)

        server_dir = Path(__file__).resolve().parent
        cwd = Path.cwd()
        candidates = []
        data_cwd = os.getenv("DELINEO_ALGORITHMS_DATA_CWD")
        if data_cwd:
            candidates.extend(CbgSexAgeSampler._cbg_b01_candidates(Path(data_cwd)))
        candidates.extend(CbgSexAgeSampler._cbg_b01_candidates(server_dir))
        candidates.extend(CbgSexAgeSampler._cbg_b01_candidates(cwd))
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _cbg_b01_candidates(base_dir: Path) -> list[Path]:
        return [
            base_dir / "data" / "safegraph_open_census_data_2016" / "data" / "cbg_b01.csv",
            base_dir / "data" / "cbg_b01.csv",
        ]

    @staticmethod
    def _normalize_cbg(cbg) -> str:
        if cbg is None or (isinstance(cbg, float) and pd.isna(cbg)):
            return ""
        text = str(cbg).strip()
        if text.endswith(".0"):
            text = text[:-2]
        digits = "".join(ch for ch in text if ch.isdigit())
        return digits.zfill(12) if digits else ""

    def sample(self, cbg: str, min_age: int, max_age: int, sex: Optional[str] = None) -> Optional[tuple[str, int]]:
        cbg = self._normalize_cbg(cbg)
        counts = self.bucket_counts_by_cbg.get(cbg)
        if not counts:
            return None

        choices: list[tuple[str, int, int]] = []
        weights: list[float] = []
        for bucket, count in counts:
            if sex is not None and bucket.sex != sex:
                continue
            overlap_min = max(int(min_age), bucket.min_age)
            overlap_max = min(int(max_age), bucket.max_age)
            if overlap_min > overlap_max:
                continue
            bucket_width = bucket.max_age - bucket.min_age + 1
            overlap_width = overlap_max - overlap_min + 1
            weight = count * (overlap_width / bucket_width)
            if weight <= 0:
                continue
            choices.append((bucket.sex, overlap_min, overlap_max))
            weights.append(weight)

        if not choices:
            return None
        sampled_sex, age_min, age_max = random.choices(choices, weights=weights, k=1)[0]
        return sampled_sex, random.randint(age_min, age_max)
