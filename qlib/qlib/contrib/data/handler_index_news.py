# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Custom data loader & handler for parquet based index & news dataset.

This module enables loading tabular parquet files that contain
 - an instrument column (e.g. industry code),
 - a date column,
 - user specified feature columns,
 - optional news embedding columns,
 - a close price column (used to build labels).

Usage (yaml):

handler:
    class: IndexNewsDataHandler
    module_path: qlib.contrib.data.handler_index_news
    kwargs:
        data_dir: /path/to/parquet
        instrument_field: L2
        date_field: date
        feature_columns:
            range: {start: 2, end: 82}
        news_columns:
            range: {start: 82, end: 594}
        close_field: close
        instruments: ["801011.SI", "801012.SI"]
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import DataLoader
from qlib.utils import lazy_sort_index


def _ensure_list(obj) -> List:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, tuple):
        return list(obj)
    return [obj]


class IndexNewsParquetDataLoader(DataLoader):
    """Load parquet files that already contain engineered feature/news columns."""

    GROUP_FEATURE = "feature"
    GROUP_NEWS = "news"
    GROUP_LABEL = "label"

    def __init__(
        self,
        data_dir: Union[str, Path],
        instrument_field: Union[str, int],
        date_field: Union[str, int],
        feature_columns: Union[List, Dict],
        news_columns: Optional[Union[List, Dict]] = None,
        close_field: Union[str, int] = "close",
        label_forward_shift: int = 2,
        label_reference_shift: int = 1,
        instruments: Optional[Sequence[str]] = None,
        exclude_instruments: Optional[Sequence[str]] = None,
        file_ext: str = ".pqt",
    ):
        self.data_dir = Path(data_dir).expanduser()
        self.file_ext = file_ext
        self.instrument_files: Dict[str, Path] = {
            p.stem: p for p in sorted(self.data_dir.glob(f"*{self.file_ext}")) if p.is_file()
        }
        if exclude_instruments:
            exclude_set = {str(inst) for inst in exclude_instruments}
            self.instrument_files = {
                name: path for name, path in self.instrument_files.items() if name not in exclude_set
            }
        if not self.instrument_files:
            raise FileNotFoundError(f"No parquet files (ext={self.file_ext}) found in {self.data_dir}")

        # Load one sample file to resolve column positions/names.
        sample_path = next(iter(self.instrument_files.values()))
        sample_df = pd.read_parquet(sample_path)
        self._all_columns = list(sample_df.columns)

        self.instrument_field = self._resolve_single_column(instrument_field)
        self.date_field = self._resolve_single_column(date_field)
        self.close_field = self._resolve_single_column(close_field)

        self.feature_columns = self._resolve_column_list(feature_columns, allow_empty=False)
        self.news_columns = self._resolve_column_list(news_columns, allow_empty=True)

        # columns required when reading parquet
        required_cols = set([self.instrument_field, self.date_field, self.close_field])
        required_cols.update(self.feature_columns)
        required_cols.update(self.news_columns)
        self.required_columns = list(required_cols)

        self.label_forward_shift = int(label_forward_shift)
        self.label_reference_shift = int(label_reference_shift)

        self.default_instruments = list(instruments) if instruments else list(self.instrument_files.keys())

        # simple in-memory cache per instrument to avoid reloading frequently
        self._cache: Dict[str, pd.DataFrame] = {}

    # --------------------------------------------------------------------- #
    # Helpers for parsing user column specification
    # --------------------------------------------------------------------- #
    def _resolve_single_column(self, spec: Union[str, int]) -> str:
        if isinstance(spec, int):
            try:
                return self._all_columns[spec]
            except IndexError as exc:
                raise KeyError(f"Column index {spec} out of range") from exc
        if isinstance(spec, str):
            if spec not in self._all_columns:
                raise KeyError(f"Column '{spec}' not found in parquet schema")
            return spec
        raise TypeError(f"Unsupported column spec: {spec}")

    def _resolve_column_list(self, spec: Optional[Union[List, Dict]], allow_empty: bool) -> List[str]:
        if spec is None:
            if allow_empty:
                return []
            raise ValueError("Column specification is required")

        columns: List[str] = []

        def add_by_indices(indices: Iterable[int]):
            for idx in indices:
                try:
                    columns.append(self._all_columns[idx])
                except IndexError as exc:
                    raise KeyError(f"Column index {idx} out of range") from exc

        def add_by_range(range_cfg: Dict):
            start = int(range_cfg.get("start", 0))
            end = range_cfg.get("end")
            end = int(end) if end is not None else len(self._all_columns)
            add_by_indices(range(start, end))

        if isinstance(spec, dict):
            if "names" in spec:
                columns.extend(spec["names"])
            if "indices" in spec:
                add_by_indices(spec["indices"])
            if "range" in spec:
                add_by_range(spec["range"])
        else:
            spec_list = _ensure_list(spec)
            if spec_list:
                if all(isinstance(item, int) for item in spec_list):
                    add_by_indices(spec_list)  # type: ignore[arg-type]
                else:
                    columns.extend(spec_list)  # type: ignore[arg-type]

        # validate & deduplicate
        seen = set()
        resolved = []
        for name in columns:
            if name not in self._all_columns:
                raise KeyError(f"Column '{name}' not found in parquet schema")
            if name not in seen:
                seen.add(name)
                resolved.append(name)

        if not resolved and not allow_empty:
            raise ValueError("Resolved column list is empty")
        return resolved

    # --------------------------------------------------------------------- #
    # Core loading logic
    # --------------------------------------------------------------------- #
    def _available_instruments(self, instruments) -> List[str]:
        if instruments is None:
            return list(self.default_instruments)
        if isinstance(instruments, str):
            if instruments.lower() == "all":
                return list(self.instrument_files.keys())
            return [instruments]
        if isinstance(instruments, dict):
            inst = instruments.get("instrument") or instruments.get("instruments")
            if inst is None:
                raise NotImplementedError("Dictionary instruments config is not supported; please pass a list.")
            return _ensure_list(inst)
        if isinstance(instruments, (list, tuple, set)):
            return list(instruments)
        raise TypeError(f"Unsupported instruments spec: {instruments}")

    def _load_single(self, instrument: str) -> pd.DataFrame:
        if instrument in self._cache:
            return self._cache[instrument]

        if instrument not in self.instrument_files:
            raise FileNotFoundError(f"No parquet file found for instrument '{instrument}'")

        df = pd.read_parquet(self.instrument_files[instrument], columns=self.required_columns).copy()
        df.rename(columns={self.instrument_field: "__instrument__", self.date_field: "__date__"}, inplace=True)

        df["__instrument__"] = df["__instrument__"].astype(str)
        if np.issubdtype(df["__date__"].dtype, np.number):
            df["__date__"] = pd.to_datetime(df["__date__"].astype(str), format="%Y%m%d", errors="coerce")
        else:
            df["__date__"] = pd.to_datetime(df["__date__"], errors="coerce")
        df.dropna(subset=["__date__"], inplace=True)
        df.sort_values("__date__", inplace=True)

        # label
        group = df.groupby("__instrument__")[self.close_field]
        forward = group.shift(-self.label_forward_shift)
        reference = group.shift(-self.label_reference_shift)
        df["LABEL0"] = forward / reference - 1

        df.set_index(["__date__", "__instrument__"], inplace=True)
        df.index.set_names(["datetime", "instrument"], inplace=True)

        data_frames = []

        feature_df = df[self.feature_columns].astype(np.float32)
        feature_df.columns = pd.MultiIndex.from_product([[self.GROUP_FEATURE], feature_df.columns])
        data_frames.append(feature_df)

        if self.news_columns:
            news_df = df[self.news_columns].astype(np.float32)
            news_df.columns = pd.MultiIndex.from_product([[self.GROUP_NEWS], news_df.columns])
            data_frames.append(news_df)

        label_df = df[["LABEL0"]]
        label_df.columns = pd.MultiIndex.from_product([[self.GROUP_LABEL], ["LABEL0"]])
        data_frames.append(label_df)

        merged = pd.concat(data_frames, axis=1)
        self._cache[instrument] = merged
        return merged

    def load(self, instruments=None, start_time=None, end_time=None) -> pd.DataFrame:
        inst_list = self._available_instruments(instruments)
        if not inst_list:
            raise ValueError("Instrument list is empty.")

        frames = [self._load_single(inst) for inst in inst_list]
        data = pd.concat(frames, axis=0)
        data = lazy_sort_index(data)

        if start_time is not None or end_time is not None:
            start = pd.Timestamp(start_time) if start_time is not None else None
            end = pd.Timestamp(end_time) if end_time is not None else None
            idx = pd.IndexSlice[start:end, :]
            data = data.loc[idx]

        return data


class IndexNewsDataHandler(DataHandlerLP):
    """
    Data handler wrapping :class:`IndexNewsParquetDataLoader`.
    """

    _LABEL_PATTERN = re.compile(
        r"Ref\(\s*\$close\s*,\s*-(\d+)\s*\)\s*/\s*Ref\(\s*\$close\s*,\s*-1\s*\)\s*-\s*1", re.IGNORECASE
    )
    _logger = logging.getLogger(__name__)

    def __init__(
        self,
        data_dir: Union[str, Path],
        instrument_field: Union[str, int],
        date_field: Union[str, int],
        feature_columns: Union[List, Dict],
        news_columns: Optional[Union[List, Dict]] = None,
        close_field: Union[str, int] = "close",
        label_forward_shift: int = 2,
        label_reference_shift: int = 1,
        label: Optional[Sequence[str]] = None,
        instruments: Optional[Sequence[str]] = None,
        exclude_instruments: Optional[Sequence[str]] = None,
        start_time=None,
        end_time=None,
        infer_processors: Optional[List] = None,
        learn_processors: Optional[List] = None,
        shared_processors: Optional[List] = None,
        process_type=DataHandlerLP.PTYPE_A,
        drop_raw: bool = False,
        fit_start_time=None,
        fit_end_time=None,
        data_loader_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        derived_shift = self._parse_label_shift(label)
        if derived_shift is not None:
            label_forward_shift = derived_shift

        loader_kwargs = dict(
            data_dir=data_dir,
            instrument_field=instrument_field,
            date_field=date_field,
            feature_columns=feature_columns,
            news_columns=news_columns,
            close_field=close_field,
            label_forward_shift=label_forward_shift,
            label_reference_shift=label_reference_shift,
            instruments=instruments,
            exclude_instruments=exclude_instruments,
        )
        if data_loader_kwargs:
            loader_kwargs.update(data_loader_kwargs)

        data_loader = {
            "class": "IndexNewsParquetDataLoader",
            "module_path": "qlib.contrib.data.handler_index_news",
            "kwargs": loader_kwargs,
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors or [],
            learn_processors=learn_processors or [],
            shared_processors=shared_processors or [],
            process_type=process_type,
            drop_raw=drop_raw,
            **kwargs,
        )

    @classmethod
    def _parse_label_shift(cls, label_spec: Optional[Sequence[str]]) -> Optional[int]:
        if not label_spec:
            return None
        expr = label_spec[0]
        if not isinstance(expr, str):
            return None
        match = cls._LABEL_PATTERN.search(expr)
        if not match:
            cls._logger.warning("Failed to parse label expression %s; fallback to default shifts.", expr)
            return None
        value = int(match.group(1))
        cls._logger.info("Detected label forward shift=%s from expression %s", value, expr)
        return value
