import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from chronos import Chronos2Pipeline

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP


class ChronosZeroShotModel(Model):
    """
    Wrapper around Chronos-2 pipeline for zero-shot forecasting on qlib datasets.
    It does not train; `fit` simply marks the model as ready and downloads the pipeline.
    """

    def __init__(
        self,
        model_path: str = "/z5s/bob/home/rcc/hub/models--amazon--chronos-2",
        repo_id: str = "amazon/chronos-2",
        device_map: str = "cuda",
        quantile_levels: Optional[List[float]] = None,
        quantile_index: int = 1,
        prediction_length: Optional[int] = None,
        usenews: bool = True,
        context_window: Optional[int] = None,
        label_shift: int = 2,
        logger_name: str = "ChronosZeroShotModel",
        **kwargs,
    ):
        super().__init__()
        self.logger = logging.getLogger(logger_name)
        self.model_path = model_path
        self.repo_id = repo_id
        self.device_map = device_map
        self.quantile_levels = quantile_levels or [0.1, 0.5, 0.9]
        self.quantile_index = quantile_index
        self.prediction_length = prediction_length
        self.usenews = usenews
        self.context_window = context_window
        self.label_shift = max(0, int(label_shift))
        self.pipeline: Optional[Chronos2Pipeline] = None
        self.fitted = False

    # ------------------------------------------------------------------ #
    # qlib interface
    # ------------------------------------------------------------------ #
    def fit(self, dataset: DatasetH, **kwargs):
        # Chronos is zero-shot; simply load pipeline (if not yet) and flag fitted
        self._ensure_pipeline()
        self.fitted = True
        self.logger.info("ChronosZeroShotModel loaded. No training performed.")

    def predict(self, dataset: DatasetH, **kwargs) -> pd.Series:
        if not self.fitted:
            raise ValueError("ChronosZeroShotModel is not fitted yet!")

        self._ensure_pipeline()
        handler = dataset.handler
        label_df = handler.fetch(col_set="label", data_key=DataHandlerLP.DK_R)
        if label_df is None or label_df.empty:
            raise ValueError("Label data is empty, cannot perform prediction.")

        label_df = self._flatten_df(label_df).rename(columns={label_df.columns[0]: "target"})

        feature_df = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_R)
        if feature_df is not None and not feature_df.empty:
            feature_df = self._flatten_df(feature_df, prefix="feat_")
            merged = label_df.join(feature_df, how="left")
        else:
            merged = label_df

        if self.usenews:
            try:
                news_df = handler.fetch(col_set="news", data_key=DataHandlerLP.DK_R)
            except Exception:
                news_df = None
            if news_df is not None and not news_df.empty:
                news_df = self._flatten_df(news_df, prefix="news_")
                merged = merged.join(news_df, how="left")

        merged = merged.reset_index().rename(columns={"datetime": "timestamp", "instrument": "id"})
        merged["timestamp"] = pd.to_datetime(merged["timestamp"])

        unique_ts = np.sort(merged["timestamp"].unique())
        syn_dates = pd.date_range(start="2000-01-01", periods=len(unique_ts), freq="D")
        ts_to_syn = {real: syn for real, syn in zip(unique_ts, syn_dates)}
        syn_to_ts = {syn: real for real, syn in ts_to_syn.items()}
        merged["syn_timestamp"] = merged["timestamp"].map(ts_to_syn)

        segments: Dict[str, List[str]] = dataset.segments
        test_start = pd.Timestamp(segments["test"][0])
        test_end = pd.Timestamp(segments["test"][1])

        future_mask = (merged["timestamp"] >= test_start) & (merged["timestamp"] <= test_end)
        future_cov = merged.loc[future_mask].copy()
        if future_cov.empty:
            raise ValueError("Insufficient data within test segment for Chronos inference.")

        chronos_ready = merged.drop(columns=["timestamp"]).rename(columns={"syn_timestamp": "timestamp"})
        chronos_ready.sort_values(["timestamp", "id"], inplace=True)

        future_syn = chronos_ready.loc[future_mask].copy()
        future_syn.sort_values(["timestamp", "id"], inplace=True)
        future_real_index = future_cov.set_index(["timestamp", "id"]).index

        future_ts = np.sort(future_syn["timestamp"].unique())
        if future_ts.size == 0:
            raise ValueError("No future timestamps detected for Chronos prediction.")

        first_future_ts = future_ts[0]
        context_df = chronos_ready.loc[chronos_ready["timestamp"] < first_future_ts].copy()
        if context_df.empty:
            raise ValueError("Context window is empty. Cannot run Chronos without history.")

        if self.label_shift > 0:
            allowed_ts = first_future_ts - pd.Timedelta(days=self.label_shift)
            context_df.loc[context_df["timestamp"] > allowed_ts, "target"] = np.nan

        if self.context_window:
            ctx_ts = np.sort(context_df["timestamp"].unique())
            if ctx_ts.size > self.context_window:
                keep_ts = set(ctx_ts[-self.context_window :])
                context_df = context_df[context_df["timestamp"].isin(keep_ts)]

        if not context_df["target"].notna().any():
            raise ValueError(
                "Context has no known targets after applying label_shift/context window. "
                "Increase historical window or reduce label_shift."
            )

        future_ids = future_syn["id"].unique()
        prediction_length = self.prediction_length or len(future_ts)
        if prediction_length > len(future_ts):
            prediction_length = len(future_ts)

        target_future_ts = future_ts[:prediction_length]
        future_index = pd.MultiIndex.from_product([target_future_ts, future_ids], names=["timestamp", "id"])
        future_df = pd.DataFrame(index=future_index).reset_index()[["timestamp", "id"]]

        quantile = self._quantile_column()
        pred_df = self.pipeline.predict_df(
            context_df,
            future_df=future_df,
            prediction_length=prediction_length,
            quantile_levels=self.quantile_levels,
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )

        quantile_col = self._resolve_quantile_column(pred_df.columns, quantile)
        pred_df = pred_df.rename(columns={quantile_col: "score"})
        pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"]).map(syn_to_ts)
        pred_series = pred_df.set_index(["timestamp", "id"])["score"].sort_index()

        # align with requested test horizon
        pred_series = pred_series.reindex(future_real_index)
        pred_series.index = pred_series.index.set_names(["datetime", "instrument"])
        pred_series.name = "score"
        return pred_series

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _ensure_pipeline(self):
        if self.pipeline is not None:
            return
        load_path = self._resolve_model_path()
        self.logger.info("Loading Chronos2 pipeline from %s", load_path)
        self.pipeline = Chronos2Pipeline.from_pretrained(load_path, device_map=self.device_map)

    def _resolve_model_path(self) -> str:
        path = Path(self.model_path).expanduser()
        if path.exists():
            if (path / "config.json").exists():
                return str(path)
            snapshots = path / "snapshots"
            if snapshots.exists():
                candidates = sorted(
                    [p for p in snapshots.iterdir() if (p / "config.json").exists()],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if candidates:
                    return str(candidates[0])
        if path.exists():
            self.logger.warning("Cannot locate config.json under %s; falling back to repo %s", path, self.repo_id)
        return self.repo_id

    @staticmethod
    def _flatten_df(df: pd.DataFrame, prefix: Optional[str] = None) -> pd.DataFrame:
        flat = df.copy()
        if isinstance(flat.columns, pd.MultiIndex):
            flat.columns = [
                "_".join([str(c) for c in tup if c not in (None, "")]).strip("_") for tup in flat.columns.to_flat_index()
            ]
        else:
            flat.columns = flat.columns.astype(str)
        if prefix:
            flat = flat.add_prefix(prefix)
        return flat

    def _quantile_column(self) -> str:
        value = self.quantile_levels[self.quantile_index % len(self.quantile_levels)]
        return f"{value}"

    @staticmethod
    def _resolve_quantile_column(columns, target: str) -> str:
        target_val = float(target)
        candidates = {
            target,
            f"{target_val:.1f}",
            f"{target_val:.2f}",
            f"{target_val:.3f}",
            f"quantile_{target}",
            f"quantile_{target_val}",
            f"quantile_{target_val:.2f}",
            f"p{int(target_val * 100)}",
            f"q{int(target_val * 100)}",
        }
        for col in columns:
            col_str = str(col)
            if col_str in candidates:
                return col_str
        raise ValueError(f"Expected one of {candidates} in Chronos output columns {list(columns)}")
