from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any


def try_import_pandas() -> Any | None:
    try:
        import pandas as pd
    except Exception:
        return None
    return pd


def read_csv_dict(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def resolve_trade_date(feature_root: Path, trade_date: str) -> str:
    if str(trade_date or "").strip():
        return str(trade_date).strip()

    day_file = feature_root / "calendars" / "day.txt"
    if day_file.exists():
        lines = [x.strip() for x in day_file.read_text(encoding="utf-8").splitlines() if x.strip()]
        if lines:
            return lines[-1]

    return datetime.now().strftime("%Y%m%d")


def load_industry_name(level_file: Path, industry_code: str) -> str:
    if not level_file.exists():
        return ""
    rows = read_csv_dict(level_file)
    for row in rows:
        if row.get("indexCode", "").strip() == industry_code:
            return row.get("industryName", "").strip()
    return ""


def _format_num(v: Any, digits: int = 4) -> str:
    if v is None:
        return "NA"
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return "NA"


def _try_read_parquet_row(path: Path, date_col: str, trade_date: str) -> dict[str, Any] | None:
    pd = try_import_pandas()
    if pd is None:
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None

    if df.empty:
        return None

    if date_col not in df.columns:
        return df.iloc[-1].to_dict()

    target = str(trade_date)
    s = df[date_col].astype(str)

    exact = df[s == target]
    if not exact.empty:
        return exact.iloc[-1].to_dict()

    if target.isdigit() and len(target) == 8:
        lower = target + "000000"
        upper = target + "235959"
        day_rows = df[(s >= lower) & (s <= upper)]
        if not day_rows.empty:
            return day_rows.iloc[-1].to_dict()

    older = df[s <= target]
    if not older.empty:
        return older.iloc[-1].to_dict()

    return df.iloc[-1].to_dict()


def build_market_snapshot(feature_root: Path, industry_code: str, industry_name: str, trade_date: str) -> str:
    lines = [f"行业: {industry_name or '未知'} ({industry_code})", f"交易日: {trade_date}"]
    feature_file = feature_root / f"{industry_code}.pqt"
    if not feature_file.exists():
        lines.append(f"未找到行业特征文件: {feature_file.name}")
        return "\n".join(lines)

    row = _try_read_parquet_row(feature_file, "date", trade_date)
    if row is None:
        lines.append("未读取到行业特征（当前环境缺少 pandas/pyarrow 或文件不可读）。")
        return "\n".join(lines)

    selected = [
        "close",
        "ret_pct",
        "high_low_spread",
        "open_close_spread",
        "volume_ratio_5",
        "ma_close_5",
        "ma_close_20",
        "ma_close_60",
        "ma_gap_5",
        "ma_gap_20",
        "rolling_vol_20",
        "ATR_14",
        "RSI_14",
        "MACD_dif",
        "MACD_dea",
        "MACD_hist",
        "KDJ_K",
        "KDJ_D",
        "KDJ_J",
        "CCI_14",
    ]
    lines.append(f"特征文件: {feature_file.name}")
    lines.append("关键指标:")
    for key in selected:
        lines.append(f"- {key}: {_format_num(row.get(key))}")
    return "\n".join(lines)


def _find_news_file(news_root: Path, industry_code: str) -> Path | None:
    cands = sorted(news_root.glob(f"*_{industry_code}.pqt"))
    return cands[0] if cands else None


def _load_news_records_from_parquet(news_file: Path, trade_date: str, limit: int = 3) -> list[dict[str, Any]]:
    pd = try_import_pandas()
    if pd is None:
        return []
    try:
        df = pd.read_parquet(news_file)
    except Exception:
        return []

    if df.empty or "dt" not in df.columns:
        return []

    dt = df["dt"].astype(str)
    upper = trade_date + "235959"
    filtered = df[dt <= upper]
    if filtered.empty:
        filtered = df
    tail = filtered.tail(limit)
    cols = [c for c in ["dt", "src", "title", "content", "channel", "category"] if c in tail.columns]
    return tail[cols].to_dict("records")


def _load_news_records_from_csv(
    news_csv_root: Path, industry_code: str, trade_date: str, limit: int = 3
) -> list[dict[str, str]]:
    cands = sorted(news_csv_root.glob(f"*_{industry_code}.csv"))
    if not cands:
        return []
    rows = read_csv_dict(cands[0])
    if not rows:
        return []

    upper = trade_date + "235959"
    filtered = [r for r in rows if str(r.get("dt", "")) <= upper]
    if not filtered:
        filtered = rows
    return filtered[-limit:]


def build_news_snapshot(
    news_root: Path,
    news_csv_root: Path,
    industry_code: str,
    industry_name: str,
    trade_date: str,
    extra_news_text: str,
    limit: int = 3,
) -> tuple[str, list[dict[str, str]]]:
    lines = [f"行业: {industry_name or '未知'} ({industry_code})", f"交易日: {trade_date}"]

    news_file = _find_news_file(news_root, industry_code)
    records: list[dict[str, Any]] = []
    source = ""

    if news_file is not None:
        records = _load_news_records_from_parquet(news_file, trade_date, limit=limit)
        source = news_file.name

    if not records:
        records = _load_news_records_from_csv(news_csv_root, industry_code, trade_date, limit=limit)
        if records:
            source = f"CSV fallback (*_{industry_code}.csv)"

    ui_records: list[dict[str, str]] = []
    if records:
        lines.append(f"新闻源: {source}")
        lines.append(f"近期新闻摘要(最多{limit}条):")
        for r in records:
            dt = str(r.get("dt", ""))
            title = str((r.get("title") or "")).strip()
            content = str((r.get("content") or "")).strip().replace("\n", " ")
            snippet = title if title else content
            snippet = snippet[:140]
            lines.append(f"- {dt} | {snippet}")
            ui_records.append({"dt": dt, "title": title, "snippet": snippet, "src": str(r.get("src", ""))})
    else:
        lines.append("未读取到行业新闻（可检查 parquet 依赖或文件路径）。")

    if extra_news_text.strip():
        lines.append("附加新闻文本:")
        lines.append(extra_news_text.strip())

    return "\n".join(lines), ui_records
