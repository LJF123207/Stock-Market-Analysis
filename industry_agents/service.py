from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .agents import bear_debater, bull_debater, investment_committee, news_researcher, technical_researcher
from .data_loader import build_market_snapshot, build_news_snapshot, load_industry_name, resolve_trade_date
from .graph import build_graph
from .types import IndustryDebateState

_GRAPH_APP = None


def _get_graph_app():
    global _GRAPH_APP
    if _GRAPH_APP is None:
        _GRAPH_APP = build_graph()
    return _GRAPH_APP


def run_industry_debate(
    *,
    industry_code: str,
    trade_date: str,
    level_file: Path,
    feature_root: Path,
    news_root: Path,
    news_csv_root: Path,
    news_text: str = "",
    news_limit: int = 3,
) -> dict[str, Any]:
    code = industry_code.strip().upper()
    resolved_date = resolve_trade_date(feature_root, trade_date)
    industry_name = load_industry_name(level_file, code)
    market_snapshot = build_market_snapshot(feature_root, code, industry_name, resolved_date)
    news_snapshot, news_records = build_news_snapshot(
        news_root=news_root,
        news_csv_root=news_csv_root,
        industry_code=code,
        industry_name=industry_name,
        trade_date=resolved_date,
        extra_news_text=news_text,
        limit=max(1, news_limit),
    )

    app = _get_graph_app()
    result = app.invoke(
        {
            "industry_code": code,
            "industry_name": industry_name,
            "trade_date": resolved_date,
            "market_snapshot": market_snapshot,
            "news_snapshot": news_snapshot,
        }
    )
    result["news_records"] = news_records
    return result


def run_industry_debate_with_progress(
    *,
    industry_code: str,
    trade_date: str,
    level_file: Path,
    feature_root: Path,
    news_root: Path,
    news_csv_root: Path,
    news_text: str = "",
    news_limit: int = 3,
    on_update: Callable[[str, Any], None] | None = None,
) -> dict[str, Any]:
    code = industry_code.strip().upper()
    resolved_date = resolve_trade_date(feature_root, trade_date)
    industry_name = load_industry_name(level_file, code)
    market_snapshot = build_market_snapshot(feature_root, code, industry_name, resolved_date)
    news_snapshot, news_records = build_news_snapshot(
        news_root=news_root,
        news_csv_root=news_csv_root,
        industry_code=code,
        industry_name=industry_name,
        trade_date=resolved_date,
        extra_news_text=news_text,
        limit=max(1, news_limit),
    )

    state: IndustryDebateState = {
        "industry_code": code,
        "industry_name": industry_name,
        "trade_date": resolved_date,
        "market_snapshot": market_snapshot,
        "news_snapshot": news_snapshot,
        "technical_report": "",
        "news_report": "",
        "bull_case": "",
        "bear_case": "",
        "final_decision": {
            "action": "hold",
            "confidence": 0.5,
            "full_report": "",
        },
    }

    def push(stage: str, payload: Any) -> None:
        if on_update is not None:
            on_update(stage, payload)

    import threading

    tech_error: list[str] = []
    news_error: list[str] = []

    def run_tech() -> None:
        try:
            out = technical_researcher(state)
            state["technical_report"] = out["technical_report"]
            push("technical_report", out["technical_report"])
        except Exception as exc:
            tech_error.append(str(exc))

    def run_news() -> None:
        try:
            out = news_researcher(state)
            state["news_report"] = out["news_report"]
            push("news_report", out["news_report"])
        except Exception as exc:
            news_error.append(str(exc))

    t1 = threading.Thread(target=run_tech, daemon=True)
    t2 = threading.Thread(target=run_news, daemon=True)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    if tech_error:
        raise RuntimeError(f"技术面研究员执行失败: {tech_error[0]}")
    if news_error:
        raise RuntimeError(f"新闻面研究员执行失败: {news_error[0]}")

    out_bull = bull_debater(state)
    state["bull_case"] = out_bull["bull_case"]
    push("bull_case", out_bull["bull_case"])

    out_bear = bear_debater(state)
    state["bear_case"] = out_bear["bear_case"]
    push("bear_case", out_bear["bear_case"])

    out_final = investment_committee(state)
    state["final_decision"] = out_final["final_decision"]
    push("final_decision", out_final["final_decision"])
    state["news_records"] = news_records

    return state
