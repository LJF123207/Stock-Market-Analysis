from __future__ import annotations

from typing import Literal

try:
    from typing_extensions import TypedDict
except Exception:
    from typing import TypedDict


class InvestmentDecision(TypedDict):
    action: Literal["buy", "sell", "hold"]
    confidence: float
    full_report: str


class IndustryDebateState(TypedDict):
    industry_code: str
    industry_name: str
    trade_date: str
    market_snapshot: str
    news_snapshot: str
    technical_report: str
    news_report: str
    bull_case: str
    bear_case: str
    final_decision: InvestmentDecision
