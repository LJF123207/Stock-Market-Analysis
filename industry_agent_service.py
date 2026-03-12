"""Backward-compatible facade for industry agent analysis service.

This module preserves existing imports used by app.py while the implementation
is organized into layered modules under `industry_agents/`.
"""

from industry_agents import resolve_trade_date, run_industry_debate, run_industry_debate_with_progress

__all__ = [
    "resolve_trade_date",
    "run_industry_debate",
    "run_industry_debate_with_progress",
]
