from __future__ import annotations

from .agents import bear_debater, bull_debater, investment_committee, news_researcher, technical_researcher
from .types import IndustryDebateState


def build_graph():
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("Missing dependency: langgraph. Install with `pip install langgraph`.") from exc

    graph = StateGraph(IndustryDebateState)
    graph.add_node("technical_researcher", technical_researcher)
    graph.add_node("news_researcher", news_researcher)
    graph.add_node("bull_debater", bull_debater)
    graph.add_node("bear_debater", bear_debater)
    graph.add_node("investment_committee", investment_committee)

    graph.add_edge(START, "technical_researcher")
    graph.add_edge(START, "news_researcher")
    graph.add_edge(["technical_researcher", "news_researcher"], "bull_debater")
    graph.add_edge("bull_debater", "bear_debater")
    graph.add_edge("bear_debater", "investment_committee")
    graph.add_edge("investment_committee", END)
    return graph.compile()
