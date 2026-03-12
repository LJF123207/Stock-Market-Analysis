from __future__ import annotations

import json
import re

from .llm_client import get_llm
from .types import IndustryDebateState, InvestmentDecision


def _extract_json_from_text(text: str) -> tuple[dict, str]:
    raw_text = str(text or "").strip()
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            try:
                parsed = json.loads(line)
                cleaned_lines = str(text or "").splitlines()
                if cleaned_lines and cleaned_lines[-1].strip() == line:
                    cleaned = "\n".join(cleaned_lines[:-1]).strip()
                else:
                    cleaned = raw_text
                return parsed, cleaned
            except Exception:
                pass

    for match in re.finditer(r"\{[^{}]*\}", str(text or "")):
        candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
            cleaned = raw_text.replace(candidate, "").strip()
            return parsed, cleaned
        except Exception:
            continue
    return {}, raw_text


def safe_parse_decision(text: str) -> InvestmentDecision:
    fallback: InvestmentDecision = {
        "action": "hold",
        "confidence": 0.5,
        "full_report": str(text or "").strip(),
    }
    data, cleaned_report = _extract_json_from_text(text)
    if not data:
        return fallback

    action = data.get("action", "hold")
    if action not in {"buy", "sell", "hold"}:
        action = "hold"

    try:
        confidence = float(data.get("confidence", 0.5))
    except Exception:
        confidence = 0.5
    confidence = max(0.0, min(confidence, 1.0))

    return {
        "action": action,
        "confidence": confidence,
        "full_report": cleaned_report,
    }


def technical_researcher(state: IndustryDebateState) -> dict[str, str]:
    system_prompt = (
        "你是一名资深股票技术分析师，擅长从技术指标中提炼短期交易信号。"
        "你的输出必须清晰、简洁、交易导向。"
    )
    user_prompt = (
        "请分析以下股票的技术指标数据，并提供一个聚焦于短期交易信号的技术面总结：\n\n"
        "1. 用 5-6 句话总结当前技术形态，重点说明价格趋势、动量和波动情况。\n"
        "2. 根据指标判断当前市场结构，例如：趋势状态（上升趋势 / 下降趋势 / 震荡）、"
        "动量状态（增强 / 减弱 / 背离）、波动状态（放大 / 收敛）。\n"
        "3. 识别看涨信号和看跌信号，并说明对应指标依据。\n"
        "4. 基于所有技术指标，判断该股票未来短期（3-10个交易日）市场情绪："
        "看涨、看跌、中性，并简要说明原因。\n\n"
        f"行业: {state['industry_name']}({state['industry_code']})\n"
        f"交易日: {state['trade_date']}\n"
        "技术指标数据：\n"
        f"{state['market_snapshot']}\n\n"
        "请以清晰、简洁、交易导向的方式提供分析，并突出任何可能影响短期价格走势的技术信号。\n"
        "输出格式：\n\n"
        "技术总结:\n"
        "(4-6句话总结技术形态)\n\n"
        "看涨信号:\n"
        "...\n\n"
        "看跌信号:\n"
        "...\n\n"
        "短期市场情绪:\n"
        "(看涨 / 看跌 / 中性)\n\n"
        "原因:\n"
        "(1-2句话说明原因)"
    )
    return {"technical_report": get_llm().ask(system=system_prompt, user=user_prompt)}


def news_researcher(state: IndustryDebateState) -> dict[str, str]:
    system_prompt = (
        "你是一名资深金融新闻分析师，擅长评估新闻事件对股票市场的短期交易影响。"
        "请基于提供的新闻做结构化分析，并给出可交易的短期判断。"
    )
    user_prompt = (
        f"请分析以下有关 {state['industry_name'] or state['industry_code']} 的财经新闻，并给出结构化分析。\n\n"
        "任务：\n"
        "1. 用 4-6 句话总结新闻核心内容。\n"
        "2. 提取新闻中最重要的市场驱动因素。\n"
        "3. 识别利多因素与利空因素。\n"
        "4. 判断未来短期（3-8个交易日）的市场情绪。\n\n"
        f"行业: {state['industry_name']}({state['industry_code']})\n"
        f"交易日: {state['trade_date']}\n"
        "新闻数据（最近三条为主）：\n"
        f"{state['news_snapshot']}\n\n"
        "输出格式：\n\n"
        "新闻总结:\n"
        "(4-6句话总结技术形态)\n\n"
        "看涨信号:\n"
        "...\n\n"
        "看跌信号:\n"
        "...\n\n"
        "短期市场情绪:\n"
        "(看涨 / 看跌 / 中性)\n\n"
        "原因:\n"
        "(1-2句话说明原因)"
    )
    return {"news_report": get_llm().ask(system=system_prompt, user=user_prompt)}


def bull_debater(state: IndustryDebateState) -> dict[str, str]:
    system_prompt = (
        "你是一名股票市场的看涨分析师（Bullish Analyst）。"
        "你的任务是基于已有分析报告，专门寻找支持股价上涨的证据，构建最有说服力的看涨论点。"
    )
    user_prompt = (
        "请重点关注：\n"
        "1. 技术指标中的看涨信号（动量增强、趋势上行、超卖反弹等）\n"
        "2. 新闻中的利好因素（业绩增长、行业利好、政策支持等）\n"
        "3. 市场情绪改善信号\n\n"
        "任务：\n"
        "1. 用 2-3 句话总结支持股价上涨的主要理由。\n"
        "2. 列出 3-5 个最关键的看涨因素。\n"
        "3. 简要说明这些因素如何推动股价上涨。\n"
        "4. 给出未来 3-8 个交易日的上涨逻辑。\n\n"
        f"行业: {state['industry_name']}({state['industry_code']})\n"
        f"技术分析:\n{state['technical_report']}\n\n"
        f"新闻分析:\n{state['news_report']}\n\n"
        "输出格式：\n\n"
        "看涨逻辑:\n"
        "(2-3句话总结看涨逻辑)\n\n"
        "关键看涨信号:\n"
        "...\n"
        "...\n"
        "...\n\n"
        "短期上涨趋势:\n"
        "(解释未来可能的上涨路径)"
    )
    return {"bull_case": get_llm().ask(system=system_prompt, user=user_prompt)}


def bear_debater(state: IndustryDebateState) -> dict[str, str]:
    system_prompt = (
        "你是一名股票市场的看跌分析师（Bearish Analyst）。"
        "你的任务是基于已有分析报告，专门寻找支持股价下跌的证据，构建最有说服力的看跌论点。"
    )
    user_prompt = (
        "请重点关注：\n"
        "1. 技术指标中的看跌信号（趋势转弱、超买回调、动量衰减等）\n"
        "2. 新闻中的利空因素（监管风险、业绩下滑、行业压力等）\n"
        "3. 市场情绪恶化信号\n\n"
        "任务：\n"
        "1. 用 2-3 句话总结支持股价下跌的主要理由。\n"
        "2. 列出 3-5 个最关键的看跌因素。\n"
        "3. 简要说明这些因素如何导致股价回调。\n"
        "4. 给出未来 3-8 个交易日的下跌逻辑。\n\n"
        f"行业: {state['industry_name']}({state['industry_code']})\n"
        f"技术分析:\n{state['technical_report']}\n\n"
        f"新闻分析:\n{state['news_report']}\n\n"
        f"看涨逻辑:\n{state['bull_case']}\n\n"
        "输出格式：\n\n"
        "看跌逻辑:\n"
        "(2-3句话总结看跌逻辑)\n\n"
        "关键看跌信号:\n"
        "...\n"
        "...\n"
        "...\n\n"
        "短期下跌趋势:\n"
        "(解释未来可能的下跌路径)"
    )
    return {"bear_case": get_llm().ask(system=system_prompt, user=user_prompt)}


def investment_committee(state: IndustryDebateState) -> dict[str, InvestmentDecision]:
    system_prompt = (
        "你是一名资深股票市场首席策略分析师（Chief Market Strategist）。"
        "你将综合新闻分析、技术分析、看涨论证与看跌论证，给出最终市场判断。"
    )
    user_prompt = (
        "分析步骤：\n"
        "1. 简要总结新闻面和技术面的核心信息。\n"
        "2. 对比看涨与看跌观点，评估双方论据强弱。\n"
        "3. 判断当前市场主要驱动因素（情绪、趋势、风险或事件）。\n"
        "4. 基于所有信息给出最终投资判断。\n\n"
        "请重点考虑：\n"
        "- 技术指标是否支持趋势延续或趋势反转\n"
        "- 新闻事件是否可能在短期影响市场情绪\n"
        "- 多空观点中哪一方证据更有说服力\n"
        "- 潜在风险与不确定性\n\n"
        f"行业: {state['industry_name']}({state['industry_code']})\n"
        f"交易日: {state['trade_date']}\n"
        f"新闻分析:\n{state['news_report']}\n\n"
        f"技术分析:\n{state['technical_report']}\n\n"
        f"看涨逻辑:\n{state['bull_case']}\n\n"
        f"看跌逻辑:\n{state['bear_case']}\n\n"
        "请先按以下格式输出：\n\n"
        "市场情况总结:\n"
        "(3-4句话总结当前市场情况)\n\n"
        "看涨方论据:\n"
        "(列出看涨方最有力的2-3个理由)\n\n"
        "看跌方论据:\n"
        "(列出看跌方最有力的2-3个理由)\n\n"
        "关键市场因素:\n"
        "(当前最重要的影响因素，例如技术趋势、新闻事件、情绪变化等)\n\n"
        "Final Decision:\n"
        "(Bullish / Bearish / Neutral)\n\n"
        "Confidence Level:\n"
        "(0-100%)\n\n"
        "原因:\n"
        "(2-3句话解释最终判断的核心逻辑)\n\n"
        "最后请额外输出一行 JSON（仅一行）用于系统解析，格式必须严格如下：\n"
        '{"action":"buy|sell|hold","confidence":0.0}\n'
        "其中 action 映射：Bullish->buy, Bearish->sell, Neutral->hold；"
        "confidence 取 0~1 小数。"
    )
    raw = get_llm().ask(system=system_prompt, user=user_prompt)
    return {"final_decision": safe_parse_decision(raw)}
