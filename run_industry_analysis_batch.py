from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

from industry_agents.service import run_industry_debate


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LEVEL_FILE = PROJECT_ROOT / "qlib" / "data_stocks" / "sw" / "level.csv"
DEFAULT_FEATURE_ROOT = PROJECT_ROOT / "qlib" / "data_stocks" / "IndexAndNews0928-0604"
DEFAULT_NEWS_ROOT = PROJECT_ROOT / "qlib" / "data_news" / "L2_dedup"
DEFAULT_NEWS_CSV_ROOT = PROJECT_ROOT / "qlib" / "data_news" / "L2"
# Set output root here directly (no frontend / no export required).
OUTPUT_ROOT = PROJECT_ROOT / "forcast" / "base" 


def _parse_date(value: str) -> datetime.date:
    raw = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {value}. Use YYYY-MM-DD or YYYYMMDD.")


def _load_trade_dates(feature_root: Path) -> list[str]:
    day_file = feature_root / "calendars" / "day.txt"
    if not day_file.exists():
        raise FileNotFoundError(f"Trading calendar not found: {day_file}")
    return [line.strip() for line in day_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def _select_trade_dates(all_dates: list[str], start_date: str, end_date: str) -> list[str]:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if start > end:
        raise ValueError(f"start_date > end_date: {start_date} > {end_date}")

    selected: list[str] = []
    for d in all_dates:
        try:
            day = _parse_date(d)
        except ValueError:
            continue
        if start <= day <= end:
            selected.append(d)
    return selected


def _normalize_date_token(value: str) -> str:
    return _parse_date(value).strftime("%Y%m%d")


def _sanitize_text(value: object) -> str:
    return str(value or "").replace("\r\n", "\n").strip()


def _is_valid_decision(action: object, confidence: object) -> bool:
    if action not in {"buy", "sell", "hold"}:
        return False
    try:
        c = float(confidence)
    except Exception:
        return False
    return 0.0 <= c <= 1.0


def _cache_file_path(cache_dir: Path, industry_code: str, trade_date: str, news_text: str, news_limit: int) -> Path:
    digest = hashlib.sha1(
        f"{industry_code.strip().upper()}|{trade_date}|{news_limit}|{news_text}".encode("utf-8")
    ).hexdigest()[:16]
    safe_code = industry_code.strip().upper().replace("/", "_")
    return cache_dir / f"{safe_code}_{trade_date}_{digest}.json"


def _run_one_day(
    *,
    industry_code: str,
    trade_date: str,
    level_file: Path,
    feature_root: Path,
    news_root: Path,
    news_csv_root: Path,
    news_text: str,
    news_limit: int,
    max_retries: int,
    retry_wait: float,
    cache_dir: Path,
    use_cache: bool,
) -> dict[str, str]:
    cache_file = _cache_file_path(cache_dir, industry_code, trade_date, news_text, news_limit)
    if use_cache and cache_file.exists():
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            action_line = str(payload.get("action_line", "")).strip()
            analysis_line = str(payload.get("analysis_line", "")).strip()
            if action_line and analysis_line:
                return {"trade_date": trade_date, "action_line": action_line, "analysis_line": analysis_line}
        except Exception:
            pass

    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            result = run_industry_debate(
                industry_code=industry_code,
                trade_date=trade_date,
                level_file=level_file,
                feature_root=feature_root,
                news_root=news_root,
                news_csv_root=news_csv_root,
                news_text=news_text,
                news_limit=max(1, int(news_limit)),
            )
            action_conf = result.get("final_decision", {}) or {}
            action = action_conf.get("action")
            confidence = action_conf.get("confidence")
            out_date = str(result.get("trade_date", trade_date))

            if _is_valid_decision(action, confidence):
                action_line = f"{out_date} {action} {float(confidence):.6f}"
            else:
                action_line = f"{out_date} fail"

            analysis_line = json.dumps(
                {
                    "date": out_date,
                    "technical_report": _sanitize_text(result.get("technical_report", "")),
                    "news_report": _sanitize_text(result.get("news_report", "")),
                    "bull_case": _sanitize_text(result.get("bull_case", "")),
                    "bear_case": _sanitize_text(result.get("bear_case", "")),
                    "final_decision": _sanitize_text(action_conf.get("full_report", "")),
                },
                ensure_ascii=False,
            )
            out = {"trade_date": trade_date, "action_line": action_line, "analysis_line": analysis_line}
            if use_cache:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(
                    json.dumps({"action_line": action_line, "analysis_line": analysis_line}, ensure_ascii=False),
                    encoding="utf-8",
                )
            return out
        except Exception as exc:
            last_error = str(exc).strip()
            if attempt < max_retries:
                time.sleep(max(0.0, retry_wait))

    return {
        "trade_date": trade_date,
        "action_line": f"{trade_date} fail",
        "analysis_line": json.dumps(
            {
                "date": trade_date,
                "technical_report": "",
                "news_report": "",
                "bull_case": "",
                "bear_case": "",
                "final_decision": f"fail ({last_error})",
            },
            ensure_ascii=False,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch run industry multi-agent analysis by trading dates and save outputs."
    )
    parser.add_argument("--industry-code", required=True, help="Industry code, e.g. 801011.SI")
    parser.add_argument("--start-date", required=True, help="Start date, YYYY-MM-DD or YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="End date, YYYY-MM-DD or YYYYMMDD")

    parser.add_argument("--news-text", default="", help="Extra news context appended to each day.")
    parser.add_argument("--news-limit", type=int, default=3, help="Max number of recent news records.")
    parser.add_argument("--max-retries", type=int, default=1, help="Retries for a failed trading day.")
    parser.add_argument("--retry-wait", type=float, default=1.5, help="Seconds to wait between retries.")
    parser.add_argument("--no-cache", action="store_true", help="Disable local day-level cache.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately if a day fails.")

    parser.add_argument("--level-file", type=Path, default=DEFAULT_LEVEL_FILE)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--news-root", type=Path, default=DEFAULT_NEWS_ROOT)
    parser.add_argument("--news-csv-root", type=Path, default=DEFAULT_NEWS_CSV_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    all_dates = _load_trade_dates(args.feature_root)
    trade_dates = _select_trade_dates(all_dates, args.start_date, args.end_date)
    if not trade_dates:
        raise RuntimeError(
            f"No trading dates found in range [{args.start_date}, {args.end_date}] under {args.feature_root}"
        )

    code_token = args.industry_code.strip().upper().replace("/", "_")
    start_token = _normalize_date_token(args.start_date)
    end_token = _normalize_date_token(args.end_date)
    out_dir = OUTPUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    action_file = out_dir / f"{code_token}_{start_token}_{end_token}_action.txt"
    analysis_file = out_dir / f"{code_token}_{start_token}_{end_token}_analysis.jsonl"

    action_lines: list[str] = ["date action confidence"]
    analysis_lines: list[str] = []
    by_date: dict[str, dict[str, str]] = {}
    cache_dir = out_dir / ".cache"
    use_cache = not args.no_cache

    print(f"[INFO] Running {len(trade_dates)} trading days for {args.industry_code} ...")
    print(f"[INFO] Output root: {out_dir}")
    print(f"[INFO] Cache: {'on' if use_cache else 'off'} ({cache_dir})")

    for i, trade_date in enumerate(trade_dates, start=1):
        print(f"[{i}/{len(trade_dates)}] {trade_date} ...", flush=True)
        out = _run_one_day(
            industry_code=args.industry_code,
            trade_date=trade_date,
            level_file=args.level_file,
            feature_root=args.feature_root,
            news_root=args.news_root,
            news_csv_root=args.news_csv_root,
            news_text=args.news_text,
            news_limit=args.news_limit,
            max_retries=max(0, int(args.max_retries)),
            retry_wait=float(args.retry_wait),
            cache_dir=cache_dir,
            use_cache=use_cache,
        )
        by_date[trade_date] = out
        if args.stop_on_error and out["action_line"].endswith(" fail"):
            break

    for trade_date in trade_dates:
        if trade_date not in by_date:
            continue
        action_lines.append(by_date[trade_date]["action_line"])
        analysis_lines.append(by_date[trade_date]["analysis_line"])

    action_file.write_text("\n".join(action_lines) + "\n", encoding="utf-8")
    analysis_file.write_text("\n".join(analysis_lines) + ("\n" if analysis_lines else ""), encoding="utf-8")

    print("[DONE] Batch analysis finished.")
    print(f"[DONE] Action file: {action_file.resolve()}")
    print(f"[DONE] Analysis file: {analysis_file.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
