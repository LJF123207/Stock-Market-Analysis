from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import numpy as np
import sqlite3
import subprocess
import re
import shutil
import os
import json
import hashlib
import threading
import uuid
from datetime import datetime
from pathlib import Path
from functools import wraps, lru_cache
import pyarrow.parquet as pq
from industry_agent_service import resolve_trade_date, run_industry_debate_with_progress
from industry_agents.data_loader import load_industry_name, build_market_snapshot, build_news_snapshot

app = Flask(__name__)
app.secret_key = "Stock-market-prediction-24"

STOCK_NAME_FILE = Path("/z5s/morph/home/ljf/Stock-Market-Prediction/qlib/data_stocks/stockname.txt")
STOCK_DATA_DIR = Path("/z5s/morph/home/ljf/Stock-Market-Prediction/qlib/data_stocks/IndexAndNews0928-0604")
NEWS_DATA_DIR = Path("/z5s/morph/home/ljf/Stock-Market-Prediction/qlib/data_news/L2_dedup")
NEWS_CSV_DIR = Path("/z5s/morph/home/ljf/Stock-Market-Prediction/qlib/data_news/L2")
INDUSTRY_LEVEL_FILE = Path("/z5s/morph/home/ljf/Stock-Market-Prediction/qlib/data_stocks/sw/level.csv")
QLIB_ROOT = Path("/z5s/morph/home/ljf/Stock-Market-Prediction/qlib")
QLIB_BENCHMARK_DIR = QLIB_ROOT / "examples" / "benchmarks"
QLIB_MLRUNS_DIR = QLIB_ROOT / "mlruns"
TREND_JOB_DIR = QLIB_ROOT / "output" / "trend_jobs"
TREND_MODEL_DIR = QLIB_ROOT / "output" / "trend_models"
TREND_PRED_DIR = QLIB_ROOT / "output" / "trend_predictions"
BACKTEST_JOB_DIR = QLIB_ROOT / "output" / "backtest_jobs"
BACKTEST_MODEL_DIR = QLIB_ROOT / "output" / "backtest_models"
MLRUNS_NAMED_DIR = QLIB_ROOT / "output" / "named_runs"
ARTIFACT_STORE_DIR = MLRUNS_NAMED_DIR / "artifacts"

FINANCIAL_COLUMN_CANDIDATES = {
    "market_cap": ["market_cap", "mkt_cap", "total_mv", "float_mv", "marketValue"],
    "pe": ["pe", "PE", "pe_ttm", "ttm_pe", "pe_ratio"],
    "pb": ["pb", "PB", "pb_lf", "pb_ratio"],
}

SCREEN_COLUMNS = [
    "symbol",
    "date",
    "preclose",
    "open",
    "close",
    "mktTradeQty",
    "mktTradeTurnover",
    "tradePriceMax",
    "tradePriceMin",
    "ret_pct",
    "MACD_dif",
    "MACD_dea",
    "MACD_hist",
    "RSI_14",
    "ma_close_5",
    "ma_close_20",
    "ma_close_60",
    "BOLL_mid",
    "BOLL_upper",
    "BOLL_lower",
    "net_trade_flow",
    "net_order_flow",
]

HISTORY_COLUMN_LABELS = {
    "date_display": "日期",
    "open": "开盘价",
    "close": "收盘价",
    "mktTradeQty": "成交量（股）",
    "mktTradeTurnover": "成交额（元）",
    "tradePriceMax": "成交最高价",
    "tradePriceMin": "成交最低价",
}

FACTOR_COLUMN_LABELS = {
    "date_display": "日期",
    "MACD_dif": "MACD DIF",
    "MACD_dea": "MACD DEA",
    "MACD_hist": "MACD HIST",
    "KDJ_K": "KDJ K",
    "KDJ_D": "KDJ D",
    "KDJ_J": "KDJ J",
    "RSI_14": "RSI_14",
    "BOLL_mid": "布林中轨",
    "BOLL_upper": "布林上轨",
    "BOLL_lower": "布林下轨",
}

SCREENER_OPTIONAL_COLUMNS = [
    ("market_cap", "市值"),
    ("pe", "PE"),
    ("pb", "PB"),
    ("RSI_14", "RSI"),
    ("MACD_dif", "MACD DIF"),
    ("MACD_hist", "MACD HIST"),
    ("mktTradeQty", "成交量"),
    ("mktTradeTurnover", "成交额"),
    ("net_trade_flow", "净成交量"),
    ("net_order_flow", "净新增订单量"),
]

SCREENER_SORT_CANDIDATES = [
    ("market_cap", "市值"),
    ("pe", "PE"),
    ("pb", "PB"),
    ("ret_pct", "日收益率"),
    ("mktTradeQty", "成交量"),
    ("mktTradeTurnover", "成交额"),
    ("RSI_14", "RSI"),
    ("MACD_dif", "MACD DIF"),
    ("net_trade_flow", "净成交量"),
]

USER_CREDENTIALS = {
    "admin": {"password": "password123", "email": "admin@example.com", "role": "admin"},
    "user1": {"password": "userpass", "email": "user@example.com", "role": "user"},
}

TREND_MODEL_OPTIONS = [
    ("alstm", "ALSTM"),
    ("gru", "GRU"),
    ("gats", "GATS"),
    ("tcn", "TCN"),
    ("lstm", "LSTM"),
    ("localformer", "Localformer"),
    ("transformer", "Transformer"),
]

TREND_MODE_OPTIONS = [
    ("tech", "仅技术"),
    ("tech_news", "技术+新闻"),
]

TREND_FUSION_OPTIONS = [
    ("add", "add"),
    ("concat", "concat"),
    ("crossattn", "crossattn"),
    ("decoder", "decoder"),
]

TREND_MODEL_CONFIG = {
    "alstm": QLIB_BENCHMARK_DIR / "ALSTM" / "workflow_config_alstm_Times_News.yaml",
    "gru": QLIB_BENCHMARK_DIR / "GRU" / "workflow_config_gru_Times_News.yaml",
    "gats": QLIB_BENCHMARK_DIR / "GATs" / "workflow_config_gats_Times_News.yaml",
    "tcn": QLIB_BENCHMARK_DIR / "TCN" / "workflow_config_tcn_Times_News.yaml",
    "lstm": QLIB_BENCHMARK_DIR / "LSTM" / "workflow_config_lstm_Times_News.yaml",
    "localformer": QLIB_BENCHMARK_DIR / "Localformer" / "workflow_config_localformer_Times_News.yaml",
    "transformer": QLIB_BENCHMARK_DIR / "Transformer" / "workflow_config_transformer_Times_News.yaml",
}

TREND_RUN_NAME_PATTERN = {
    "alstm": ("alstm",),
    "gru": ("gru",),
    "gats": ("gats",),
    "tcn": ("tcn",),
    "lstm": ("lstm",),
    "localformer": ("localformer",),
    "transformer": ("transformer",),
}

TREND_TRAIN_JOBS = {}
BACKTEST_JOBS = {}
INDUSTRY_ANALYSIS_JOBS = {}
INDUSTRY_ANALYSIS_LOCK = threading.Lock()
NEXT_DAY_HORIZON = 1
BACKTEST_DEFAULT_START = "2024-02-09"
BACKTEST_DEFAULT_END = "2025-06-04"
APP_EXPERIMENT_NAME = "workflow_local"
BACKTEST_BUY_METHOD_OPTIONS = [
    ("top", "买入高分(top)"),
    ("random", "买入随机(random)"),
]
BACKTEST_SELL_METHOD_OPTIONS = [
    ("bottom", "卖出低分(bottom)"),
    ("random", "卖出随机(random)"),
]


@app.route("/")
def home():
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        new_username = request.form.get("new_username")
        new_email = request.form.get("new_email")
        new_password = request.form.get("new_password")

        if not new_username or not new_email or not new_password:
            flash("All fields are required.", "error")
            return redirect(url_for("register"))

        if new_username in USER_CREDENTIALS:
            flash("Username already exists.", "error")
            return redirect(url_for("register"))

        USER_CREDENTIALS[new_username] = {
            "password": new_password,
            "email": new_email,
            "role": "user",
        }

        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username in USER_CREDENTIALS and USER_CREDENTIALS[username]["password"] == password:
            session["username"] = username
            session["role"] = USER_CREDENTIALS[username]["role"]
            return redirect(url_for("admin_dashboard" if session["role"] == "admin" else "index"))

        flash("Invalid username or password.", "error")

    return render_template("login.html")


@app.route("/admin_dashboard")
def admin_dashboard():
    if "username" not in session or session.get("role") != "admin":
        return redirect(url_for("login"))

    with sqlite3.connect("feedback.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT username, feedback FROM feedback")
        feedback_list = cursor.fetchall()

    return render_template("admin_dashboard.html", feedback_list=feedback_list, users=USER_CREDENTIALS)


@app.route("/logout")
def logout():
    session.pop("username", None)
    session.pop("role", None)
    return redirect(url_for("login"))


@app.route("/tutorial")
def tutorial():
    return render_template("tutorial.html")


def role_required(role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if "username" not in session or session.get("role") != role:
                flash("Unauthorized access!", "error")
                return redirect(url_for("login"))
            return f(*args, **kwargs)

        return decorated_function

    return decorator


@lru_cache(maxsize=1)
def load_stock_mapping():
    mapping = {}
    if STOCK_NAME_FILE.exists():
        lines = STOCK_NAME_FILE.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if not line.startswith("|"):
                continue
            parts = [p.strip() for p in line.strip().split("|")[1:-1]]
            if len(parts) < 2:
                continue
            name, code = parts[0], parts[1]
            if code == "指数代码" or code.startswith("---"):
                continue
            mapping[code] = name

    for parquet_path in STOCK_DATA_DIR.glob("*.pqt"):
        code = parquet_path.stem
        mapping.setdefault(code, code)

    return mapping


def _resolve_keyword_to_codes(keyword):
    text = str(keyword or "").strip()
    if not text:
        return []

    mapping = load_stock_mapping()
    upper_text = text.upper()
    lower_text = text.lower()

    exact = []
    partial = []
    for code, name in mapping.items():
        if upper_text == code.upper():
            exact.append(code)
        elif lower_text in code.lower() or lower_text in name.lower():
            partial.append(code)

    return exact if exact else partial[:30]


def _build_keyword_token(keyword):
    text = str(keyword or "").strip()
    if not text:
        return "all"
    codes = _resolve_keyword_to_codes(text)
    if codes:
        base = "|".join(sorted(codes))
    else:
        base = text.lower()
    return hashlib.md5(base.encode("utf-8")).hexdigest()[:8]


def _trend_config_id(model_key, mode, horizon, keyword, fusion_type, layer_num):
    # Trend model identity is independent of keyword/horizon by design.
    return f"{model_key}_{mode}_{fusion_type}_L{int(layer_num)}"


def _backtest_config_id(
    model_key,
    mode,
    keyword,
    start_date,
    end_date,
    topk,
    n_drop,
    account,
    open_cost,
    close_cost,
    min_cost,
    fusion_type,
    layer_num,
    method_buy,
    method_sell,
    hold_thresh,
    only_tradable,
    forbid_all_trade_at_limit,
):
    key_token = _build_keyword_token(keyword)
    start_date = _parse_date_input(start_date, BACKTEST_DEFAULT_START)
    end_date = _parse_date_input(end_date, BACKTEST_DEFAULT_END)
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    return (
        f"{model_key}_{mode}_{fusion_type}_L{int(layer_num)}_BT_"
        f"{start_date.replace('-', '')}_{end_date.replace('-', '')}_K{key_token}_"
        f"T{int(topk)}_D{int(n_drop)}_A{int(float(account))}_"
        f"OC{float(open_cost):.6f}_CC{float(close_cost):.6f}_MC{float(min_cost):.2f}_"
        f"MB{method_buy}_MS{method_sell}_H{int(hold_thresh)}_"
        f"OT{1 if only_tradable else 0}_FL{1 if forbid_all_trade_at_limit else 0}"
    )


def _is_default_backtest_profile(form_data):
    def _float_eq(a, b, eps=1e-12):
        return abs(float(a) - float(b)) <= eps

    return (
        _parse_date_input(form_data.get("start_date"), BACKTEST_DEFAULT_START) == BACKTEST_DEFAULT_START
        and _parse_date_input(form_data.get("end_date"), BACKTEST_DEFAULT_END) == BACKTEST_DEFAULT_END
        and int(form_data.get("topk", 0)) == 30
        and int(form_data.get("n_drop", 0)) == 5
        and _float_eq(form_data.get("account", 0), 100000000)
        and _float_eq(form_data.get("open_cost", 0), 0.000085)
        and _float_eq(form_data.get("close_cost", 0), 0.001085)
        and _float_eq(form_data.get("min_cost", 0), 5)
        and str(form_data.get("method_buy", "")).lower() == "top"
        and str(form_data.get("method_sell", "")).lower() == "bottom"
        and int(form_data.get("hold_thresh", 0)) == 1
        and bool(form_data.get("only_tradable", True)) is False
        and bool(form_data.get("forbid_all_trade_at_limit", False)) is True
    )


def _safe_copy(src, dst):
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False


def _trend_named_paths(config_id):
    named_dir = MLRUNS_NAMED_DIR / "trend" / config_id
    return {
        "dir": named_dir,
        "pred": named_dir / "pred.pkl",
        "label": named_dir / "label.pkl",
        "model": named_dir / "params.pkl",
        "meta": named_dir / "meta.json",
    }


def _backtest_named_paths(config_id):
    named_dir = MLRUNS_NAMED_DIR / "backtest" / config_id
    return {
        "dir": named_dir,
        "report": named_dir / "report_normal_1day.pkl",
        "indicator": named_dir / "indicator_analysis_1day.pkl",
        "meta": named_dir / "meta.json",
    }


def _trend_artifact_store_dir(config_id):
    return ARTIFACT_STORE_DIR / "trend" / config_id / "artifacts"


def _backtest_artifact_store_dir(config_id):
    return ARTIFACT_STORE_DIR / "backtest" / config_id / "artifacts"


def _sync_artifacts_tree(src_root, dst_root):
    src = Path(src_root)
    dst = Path(dst_root)
    if not src.exists():
        return False
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                # Skip copying unchanged files to reduce post-process latency.
                if target.exists():
                    try:
                        if target.stat().st_size == item.stat().st_size and target.stat().st_mtime >= item.stat().st_mtime:
                            continue
                    except Exception:
                        pass
                shutil.copy2(item, target)
            except Exception:
                pass
    return True


def _write_named_meta(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _has_trend_cache(config_id):
    art_dir = _trend_artifact_store_dir(config_id)
    art_pred = art_dir / "pred.pkl"
    if art_pred.exists():
        return str(art_pred)

    # Compatibility: migrate legacy trend artifact dirs like
    # "<config_id>_H*_K*/artifacts" to the new unified "<config_id>/artifacts".
    legacy_candidates = sorted(
        [
            p
            for p in (ARTIFACT_STORE_DIR / "trend").glob(f"{config_id}_H*_K*")
            if (p / "artifacts" / "pred.pkl").exists()
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if legacy_candidates:
        _sync_artifacts_tree(legacy_candidates[0] / "artifacts", art_dir)
        if art_pred.exists():
            return str(art_pred)

    # Backfill unified artifact store from historical metadata, then only serve from unified store.
    trend_meta = _trend_named_paths(config_id)["meta"]
    if trend_meta.exists():
        try:
            meta = json.loads(trend_meta.read_text(encoding="utf-8"))
            src_root = meta.get("artifact_store") or meta.get("artifact_root")
            if src_root:
                _sync_artifacts_tree(Path(src_root), art_dir)
        except Exception:
            pass

    return str(art_pred) if art_pred.exists() else None


def _has_backtest_cache(config_id):
    art_report = _backtest_artifact_store_dir(config_id) / "portfolio_analysis" / "report_normal_1day.pkl"
    if art_report.exists():
        return str(art_report)
    paths = _backtest_named_paths(config_id)
    if paths["report"].exists():
        return str(paths["report"])
    return None


def _materialize_backtest_cache_from_trend(form_data, backtest_config_id):
    if not _is_default_backtest_profile(form_data):
        return None

    trend_config_id = _trend_config_id(
        form_data["model"],
        form_data["mode"],
        NEXT_DAY_HORIZON,
        form_data["keyword"],
        form_data["fusion_type"],
        form_data["layer_num"],
    )

    trend_store = _trend_artifact_store_dir(trend_config_id)
    if not trend_store.exists():
        trend_meta = _trend_named_paths(trend_config_id)["meta"]
        if trend_meta.exists():
            try:
                meta = json.loads(trend_meta.read_text(encoding="utf-8"))
                fallback_art_root = meta.get("artifact_store") or meta.get("artifact_root")
                if fallback_art_root:
                    fallback_path = Path(fallback_art_root)
                    if fallback_path.exists():
                        _sync_artifacts_tree(fallback_path, trend_store)
            except Exception:
                pass
    if not trend_store.exists():
        return None

    report_src = trend_store / "portfolio_analysis" / "report_normal_1day.pkl"
    indicator_src = trend_store / "portfolio_analysis" / "indicator_analysis_1day.pkl"
    if not report_src.exists():
        return None

    backtest_store = _backtest_artifact_store_dir(backtest_config_id)
    _sync_artifacts_tree(trend_store, backtest_store)

    named = _backtest_named_paths(backtest_config_id)
    _safe_copy(report_src, named["report"])
    if indicator_src.exists():
        _safe_copy(indicator_src, named["indicator"])
    _write_named_meta(
        named["meta"],
        {
            "config_id": backtest_config_id,
            "reused_from_trend": trend_config_id,
            "artifact_root": str(backtest_store),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    return str(report_src)


def _prepare_trend_config(model_key, mode, horizon, keyword, fusion_type, layer_num):
    base_cfg = TREND_MODEL_CONFIG.get(model_key)
    if base_cfg is None or not base_cfg.exists():
        raise FileNotFoundError(f"未找到模型配置文件: {base_cfg}")

    text = base_cfg.read_text(encoding="utf-8")
    if re.search(r"(?m)^experiment_name:\s*.+$", text):
        text = re.sub(r"(?m)^experiment_name:\s*.+$", f"experiment_name: {APP_EXPERIMENT_NAME}", text, count=1)
    else:
        text = f"experiment_name: {APP_EXPERIMENT_NAME}\n" + text
    use_news = mode == "tech_news"

    text = re.sub(
        r"(?m)^(\s*usenews:\s*)(true|false)\s*$",
        lambda m: f"{m.group(1)}{'true' if use_news else 'false'}",
        text,
    )
    text = re.sub(
        r"(?m)^(\s*d_news:\s*)\d+\s*$",
        lambda m: f"{m.group(1)}{512 if use_news else 0}",
        text,
    )
    text = re.sub(
        r"(?m)^(\s*fusion_type:\s*)(\S+)(.*)$",
        lambda m: f"{m.group(1)}{fusion_type}{m.group(3)}",
        text,
    )
    text = re.sub(
        r"(?m)^(\s*layer_num:\s*)\d+\s*$",
        lambda m: f"{m.group(1)}{int(layer_num)}",
        text,
    )

    # Keep a unified model regardless of keyword; do not rewrite instruments by keyword.

    text = re.sub(
        r"(?m)^(\s*label_forward_shift:\s*)\d+\s*$",
        lambda m: f"{m.group(1)}{int(horizon)}",
        text,
    )
    text = re.sub(
        r"(?m)^(\s*label_reference_shift:\s*)\d+\s*$",
        lambda m: f"{m.group(1)}0",
        text,
    )
    if "label_forward_shift" not in text:
        text = text.replace(
            "    close_field: close",
            f"    close_field: close\n    label_forward_shift: {int(horizon)}\n    label_reference_shift: 0",
            1,
        )
    elif "label_reference_shift" not in text:
        text = text.replace(
            f"    label_forward_shift: {int(horizon)}",
            f"    label_forward_shift: {int(horizon)}\n    label_reference_shift: 0",
            1,
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_id = _trend_config_id(model_key, mode, horizon, keyword, fusion_type, layer_num)
    model_tag = f"{config_id}_{stamp}"
    TREND_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_path = TREND_MODEL_DIR / f"{model_tag}.pth"
    text = re.sub(
        r"(?m)^(\s*save_path:\s*).*$",
        lambda m: f"{m.group(1)}{save_path}",
        text,
    )

    TREND_JOB_DIR.mkdir(parents=True, exist_ok=True)
    out_cfg = TREND_JOB_DIR / f"{model_tag}.yaml"
    out_cfg.write_text(text, encoding="utf-8")
    return out_cfg, save_path, model_tag, config_id


def _parse_run_meta_from_log(log_path):
    exp_id = None
    run_id = None
    if not Path(log_path).exists():
        return exp_id, run_id
    pattern = re.compile(r"Recorder\s+([0-9a-f]{32})\s+starts running under Experiment\s+([^\s]+)")
    for line in Path(log_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pattern.search(line)
        if m:
            run_id = m.group(1)
            exp_id = m.group(2)
    return exp_id, run_id


def _resolve_artifact_root(exp_id, run_id):
    if not run_id:
        return None
    exp_text = str(exp_id).strip() if exp_id is not None else ""
    for root in _candidate_mlruns_roots():
        # Fast path: exact experiment id/path (works for numeric or named experiment dirs).
        if exp_text:
            direct = root / exp_text / str(run_id)
            meta = direct / "meta.yaml"
            if meta.exists():
                try:
                    text = meta.read_text(encoding="utf-8", errors="ignore")
                    m = re.search(r"^artifact_uri:\s*(.+)$", text, flags=re.MULTILINE)
                    if m:
                        uri = m.group(1).strip()
                        if uri.startswith("file://"):
                            p = Path(uri.replace("file://", "")).expanduser()
                            if p.exists():
                                return p
                except Exception:
                    pass
            direct_art = direct / "artifacts"
            if direct_art.exists():
                return direct_art

        # Fallback: locate run id under one level of experiments (avoid expensive recursive scan).
        candidates = list(root.glob(f"*/{run_id}/artifacts"))
        if candidates:
            return candidates[0]
    return None


def _postprocess_completed_job(job):
    if job.get("status") != "success":
        return
    if job.get("pred_path"):
        return
    exp_id, run_id = _parse_run_meta_from_log(job.get("log_path"))
    job["exp_id"] = exp_id
    job["run_id"] = run_id
    art_root = _resolve_artifact_root(exp_id, run_id)
    if art_root is None:
        return
    config_id = job.get("config_id") or job.get("model_tag", run_id)
    trend_store = _trend_artifact_store_dir(config_id)
    trend_store.mkdir(parents=True, exist_ok=True)
    job["artifact_dir"] = str(trend_store)

    src_pred = art_root / "pred.pkl"
    if not src_pred.exists():
        return
    TREND_PRED_DIR.mkdir(parents=True, exist_ok=True)
    dst_pred = TREND_PRED_DIR / f"{config_id}.pkl"
    try:
        shutil.copy2(src_pred, dst_pred)
        _safe_copy(src_pred, trend_store / "pred.pkl")
    except Exception:
        return
    job["pred_path"] = dst_pred
    src_label = art_root / "label.pkl"
    if src_label.exists():
        dst_label = TREND_PRED_DIR / f"{config_id}_label.pkl"
        if _safe_copy(src_label, dst_label):
            job["label_path"] = dst_label
        _safe_copy(src_label, trend_store / "label.pkl")
    src_model = art_root / "params.pkl"
    if src_model.exists():
        _safe_copy(src_model, TREND_MODEL_DIR / f"{config_id}.pkl")
        _safe_copy(src_model, trend_store / "params.pkl")

    named = _trend_named_paths(config_id)
    _safe_copy(src_pred, named["pred"])
    if src_label.exists():
        _safe_copy(src_label, named["label"])
    if src_model.exists():
        _safe_copy(src_model, named["model"])
    _write_named_meta(
        named["meta"],
        {
            "config_id": config_id,
            "exp_id": exp_id,
            "run_id": run_id,
            "artifact_root": str(art_root),
            "artifact_store": str(trend_store),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


def _refresh_trend_jobs():
    finished = []
    for job_id, job in TREND_TRAIN_JOBS.items():
        proc = job.get("process")
        if proc is None:
            if job.get("status") == "success":
                _postprocess_completed_job(job)
            continue
        ret = proc.poll()
        if ret is None:
            job["status"] = "running"
        else:
            job["status"] = "success" if ret == 0 else "failed"
            job["return_code"] = ret
            job["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            job.pop("process", None)
            _postprocess_completed_job(job)
            finished.append(job_id)
    return finished


def _list_recent_jobs(limit=8):
    _refresh_trend_jobs()
    jobs = []
    for job_id, job in sorted(TREND_TRAIN_JOBS.items(), key=lambda kv: kv[0], reverse=True):
        jobs.append(
            {
                "id": job_id,
                "model": job.get("model"),
                "mode": job.get("mode"),
                "fusion_type": job.get("fusion_type", "-"),
                "layer_num": job.get("layer_num", "-"),
                "keyword": job.get("keyword") or "-",
                "horizon": job.get("horizon"),
                "status": job.get("status"),
                "created_at": job.get("created_at"),
                "log_path": str(job.get("log_path", "")),
                "config_path": str(job.get("config_path", "")),
                "pred_path": str(job.get("pred_path", "")),
            }
        )
        if len(jobs) >= limit:
            break
    return jobs


def _start_trend_training(model_key, mode, horizon, keyword, fusion_type, layer_num):
    config_path, model_save_path, model_tag, config_id = _prepare_trend_config(
        model_key,
        mode,
        horizon,
        keyword,
        fusion_type=fusion_type,
        layer_num=layer_num,
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = TREND_JOB_DIR / f"{model_tag}.log"

    cmd = _resolve_qrun_command() + [str(config_path)]
    env = os.environ.copy()
    # Force mlflow artifacts into local project path.
    env["MLFLOW_TRACKING_URI"] = f"file://{QLIB_MLRUNS_DIR}"
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, cwd=str(QLIB_ROOT), stdout=f, stderr=subprocess.STDOUT, env=env)

    job_id = stamp + "_" + model_key
    TREND_TRAIN_JOBS[job_id] = {
        "process": proc,
        "model": model_key,
        "mode": mode,
        "horizon": int(horizon),
        "fusion_type": fusion_type,
        "layer_num": int(layer_num),
        "keyword": keyword,
        "status": "running",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "log_path": log_path,
        "config_path": config_path,
        "model_save_path": model_save_path,
        "model_tag": model_tag,
        "config_id": config_id,
    }
    return job_id, log_path, config_path


def _python_has_qlib(py_cmd):
    try:
        check = subprocess.run(
            [py_cmd, "-c", "import qlib"],
            cwd=str(QLIB_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return check.returncode == 0
    except Exception:
        return False


def _resolve_qrun_command():
    qrun_bin = os.environ.get("QRUN_BIN", "").strip()
    if qrun_bin:
        return [qrun_bin]

    found_qrun = shutil.which("qrun")
    if found_qrun:
        return [found_qrun]

    preferred_python = os.environ.get("QLIB_PYTHON", "").strip()
    py_candidates = [preferred_python] if preferred_python else []
    py_candidates += ["python3", "python"]

    for py in py_candidates:
        if not py:
            continue
        if _python_has_qlib(py):
            return [py, "-m", "qlib.workflow.cli", "workflow"]

    raise FileNotFoundError(
        "未找到可用的 qrun 命令，也未找到可导入 qlib 的 Python 解释器。"
        "请在启动 Flask 前激活 qlib 环境，或设置环境变量 QRUN_BIN/QLIB_PYTHON。"
    )


def _candidate_mlruns_roots():
    roots = []
    if QLIB_MLRUNS_DIR.exists():
        roots.append(QLIB_MLRUNS_DIR)
    for p in Path("/z5s").glob("*/home/*/qlib/mlruns"):
        if p.exists() and p not in roots:
            roots.append(p)
    return roots


def _iter_pred_files():
    files = []
    for root in _candidate_mlruns_roots():
        files.extend(root.rglob("pred.pkl"))
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def _match_pred_file(path, model_key, mode):
    run_name = path.parent.parent.name.lower()
    tokens = TREND_RUN_NAME_PATTERN.get(model_key, (model_key,))
    if not any(token in run_name for token in tokens):
        return False
    if mode == "tech_news":
        return "_news_" in run_name or run_name.endswith("_news")
    return "_news_" not in run_name and not run_name.endswith("_news")


def _find_latest_pred_file(model_key, mode, fusion_type, layer_num, horizon, keyword):
    config_id = _trend_config_id(model_key, mode, horizon, keyword, fusion_type, layer_num)
    cached_pred = _has_trend_cache(config_id)
    return Path(cached_pred) if cached_pred else None


def _find_label_file_for_pred(pred_path, config_id):
    pred_path = Path(pred_path)
    candidates = []
    candidates.append(_trend_artifact_store_dir(config_id) / "label.pkl")

    # 1) direct sibling in mlruns artifacts directory
    candidates.append(pred_path.parent / "label.pkl")

    # 2) local cached label
    # 2) named cache label
    named = _trend_named_paths(config_id)
    candidates.append(named["label"])

    # 3) fallback via named metadata artifact_root
    meta_path = named["meta"]
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            art_root = meta.get("artifact_root")
            if art_root:
                candidates.append(Path(art_root) / "label.pkl")
        except Exception:
            pass

    for p in candidates:
        if p.exists():
            return p
    return None


def _normalize_pred_df(pred_df):
    df = pred_df.copy()
    if isinstance(df, pd.Series):
        df = df.to_frame("score")
    if "score" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "score"})

    if isinstance(df.index, pd.MultiIndex):
        names = [str(n).lower() if n is not None else "" for n in df.index.names]
        dt_level = names.index("datetime") if "datetime" in names else 0
        ins_level = names.index("instrument") if "instrument" in names else 1
        out = df.reset_index()
        out.rename(columns={out.columns[dt_level]: "datetime", out.columns[ins_level]: "instrument"}, inplace=True)
    else:
        out = df.reset_index()
        if "datetime" not in out.columns:
            out.rename(columns={out.columns[0]: "datetime"}, inplace=True)
        if "instrument" not in out.columns:
            out["instrument"] = ""

    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out["date_display"] = out["datetime"].dt.strftime("%Y-%m-%d")
    out["instrument"] = out["instrument"].astype(str)
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out = out.dropna(subset=["datetime", "score"])
    return out


def _normalize_label_df(label_df):
    df = label_df.copy()
    if isinstance(df, pd.Series):
        df = df.to_frame("label")
    if "LABEL0" in df.columns:
        df = df.rename(columns={"LABEL0": "label"})
    elif "label" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "label"})

    if isinstance(df.index, pd.MultiIndex):
        names = [str(n).lower() if n is not None else "" for n in df.index.names]
        dt_level = names.index("datetime") if "datetime" in names else 0
        ins_level = names.index("instrument") if "instrument" in names else 1
        out = df.reset_index()
        out.rename(columns={out.columns[dt_level]: "datetime", out.columns[ins_level]: "instrument"}, inplace=True)
    else:
        out = df.reset_index()
        if "datetime" not in out.columns:
            out.rename(columns={out.columns[0]: "datetime"}, inplace=True)
        if "instrument" not in out.columns:
            out["instrument"] = ""

    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out["date_display"] = out["datetime"].dt.strftime("%Y-%m-%d")
    out["instrument"] = out["instrument"].astype(str)
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    out = out.dropna(subset=["datetime"])
    return out


def _format_return(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value) * 100:.3f}%"
    return value


@lru_cache(maxsize=4096)
def _get_close_by_code_date(code, date_text):
    df = load_stock_data(code)
    if df is None or df.empty:
        return None
    hit = df[df["date_display"] == date_text]
    if hit.empty:
        return _safe_float(df.iloc[-1].get("close"))
    return _safe_float(hit.iloc[-1].get("close"))


@lru_cache(maxsize=4096)
def _get_close_by_code_date_exact(code, date_text):
    df = load_stock_data(code)
    if df is None or df.empty:
        return None
    hit = df[df["date_display"] == date_text]
    if hit.empty:
        return None
    return _safe_float(hit.iloc[-1].get("close"))


def _build_prediction_rows(model_key, mode, keyword, horizon, fusion_type, layer_num):
    config_id = _trend_config_id(model_key, mode, horizon, keyword, fusion_type, layer_num)
    pred_path = _find_latest_pred_file(model_key, mode, fusion_type, layer_num, horizon, keyword)
    if pred_path is None:
        return [], None, "未找到该模型与模式对应的预测结果，请先执行模型训练。"

    try:
        pred_df = pd.read_pickle(pred_path)
        df = _normalize_pred_df(pred_df)
    except Exception as exc:
        return [], str(pred_path), f"读取预测文件失败：{exc}"

    label_path = _find_label_file_for_pred(pred_path, config_id)
    if label_path is not None:
        try:
            label_df = pd.read_pickle(label_path)
            labels = _normalize_label_df(label_df)[["datetime", "instrument", "label"]]
            df = df.merge(labels, on=["datetime", "instrument"], how="left")
        except Exception:
            pass

    code_filter = _resolve_keyword_to_codes(keyword)
    if code_filter:
        df = df[df["instrument"].isin(code_filter)]
    elif keyword:
        kw = keyword.lower()
        mapping = load_stock_mapping()
        matched_codes = [code for code, name in mapping.items() if kw in code.lower() or kw in name.lower()]
        if matched_codes:
            df = df[df["instrument"].isin(matched_codes)]

    if df.empty:
        return [], str(pred_path), "当前筛选条件下没有可展示的预测记录。"

    df = df.sort_values(["instrument", "datetime"])
    latest_df = df.groupby("instrument", as_index=False).tail(1).sort_values(["datetime", "score"], ascending=[False, False])

    focus_code = None
    if code_filter and len(code_filter) == 1:
        focus_code = code_filter[0]
    elif keyword:
        candidates = _resolve_keyword_to_codes(keyword)
        if len(candidates) == 1:
            focus_code = candidates[0]
    if not focus_code and not latest_df.empty:
        focus_code = str(latest_df.iloc[0]["instrument"])

    code_df = df[df["instrument"] == focus_code].sort_values("datetime")
    if code_df.empty:
        return [], str(pred_path), "未找到可用于构建测试集全量序列的预测记录。"
    mapping = load_stock_mapping()
    rows = []
    for _, row in code_df.iterrows():
        score = _safe_float(row["score"])
        real_ret = _safe_float(row.get("label"))
        base_date = row["date_display"]
        pred_date = _next_trade_day(base_date)
        rows.append(
            {
                "date": pred_date,
                "base_date": base_date,
                "code": focus_code,
                "name": mapping.get(focus_code, focus_code),
                "score": score,
                "real_return": _format_return(real_ret),
                "pred_return": _format_return(score),
                "real_return_val": real_ret,
                "pred_return_val": score,
                "point_type": "history" if real_ret is not None else "forecast",
            }
        )
    rows = sorted(rows, key=lambda x: x.get("date", ""))
    return rows, str(pred_path), None


def _build_prediction_chart(rows, keyword):
    if not rows:
        return None

    focus_code = None
    codes = _resolve_keyword_to_codes(keyword)
    if len(codes) == 1:
        focus_code = codes[0]
    if not focus_code:
        focus_code = rows[0].get("code")

    series_rows = [r for r in rows if r.get("code") == focus_code]
    if not series_rows:
        return None

    labels = [str(r.get("date", "")) for r in series_rows]
    baseline = [_safe_float(r.get("real_return_val")) for r in series_rows]
    predicted = [_safe_float(r.get("pred_return_val")) for r in series_rows]

    return {
        "code": focus_code,
        "name": series_rows[-1].get("name", focus_code),
        "labels": labels,
        "baseline": baseline,
        "predicted": predicted,
    }


def _parse_date_input(date_text, fallback):
    text = str(date_text or "").strip()
    if not text:
        text = fallback
    ts = pd.to_datetime(text, errors="coerce")
    if pd.isna(ts):
        ts = pd.to_datetime(fallback, errors="coerce")
    return ts.strftime("%Y-%m-%d")


def _prepare_backtest_config(
    model_key,
    mode,
    keyword,
    start_date,
    end_date,
    topk,
    n_drop,
    account,
    open_cost,
    close_cost,
    min_cost,
    fusion_type,
    layer_num,
    method_buy,
    method_sell,
    hold_thresh,
    only_tradable,
    forbid_all_trade_at_limit,
):
    base_cfg = TREND_MODEL_CONFIG.get(model_key)
    if base_cfg is None or not base_cfg.exists():
        raise FileNotFoundError(f"未找到模型配置文件: {base_cfg}")

    text = base_cfg.read_text(encoding="utf-8")
    if re.search(r"(?m)^experiment_name:\s*.+$", text):
        text = re.sub(r"(?m)^experiment_name:\s*.+$", f"experiment_name: {APP_EXPERIMENT_NAME}", text, count=1)
    else:
        text = f"experiment_name: {APP_EXPERIMENT_NAME}\n" + text
    use_news = mode == "tech_news"
    text = re.sub(
        r"(?m)^(\s*usenews:\s*)(true|false)\s*$",
        lambda m: f"{m.group(1)}{'true' if use_news else 'false'}",
        text,
    )
    text = re.sub(
        r"(?m)^(\s*d_news:\s*)\d+\s*$",
        lambda m: f"{m.group(1)}{512 if use_news else 0}",
        text,
    )
    text = re.sub(
        r"(?m)^(\s*fusion_type:\s*)(\S+)(.*)$",
        lambda m: f"{m.group(1)}{fusion_type}{m.group(3)}",
        text,
    )
    text = re.sub(
        r"(?m)^(\s*layer_num:\s*)\d+\s*$",
        lambda m: f"{m.group(1)}{int(layer_num)}",
        text,
    )

    selected_codes = _resolve_keyword_to_codes(keyword)
    if selected_codes:
        inline = ", ".join([f"'{c}'" for c in selected_codes[:50]])
        text = re.sub(
            r"(?m)^(\s*instruments:\s*).*$",
            f"\\1[{inline}]",
            text,
            count=1,
        )

    start_date = _parse_date_input(start_date, BACKTEST_DEFAULT_START)
    end_date = _parse_date_input(end_date, BACKTEST_DEFAULT_END)
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    # Only modify the backtest period in `port_analysis_config`.
    # Keep data_handler/train-valid range unchanged; otherwise training set may become empty.
    port_idx = text.find("port_analysis_config")
    if port_idx >= 0:
        head = text[:port_idx]
        tail = text[port_idx:]
        tail = re.sub(
            r"(?m)^(\s*start_time:\s*)\d{4}-\d{2}-\d{2}\s*$",
            lambda m: f"{m.group(1)}{start_date}",
            tail,
            count=1,
        )
        tail = re.sub(
            r"(?m)^(\s*end_time:\s*)\d{4}-\d{2}-\d{2}\s*$",
            lambda m: f"{m.group(1)}{end_date}",
            tail,
            count=1,
        )
        text = head + tail

    text = re.sub(
        r"(?m)^(\s*test:\s*)\[[^\]]+\]\s*$",
        lambda m: f"{m.group(1)}[{start_date}, {end_date}]",
        text,
        count=1,
    )

    text = re.sub(r"(?m)^(\s*topk:\s*)\d+\s*$", lambda m: f"{m.group(1)}{int(topk)}", text, count=1)
    text = re.sub(r"(?m)^(\s*n_drop:\s*)\d+\s*$", lambda m: f"{m.group(1)}{int(n_drop)}", text, count=1)
    text = re.sub(
        r"(?m)^(\s*method_buy:\s*)(\S+)\s*$",
        lambda m: f"{m.group(1)}{method_buy}",
        text,
        count=1,
    )
    text = re.sub(
        r"(?m)^(\s*method_sell:\s*)(\S+)\s*$",
        lambda m: f"{m.group(1)}{method_sell}",
        text,
        count=1,
    )
    text = re.sub(
        r"(?m)^(\s*hold_thresh:\s*)\d+\s*$",
        lambda m: f"{m.group(1)}{int(hold_thresh)}",
        text,
        count=1,
    )
    text = re.sub(
        r"(?m)^(\s*only_tradable:\s*)(true|false)\s*$",
        lambda m: f"{m.group(1)}{'true' if only_tradable else 'false'}",
        text,
        count=1,
    )
    text = re.sub(
        r"(?m)^(\s*forbid_all_trade_at_limit:\s*)(true|false)\s*$",
        lambda m: f"{m.group(1)}{'true' if forbid_all_trade_at_limit else 'false'}",
        text,
        count=1,
    )
    if "method_buy:" not in text:
        text = text.replace(
            f"            n_drop: {int(n_drop)}",
            (
                f"            n_drop: {int(n_drop)}\n"
                f"            method_sell: {method_sell}\n"
                f"            method_buy: {method_buy}\n"
                f"            hold_thresh: {int(hold_thresh)}\n"
                f"            only_tradable: {'true' if only_tradable else 'false'}\n"
                f"            forbid_all_trade_at_limit: {'true' if forbid_all_trade_at_limit else 'false'}"
            ),
            1,
        )
    text = re.sub(
        r"(?m)^(\s*account:\s*)\d+(\.\d+)?\s*$",
        lambda m: f"{m.group(1)}{float(account):.0f}",
        text,
        count=1,
    )
    text = re.sub(
        r"(?m)^(\s*open_cost:\s*)[0-9.]+\s*$",
        lambda m: f"{m.group(1)}{float(open_cost):.6f}",
        text,
        count=1,
    )
    text = re.sub(
        r"(?m)^(\s*close_cost:\s*)[0-9.]+\s*(#.*)?$",
        lambda m: f"{m.group(1)}{float(close_cost):.6f}",
        text,
        count=1,
    )
    text = re.sub(
        r"(?m)^(\s*min_cost:\s*)\d+(\.\d+)?\s*$",
        lambda m: f"{m.group(1)}{float(min_cost):.2f}",
        text,
        count=1,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_id = _backtest_config_id(
        model_key=model_key,
        mode=mode,
        keyword=keyword,
        start_date=start_date,
        end_date=end_date,
        topk=topk,
        n_drop=n_drop,
        account=account,
        open_cost=open_cost,
        close_cost=close_cost,
        min_cost=min_cost,
        fusion_type=fusion_type,
        layer_num=layer_num,
        method_buy=method_buy,
        method_sell=method_sell,
        hold_thresh=hold_thresh,
        only_tradable=only_tradable,
        forbid_all_trade_at_limit=forbid_all_trade_at_limit,
    )
    model_tag = f"{config_id}_{stamp}"
    BACKTEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_path = BACKTEST_MODEL_DIR / f"{model_tag}.pth"
    text = re.sub(
        r"(?m)^(\s*save_path:\s*).*$",
        lambda m: f"{m.group(1)}{save_path}",
        text,
        count=1,
    )

    BACKTEST_JOB_DIR.mkdir(parents=True, exist_ok=True)
    out_cfg = BACKTEST_JOB_DIR / f"{model_tag}.yaml"
    out_cfg.write_text(text, encoding="utf-8")
    return out_cfg, model_tag, start_date, end_date, config_id


def _safe_tail_lines(path, line_count=160):
    p = Path(path)
    if not p.exists():
        return ""
    try:
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return ""
    if len(lines) <= line_count:
        return "\n".join(lines)
    return "\n".join(lines[-line_count:])


def _start_backtest_job(
    model_key,
    mode,
    keyword,
    start_date,
    end_date,
    topk,
    n_drop,
    account,
    open_cost,
    close_cost,
    min_cost,
    fusion_type,
    layer_num,
    method_buy,
    method_sell,
    hold_thresh,
    only_tradable,
    forbid_all_trade_at_limit,
):
    config_path, model_tag, start_date, end_date, config_id = _prepare_backtest_config(
        model_key=model_key,
        mode=mode,
        keyword=keyword,
        start_date=start_date,
        end_date=end_date,
        topk=topk,
        n_drop=n_drop,
        account=account,
        open_cost=open_cost,
        close_cost=close_cost,
        min_cost=min_cost,
        fusion_type=fusion_type,
        layer_num=layer_num,
        method_buy=method_buy,
        method_sell=method_sell,
        hold_thresh=hold_thresh,
        only_tradable=only_tradable,
        forbid_all_trade_at_limit=forbid_all_trade_at_limit,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = BACKTEST_JOB_DIR / f"{model_tag}.log"
    cmd = _resolve_qrun_command() + [str(config_path)]
    env = os.environ.copy()
    # Force mlflow artifacts into local project path.
    env["MLFLOW_TRACKING_URI"] = f"file://{QLIB_MLRUNS_DIR}"
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, cwd=str(QLIB_ROOT), stdout=f, stderr=subprocess.STDOUT, env=env)

    job_id = f"{stamp}_{model_key}_bt"
    BACKTEST_JOBS[job_id] = {
        "process": proc,
        "status": "running",
        "model": model_key,
        "mode": mode,
        "keyword": keyword,
        "topk": int(topk),
        "n_drop": int(n_drop),
        "account": float(account),
        "open_cost": float(open_cost),
        "close_cost": float(close_cost),
        "min_cost": float(min_cost),
        "fusion_type": fusion_type,
        "layer_num": int(layer_num),
        "method_buy": method_buy,
        "method_sell": method_sell,
        "hold_thresh": int(hold_thresh),
        "only_tradable": bool(only_tradable),
        "forbid_all_trade_at_limit": bool(forbid_all_trade_at_limit),
        "start_date": start_date,
        "end_date": end_date,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "log_path": log_path,
        "config_path": config_path,
        "model_tag": model_tag,
        "config_id": config_id,
    }
    return job_id, log_path, config_path


def _load_backtest_report(art_root):
    if art_root is None:
        return None
    report_path = Path(art_root) / "portfolio_analysis" / "report_normal_1day.pkl"
    if not report_path.exists():
        return None
    try:
        return pd.read_pickle(report_path)
    except Exception:
        return None


def _build_backtest_result(report_df):
    if report_df is None or report_df.empty:
        return None, None
    df = report_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, errors="coerce")
        except Exception:
            return None, None
    df = df[~df.index.isna()].sort_index()
    if df.empty:
        return None, None

    daily = pd.to_numeric(df.get("return"), errors="coerce").fillna(0.0)
    if "cost" in df.columns:
        daily = daily - pd.to_numeric(df.get("cost"), errors="coerce").fillna(0.0)

    wealth = (1.0 + daily).cumprod()
    cum_curve = wealth - 1.0
    drawdown = wealth / wealth.cummax() - 1.0

    bench_curve = None
    if "bench" in df.columns:
        bench_daily = pd.to_numeric(df.get("bench"), errors="coerce").fillna(0.0)
        bench_curve = (1.0 + bench_daily).cumprod() - 1.0

    n = len(daily)
    cum_return = float(cum_curve.iloc[-1]) if n else 0.0
    annual_return = (1.0 + cum_return) ** (252 / max(n, 1)) - 1.0
    vol = float(daily.std(ddof=0) * np.sqrt(252)) if n else 0.0
    sharpe = float(daily.mean() / daily.std(ddof=0) * np.sqrt(252)) if n and daily.std(ddof=0) > 1e-12 else None
    max_drawdown = float(drawdown.min()) if n else 0.0
    win_rate = float((daily > 0).mean()) if n else 0.0

    metrics = {
        "累计收益率": _format_one_decimal(cum_return * 100),
        "年化收益率": _format_one_decimal(annual_return * 100),
        "最大回撤": _format_one_decimal(max_drawdown * 100),
        "Sharpe比率": _format_one_decimal(sharpe),
        "波动率(年化)": _format_one_decimal(vol * 100),
        "胜率": _format_one_decimal(win_rate * 100),
        "交易日数": int(n),
    }

    labels = [d.strftime("%Y-%m-%d") for d in df.index]
    chart = {
        "labels": labels,
        "strategy_cum": [round(float(x), 6) for x in cum_curve.tolist()],
        "benchmark_cum": [round(float(x), 6) for x in bench_curve.tolist()] if bench_curve is not None else [],
        "drawdown": [round(float(x), 6) for x in drawdown.tolist()],
        "daily_return": [round(float(x), 6) for x in daily.tolist()],
    }
    return metrics, chart


def _postprocess_backtest_job(job):
    if job.get("status") != "success":
        return
    if job.get("metrics") and job.get("chart"):
        return

    exp_id, run_id = _parse_run_meta_from_log(job.get("log_path"))
    job["exp_id"] = exp_id
    job["run_id"] = run_id
    art_root = _resolve_artifact_root(exp_id, run_id)
    if art_root is None:
        return
    job["artifact_root"] = str(art_root)
    config_id = job.get("config_id") or job.get("model_tag", run_id)
    backtest_store = _backtest_artifact_store_dir(config_id)
    backtest_store.mkdir(parents=True, exist_ok=True)
    job["artifact_dir"] = str(backtest_store)

    report_df = _load_backtest_report(art_root)
    metrics, chart = _build_backtest_result(report_df)
    if metrics:
        job["metrics"] = metrics
    if chart:
        job["chart"] = chart

    report_src = Path(art_root) / "portfolio_analysis" / "report_normal_1day.pkl"
    indicator_src = Path(art_root) / "portfolio_analysis" / "indicator_analysis_1day.pkl"
    _safe_copy(report_src, backtest_store / "portfolio_analysis" / "report_normal_1day.pkl")
    if indicator_src.exists():
        _safe_copy(indicator_src, backtest_store / "portfolio_analysis" / "indicator_analysis_1day.pkl")
    named = _backtest_named_paths(config_id)
    _safe_copy(report_src, named["report"])
    _safe_copy(indicator_src, named["indicator"])
    _write_named_meta(
        named["meta"],
        {
            "config_id": config_id,
            "exp_id": exp_id,
            "run_id": run_id,
            "artifact_root": str(art_root),
            "artifact_store": str(backtest_store),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


def _refresh_backtest_jobs():
    for _, job in BACKTEST_JOBS.items():
        proc = job.get("process")
        if proc is None:
            if job.get("status") == "success":
                _postprocess_backtest_job(job)
            continue
        ret = proc.poll()
        if ret is None:
            job["status"] = "running"
        else:
            job["status"] = "success" if ret == 0 else "failed"
            job["return_code"] = ret
            job["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            job.pop("process", None)
            _postprocess_backtest_job(job)


def _list_backtest_jobs(limit=10):
    _refresh_backtest_jobs()
    jobs = []
    for job_id, job in sorted(BACKTEST_JOBS.items(), key=lambda kv: kv[0], reverse=True):
        jobs.append(
            {
                "id": job_id,
                "model": job.get("model"),
                "mode": job.get("mode"),
                "keyword": job.get("keyword") or "-",
                "start_date": job.get("start_date"),
                "end_date": job.get("end_date"),
                "topk": job.get("topk"),
                "n_drop": job.get("n_drop"),
                "method_buy": job.get("method_buy"),
                "method_sell": job.get("method_sell"),
                "status": job.get("status"),
                "created_at": job.get("created_at"),
                "log_path": str(job.get("log_path", "")),
                "config_path": str(job.get("config_path", "")),
                "has_result": bool(job.get("metrics") and job.get("chart")),
            }
        )
        if len(jobs) >= limit:
            break
    return jobs


def _format_date_for_display(value):
    if pd.isna(value):
        return ""
    text = str(value)
    if text.isdigit() and len(text) >= 8:
        text = text[:8]
        return f"{text[0:4]}-{text[4:6]}-{text[6:8]}"
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return text
    return ts.strftime("%Y-%m-%d")


@lru_cache(maxsize=1)
def _load_trade_calendar():
    cal_file = STOCK_DATA_DIR / "calendars" / "day.txt"
    if not cal_file.exists():
        return []
    days = []
    for line in cal_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip()
        if not text:
            continue
        days.append(_format_date_for_display(text))
    return sorted(set([d for d in days if d]))


def _next_trade_day(date_text):
    ts = pd.to_datetime(date_text, errors="coerce")
    if pd.isna(ts):
        return ""

    target = ts.strftime("%Y-%m-%d")
    cal = _load_trade_calendar()
    for d in cal:
        if d > target:
            return d

    # fallback if calendar does not contain future sessions
    next_day = ts + pd.offsets.BDay(1)
    return next_day.strftime("%Y-%m-%d")


def load_stock_data(code):
    stock_code = code.upper().strip()
    parquet_path = STOCK_DATA_DIR / f"{stock_code}.pqt"
    if not parquet_path.exists():
        return None

    df = pd.read_parquet(parquet_path)
    if "date" in df.columns:
        date_series = df["date"].astype(str).str[:8]
        df["_date_sort"] = pd.to_datetime(date_series, errors="coerce")
    else:
        df["_date_sort"] = pd.NaT

    df = df.sort_values("_date_sort")
    df["date_display"] = df["date"].apply(_format_date_for_display) if "date" in df.columns else ""
    return df


def _format_one_decimal(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.1f}"
    return value


def _to_rounded_list(values):
    result = []
    for v in values:
        if v is None or pd.isna(v):
            result.append(None)
        else:
            result.append(round(float(v), 1))
    return result


def _extract_news_title_and_body(raw_text, title_hint):
    text = str(raw_text or "").strip()
    title = str(title_hint or "").strip()

    if title:
        return title, text

    if text.startswith("【") and "】" in text:
        end = text.find("】")
        extracted = text[1:end].strip()
        remain = text[end + 1 :].strip()
        if extracted:
            return extracted, remain

    for sep in ("：", ":"):
        if sep in text:
            left, right = text.split(sep, 1)
            left = left.strip("【】 ").strip()
            right = right.strip()
            if left:
                return left, right

    return "新闻详情", text


def load_related_news(stock_code):
    code = stock_code.upper().strip()
    matched = list(NEWS_DATA_DIR.glob(f"*_{code}.pqt"))
    if not matched:
        return []

    news_path = matched[0]
    df = pd.read_parquet(news_path)
    if df.empty:
        return []

    if "dt" in df.columns:
        dt_series = df["dt"].astype(str).str[:14]
        df["_dt_sort"] = pd.to_datetime(dt_series, format="%Y%m%d%H%M%S", errors="coerce")
        df["dt_display"] = df["_dt_sort"].dt.strftime("%Y-%m-%d %H:%M").fillna(df["dt"].astype(str))
    else:
        df["_dt_sort"] = pd.NaT
        df["dt_display"] = ""

    df = df.sort_values("_dt_sort", ascending=False).head(20)
    records = []
    for _, row in df.iterrows():
        raw_text = row.get("content", "")
        title_hint = row.get("title", "")
        title, body = _extract_news_title_and_body(raw_text, title_hint)
        preview = body[:110] + "..." if len(body) > 110 else body
        records.append(
            {
                "title": title,
                "preview": preview,
                "content": body if body else raw_text,
                "dt": row.get("dt_display", ""),
                "src": row.get("src", ""),
                "channel": row.get("channel", ""),
            }
        )
    return records


def _extract_technical_indicators_from_snapshot(snapshot_text):
    indicators = []
    in_metrics = False
    for raw in str(snapshot_text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("关键指标"):
            in_metrics = True
            continue
        if not in_metrics:
            continue
        if not line.startswith("- "):
            continue
        body = line[2:].strip()
        if ":" not in body:
            continue
        key, value = body.split(":", 1)
        indicators.append({"name": key.strip(), "value": value.strip()})
    return indicators


def _format_news_dt_display(raw_dt):
    text = str(raw_dt or "").strip()
    if text.isdigit() and len(text) >= 14:
        try:
            return datetime.strptime(text[:14], "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M")
        except Exception:
            return text
    return text


def _normalize_analysis_news_records(records):
    out = []
    for item in records or []:
        out.append(
            {
                "dt": _format_news_dt_display(item.get("dt", "")),
                "src": str(item.get("src", "")).strip(),
                "title": str(item.get("title", "")).strip(),
                "snippet": str(item.get("snippet", "")).strip(),
            }
        )
    return out


def _safe_float(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def _first_existing_column(columns, candidates):
    for col in candidates:
        if col in columns:
            return col
    return None


def _col_has_data(df, col):
    return col in df.columns and df[col].notna().sum() > 0


def get_screener_availability(snapshot_df):
    return {
        "market_cap": _col_has_data(snapshot_df, "market_cap"),
        "pe": _col_has_data(snapshot_df, "pe"),
        "pb": _col_has_data(snapshot_df, "pb"),
        "rsi": _col_has_data(snapshot_df, "RSI_14"),
        "macd_dif": _col_has_data(snapshot_df, "MACD_dif"),
        "macd_hist": _col_has_data(snapshot_df, "MACD_hist"),
        "volume": _col_has_data(snapshot_df, "mktTradeQty"),
        "turnover": _col_has_data(snapshot_df, "mktTradeTurnover"),
        "net_trade_flow": _col_has_data(snapshot_df, "net_trade_flow"),
        "net_order_flow": _col_has_data(snapshot_df, "net_order_flow"),
        "ma_trend": _col_has_data(snapshot_df, "ma_close_5")
        and _col_has_data(snapshot_df, "ma_close_20")
        and _col_has_data(snapshot_df, "ma_close_60"),
        "boll_state": _col_has_data(snapshot_df, "BOLL_upper")
        and _col_has_data(snapshot_df, "BOLL_lower")
        and _col_has_data(snapshot_df, "close"),
    }


@lru_cache(maxsize=1)
def build_screener_snapshot():
    mapping = load_stock_mapping()
    records = []

    for parquet_path in STOCK_DATA_DIR.glob("*.pqt"):
        code = parquet_path.stem
        try:
            schema_cols = set(pq.ParquetFile(parquet_path).schema.names)
            required = [c for c in SCREEN_COLUMNS if c in schema_cols]
            for cands in FINANCIAL_COLUMN_CANDIDATES.values():
                for c in cands:
                    if c in schema_cols and c not in required:
                        required.append(c)

            df = pd.read_parquet(parquet_path, columns=required if required else None)
            if df.empty:
                continue
            if "date" in df.columns:
                df = df.sort_values("date")
            latest = df.tail(1).iloc[0].to_dict()

            rec = {
                "code": code,
                "name": mapping.get(code, code),
                "date": _format_date_for_display(latest.get("date")),
                "open": _safe_float(latest.get("open")),
                "close": _safe_float(latest.get("close")),
                "ret_pct": _safe_float(latest.get("ret_pct")),
                "mktTradeQty": _safe_float(latest.get("mktTradeQty")),
                "mktTradeTurnover": _safe_float(latest.get("mktTradeTurnover")),
                "MACD_dif": _safe_float(latest.get("MACD_dif")),
                "MACD_dea": _safe_float(latest.get("MACD_dea")),
                "MACD_hist": _safe_float(latest.get("MACD_hist")),
                "RSI_14": _safe_float(latest.get("RSI_14")),
                "ma_close_5": _safe_float(latest.get("ma_close_5")),
                "ma_close_20": _safe_float(latest.get("ma_close_20")),
                "ma_close_60": _safe_float(latest.get("ma_close_60")),
                "BOLL_mid": _safe_float(latest.get("BOLL_mid")),
                "BOLL_upper": _safe_float(latest.get("BOLL_upper")),
                "BOLL_lower": _safe_float(latest.get("BOLL_lower")),
                "net_trade_flow": _safe_float(latest.get("net_trade_flow")),
                "net_order_flow": _safe_float(latest.get("net_order_flow")),
            }

            market_cap_col = _first_existing_column(latest.keys(), FINANCIAL_COLUMN_CANDIDATES["market_cap"])
            pe_col = _first_existing_column(latest.keys(), FINANCIAL_COLUMN_CANDIDATES["pe"])
            pb_col = _first_existing_column(latest.keys(), FINANCIAL_COLUMN_CANDIDATES["pb"])
            rec["market_cap"] = _safe_float(latest.get(market_cap_col)) if market_cap_col else None
            rec["pe"] = _safe_float(latest.get(pe_col)) if pe_col else None
            rec["pb"] = _safe_float(latest.get(pb_col)) if pb_col else None
            records.append(rec)
        except Exception:
            continue

    return pd.DataFrame(records)


def _apply_range_filter(df, col, min_val, max_val):
    if col not in df.columns or df[col].notna().sum() == 0:
        return df
    if min_val is not None:
        df = df[df[col] >= min_val]
    if max_val is not None:
        df = df[df[col] <= max_val]
    return df


def _parse_float_arg(args, key):
    raw = str(args.get(key, "")).strip()
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def filter_screener_df(args, snapshot_df=None, availability=None):
    df = (snapshot_df if snapshot_df is not None else build_screener_snapshot()).copy()
    if df.empty:
        return df
    availability = availability or get_screener_availability(df)

    keyword = str(args.get("keyword", "")).strip().lower()
    if keyword:
        df = df[
            df["code"].str.lower().str.contains(keyword, na=False)
            | df["name"].str.lower().str.contains(keyword, na=False)
        ]

    range_keys = [
        ("market_cap", "market_cap_min", "market_cap_max", "market_cap"),
        ("pe", "pe_min", "pe_max", "pe"),
        ("pb", "pb_min", "pb_max", "pb"),
        ("RSI_14", "rsi_min", "rsi_max", "rsi"),
        ("MACD_dif", "macd_dif_min", "macd_dif_max", "macd_dif"),
        ("MACD_hist", "macd_hist_min", "macd_hist_max", "macd_hist"),
        ("mktTradeQty", "volume_min", "volume_max", "volume"),
        ("mktTradeTurnover", "turnover_min", "turnover_max", "turnover"),
        ("net_trade_flow", "net_trade_flow_min", "net_trade_flow_max", "net_trade_flow"),
        ("net_order_flow", "net_order_flow_min", "net_order_flow_max", "net_order_flow"),
    ]
    for col, min_k, max_k, avail_key in range_keys:
        if not availability.get(avail_key, False):
            continue
        df = _apply_range_filter(df, col, _parse_float_arg(args, min_k), _parse_float_arg(args, max_k))

    ma_trend = str(args.get("ma_trend", "")).strip()
    if availability.get("ma_trend", False) and ma_trend == "bullish":
        df = df[(df["ma_close_5"] > df["ma_close_20"]) & (df["ma_close_20"] > df["ma_close_60"])]
    elif availability.get("ma_trend", False) and ma_trend == "ma5_above_ma20":
        df = df[df["ma_close_5"] > df["ma_close_20"]]

    boll_state = str(args.get("boll_state", "")).strip()
    if availability.get("boll_state", False) and boll_state == "above_upper":
        df = df[df["close"] > df["BOLL_upper"]]
    elif availability.get("boll_state", False) and boll_state == "below_lower":
        df = df[df["close"] < df["BOLL_lower"]]
    elif availability.get("boll_state", False) and boll_state == "between":
        df = df[(df["close"] >= df["BOLL_lower"]) & (df["close"] <= df["BOLL_upper"])]

    sort_by = str(args.get("sort_by", "mktTradeTurnover")).strip()
    sort_order = str(args.get("sort_order", "desc")).strip()
    if sort_by not in df.columns:
        sort_by = "mktTradeTurnover"
    df = df.sort_values(sort_by, ascending=(sort_order == "asc"), na_position="last")
    return df


def build_screener_chart(stock_code, window=60):
    df = load_stock_data(stock_code)
    if df is None or df.empty:
        return None

    cols = ["date_display", "open", "close", "tradePriceMax", "tradePriceMin", "RSI_14"]
    cols = [c for c in cols if c in df.columns]
    cdf = df[cols].tail(window).copy()
    if cdf.empty:
        return None

    candles = []
    for _, row in cdf.iterrows():
        candles.append(
            {
                "date": row.get("date_display", ""),
                "open": _safe_float(row.get("open")),
                "close": _safe_float(row.get("close")),
                "high": _safe_float(row.get("tradePriceMax")),
                "low": _safe_float(row.get("tradePriceMin")),
                "rsi": _safe_float(row.get("RSI_14")),
            }
        )
    return candles


@app.route("/index")
@role_required("user")
def index():
    return render_template("index.html")


@app.route("/trend_predict", methods=["GET", "POST"])
@role_required("user")
def trend_predict():
    form_data = {
        "keyword": "",
        "model": "alstm",
        "mode": "tech",
        "horizon": NEXT_DAY_HORIZON,
        "fusion_type": "decoder",
        "layer_num": 6,
    }
    action_result = None
    prediction_rows = []
    prediction_error = None
    prediction_source = None
    prediction_chart = None

    if request.method == "POST":
        form_data["keyword"] = str(request.form.get("keyword", "")).strip()
        form_data["model"] = str(request.form.get("model", "alstm")).strip().lower()
        form_data["mode"] = str(request.form.get("mode", "tech")).strip()
        form_data["fusion_type"] = str(request.form.get("fusion_type", "decoder")).strip().lower()

        # fixed to next-day forecasting
        form_data["horizon"] = NEXT_DAY_HORIZON
        layer_raw = str(request.form.get("layer_num", "6")).strip()
        try:
            form_data["layer_num"] = max(1, min(12, int(layer_raw)))
        except ValueError:
            form_data["layer_num"] = 6

        valid_models = {m[0] for m in TREND_MODEL_OPTIONS}
        valid_modes = {m[0] for m in TREND_MODE_OPTIONS}
        valid_fusions = {f[0] for f in TREND_FUSION_OPTIONS}
        if form_data["model"] not in valid_models:
            form_data["model"] = "alstm"
        if form_data["mode"] not in valid_modes:
            form_data["mode"] = "tech"
        if form_data["fusion_type"] not in valid_fusions:
            form_data["fusion_type"] = "decoder"

        if form_data["mode"] != "tech_news":
            form_data["fusion_type"] = "add"
            form_data["layer_num"] = 1
        elif form_data["fusion_type"] in {"add", "concat"}:
            form_data["layer_num"] = 1

        action = str(request.form.get("action", "predict")).strip().lower()
        action_text = "模型训练" if action == "train" else "趋势预测"
        mode_text = "仅技术特征" if form_data["mode"] == "tech" else "技术+新闻特征"
        model_text = dict(TREND_MODEL_OPTIONS).get(form_data["model"], form_data["model"])
        fusion_text = (
            f"融合={form_data['fusion_type']}, 层数={form_data['layer_num']}"
            if form_data["mode"] == "tech_news"
            else "融合=无（仅技术）"
        )

        if action == "train":
            try:
                config_id = _trend_config_id(
                    form_data["model"],
                    form_data["mode"],
                    form_data["horizon"],
                    form_data["keyword"],
                    form_data["fusion_type"],
                    form_data["layer_num"],
                )
                cached_pred = _has_trend_cache(config_id)
                if cached_pred:
                    action_result = {
                        "action_text": action_text,
                        "summary": (
                            f"检测到相同配置已有训练结果，已复用缓存：标的={form_data['keyword'] or '全部'}，"
                            f"模型={model_text}，模式={mode_text}（{fusion_text}）。"
                        ),
                        "command_hint": f"缓存预测文件: {cached_pred}",
                        "extra": f"配置ID: {config_id}",
                    }
                    jobs = _list_recent_jobs(limit=8)
                    trend_pending_jobs = any(
                        j.get("status") == "running" or (j.get("status") == "success" and not j.get("pred_path"))
                        for j in jobs
                    )
                    return render_template(
                        "trend_predict.html",
                        form_data=form_data,
                        model_options=TREND_MODEL_OPTIONS,
                        mode_options=TREND_MODE_OPTIONS,
                        fusion_options=TREND_FUSION_OPTIONS,
                        action_result=action_result,
                        jobs=jobs,
                        prediction_rows=prediction_rows,
                        prediction_error=prediction_error,
                        prediction_source=prediction_source,
                        prediction_chart=prediction_chart,
                        has_pending_trend_jobs=trend_pending_jobs,
                    )

                job_id, log_path, config_path = _start_trend_training(
                    form_data["model"],
                    form_data["mode"],
                    form_data["horizon"],
                    form_data["keyword"],
                    form_data["fusion_type"],
                    form_data["layer_num"],
                )
                action_result = {
                    "action_text": action_text,
                    "summary": (
                        f"训练任务已提交：标的={form_data['keyword'] or '全部'}，"
                        f"模型={model_text}，模式={mode_text}（{fusion_text}），预测时长=下一交易日。"
                    ),
                    "command_hint": f"qrun {config_path}",
                    "extra": f"任务ID: {job_id} | 日志: {log_path}",
                }
            except Exception as exc:
                action_result = {
                    "action_text": action_text,
                    "summary": f"训练任务提交失败：{exc}",
                    "command_hint": "",
                    "extra": "",
                }
        else:
            prediction_rows, prediction_source, prediction_error = _build_prediction_rows(
                form_data["model"],
                form_data["mode"],
                form_data["keyword"],
                form_data["horizon"],
                form_data["fusion_type"],
                form_data["layer_num"],
            )
            action_result = {
                "action_text": action_text,
                "summary": (
                    f"已按条件检索预测结果：标的={form_data['keyword'] or '全部'}，"
                    f"模型={model_text}，模式={mode_text}（{fusion_text}），预测时长=下一交易日。"
                ),
                "command_hint": f"数据来源: {prediction_source}" if prediction_source else "",
                "extra": prediction_error or "展示测试集真实收益率(label.pkl)与预测收益率(score)。",
            }
            prediction_chart = _build_prediction_chart(prediction_rows, form_data["keyword"])

    jobs = _list_recent_jobs(limit=8)
    trend_pending_jobs = any(
        j.get("status") == "running" or (j.get("status") == "success" and not j.get("pred_path")) for j in jobs
    )

    return render_template(
        "trend_predict.html",
        form_data=form_data,
        model_options=TREND_MODEL_OPTIONS,
        mode_options=TREND_MODE_OPTIONS,
        fusion_options=TREND_FUSION_OPTIONS,
        action_result=action_result,
        jobs=jobs,
        prediction_rows=prediction_rows,
        prediction_error=prediction_error,
        prediction_source=prediction_source,
        prediction_chart=prediction_chart,
        has_pending_trend_jobs=trend_pending_jobs,
    )


@app.route("/backtest", methods=["GET", "POST"])
@role_required("user")
def backtest_page():
    form_data = {
        "keyword": "",
        "model": "alstm",
        "mode": "tech_news",
        "start_date": BACKTEST_DEFAULT_START,
        "end_date": BACKTEST_DEFAULT_END,
        "topk": 30,
        "n_drop": 5,
        "account": 100000000,
        "open_cost": 0.000085,
        "close_cost": 0.001085,
        "min_cost": 5,
        "fusion_type": "decoder",
        "layer_num": 6,
        "method_buy": "top",
        "method_sell": "bottom",
        "hold_thresh": 1,
        "only_tradable": False,
        "forbid_all_trade_at_limit": True,
    }
    action_result = None
    selected_job_id = str(request.args.get("job_id", "")).strip()

    if request.method == "POST":
        form_data["keyword"] = str(request.form.get("keyword", "")).strip()
        form_data["model"] = str(request.form.get("model", "alstm")).strip().lower()
        form_data["mode"] = str(request.form.get("mode", "tech_news")).strip()
        form_data["start_date"] = _parse_date_input(request.form.get("start_date"), BACKTEST_DEFAULT_START)
        form_data["end_date"] = _parse_date_input(request.form.get("end_date"), BACKTEST_DEFAULT_END)
        form_data["fusion_type"] = str(request.form.get("fusion_type", "decoder")).strip().lower()
        form_data["method_buy"] = str(request.form.get("method_buy", "top")).strip().lower()
        form_data["method_sell"] = str(request.form.get("method_sell", "bottom")).strip().lower()
        form_data["only_tradable"] = str(request.form.get("only_tradable", "")).strip() == "on"
        form_data["forbid_all_trade_at_limit"] = str(request.form.get("forbid_all_trade_at_limit", "")).strip() != ""

        try:
            form_data["topk"] = max(1, int(request.form.get("topk", "30")))
        except ValueError:
            form_data["topk"] = 30
        try:
            form_data["n_drop"] = max(0, int(request.form.get("n_drop", "5")))
        except ValueError:
            form_data["n_drop"] = 5
        if form_data["n_drop"] >= form_data["topk"]:
            form_data["n_drop"] = max(0, form_data["topk"] - 1)

        try:
            form_data["account"] = max(1.0, float(request.form.get("account", "100000000")))
        except ValueError:
            form_data["account"] = 100000000.0
        try:
            form_data["open_cost"] = max(0.0, float(request.form.get("open_cost", "0.000085")))
        except ValueError:
            form_data["open_cost"] = 0.000085
        try:
            form_data["close_cost"] = max(0.0, float(request.form.get("close_cost", "0.001085")))
        except ValueError:
            form_data["close_cost"] = 0.001085
        try:
            form_data["min_cost"] = max(0.0, float(request.form.get("min_cost", "5")))
        except ValueError:
            form_data["min_cost"] = 5.0
        try:
            form_data["layer_num"] = max(1, min(12, int(request.form.get("layer_num", "6"))))
        except ValueError:
            form_data["layer_num"] = 6
        try:
            form_data["hold_thresh"] = max(1, int(request.form.get("hold_thresh", "1")))
        except ValueError:
            form_data["hold_thresh"] = 1

        valid_models = {m[0] for m in TREND_MODEL_OPTIONS}
        valid_modes = {m[0] for m in TREND_MODE_OPTIONS}
        valid_fusions = {f[0] for f in TREND_FUSION_OPTIONS}
        if form_data["model"] not in valid_models:
            form_data["model"] = "alstm"
        if form_data["mode"] not in valid_modes:
            form_data["mode"] = "tech_news"
        if form_data["fusion_type"] not in valid_fusions:
            form_data["fusion_type"] = "decoder"
        if form_data["method_buy"] not in {"top", "random"}:
            form_data["method_buy"] = "top"
        if form_data["method_sell"] not in {"bottom", "random"}:
            form_data["method_sell"] = "bottom"
        if form_data["mode"] != "tech_news":
            form_data["fusion_type"] = "add"
            form_data["layer_num"] = 1
        elif form_data["fusion_type"] in {"add", "concat"}:
            form_data["layer_num"] = 1

        try:
            config_id = _backtest_config_id(
                model_key=form_data["model"],
                mode=form_data["mode"],
                keyword=form_data["keyword"],
                start_date=form_data["start_date"],
                end_date=form_data["end_date"],
                topk=form_data["topk"],
                n_drop=form_data["n_drop"],
                account=form_data["account"],
                open_cost=form_data["open_cost"],
                close_cost=form_data["close_cost"],
                min_cost=form_data["min_cost"],
                fusion_type=form_data["fusion_type"],
                layer_num=form_data["layer_num"],
                method_buy=form_data["method_buy"],
                method_sell=form_data["method_sell"],
                hold_thresh=form_data["hold_thresh"],
                only_tradable=form_data["only_tradable"],
                forbid_all_trade_at_limit=form_data["forbid_all_trade_at_limit"],
            )
            cached_report = _has_backtest_cache(config_id)
            if not cached_report:
                cached_report = _materialize_backtest_cache_from_trend(form_data, config_id)
            if cached_report:
                report_df = pd.read_pickle(cached_report)
                metrics, backtest_chart = _build_backtest_result(report_df)
                cache_job_id = f"cache_{config_id}"
                BACKTEST_JOBS[cache_job_id] = {
                    "status": "cached",
                    "model": form_data["model"],
                    "mode": form_data["mode"],
                    "keyword": form_data["keyword"],
                    "topk": form_data["topk"],
                    "n_drop": form_data["n_drop"],
                    "method_buy": form_data["method_buy"],
                    "method_sell": form_data["method_sell"],
                    "start_date": form_data["start_date"],
                    "end_date": form_data["end_date"],
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "log_path": "",
                    "config_path": "",
                    "metrics": metrics,
                    "chart": backtest_chart,
                    "config_id": config_id,
                }
                selected_job_id = cache_job_id
                action_result = {
                    "summary": (
                        f"检测到相同配置已有回测结果，已直接复用缓存："
                        f"区间={form_data['start_date']}~{form_data['end_date']}。"
                    ),
                    "command_hint": f"缓存回测文件: {cached_report}",
                    "extra": f"配置ID: {config_id}",
                }
                jobs = _list_backtest_jobs(limit=10)
                selected_job = BACKTEST_JOBS.get(selected_job_id)
                log_text = _safe_tail_lines(selected_job.get("log_path")) if selected_job else ""
                return render_template(
                    "backtest.html",
                    form_data=form_data,
                    model_options=TREND_MODEL_OPTIONS,
                    mode_options=TREND_MODE_OPTIONS,
                    fusion_options=TREND_FUSION_OPTIONS,
                    buy_method_options=BACKTEST_BUY_METHOD_OPTIONS,
                    sell_method_options=BACKTEST_SELL_METHOD_OPTIONS,
                    action_result=action_result,
                    jobs=jobs,
                    selected_job_id=selected_job_id,
                    selected_job=selected_job,
                    selected_job_running=False,
                    metrics=metrics,
                    backtest_chart=backtest_chart,
                    log_text=log_text,
                )

            job_id, log_path, config_path = _start_backtest_job(
                model_key=form_data["model"],
                mode=form_data["mode"],
                keyword=form_data["keyword"],
                start_date=form_data["start_date"],
                end_date=form_data["end_date"],
                topk=form_data["topk"],
                n_drop=form_data["n_drop"],
                account=form_data["account"],
                open_cost=form_data["open_cost"],
                close_cost=form_data["close_cost"],
                min_cost=form_data["min_cost"],
                fusion_type=form_data["fusion_type"],
                layer_num=form_data["layer_num"],
                method_buy=form_data["method_buy"],
                method_sell=form_data["method_sell"],
                hold_thresh=form_data["hold_thresh"],
                only_tradable=form_data["only_tradable"],
                forbid_all_trade_at_limit=form_data["forbid_all_trade_at_limit"],
            )
            selected_job_id = job_id
            action_result = {
                "summary": (
                    f"回测任务已提交：模型={form_data['model']}，模式={form_data['mode']}，"
                    f"区间={form_data['start_date']}~{form_data['end_date']}，"
                    f"topk={form_data['topk']}，n_drop={form_data['n_drop']}，"
                    f"buy={form_data['method_buy']}，sell={form_data['method_sell']}。"
                ),
                "command_hint": f"qrun {config_path}",
                "extra": f"任务ID: {job_id} | 日志: {log_path}",
            }
        except Exception as exc:
            action_result = {
                "summary": f"回测任务提交失败：{exc}",
                "command_hint": "",
                "extra": "",
            }

    jobs = _list_backtest_jobs(limit=10)
    if not selected_job_id:
        for item in jobs:
            if item.get("status") in {"running", "success"}:
                selected_job_id = item["id"]
                break

    selected_job = BACKTEST_JOBS.get(selected_job_id)
    metrics = selected_job.get("metrics") if selected_job else None
    backtest_chart = selected_job.get("chart") if selected_job else None
    log_text = _safe_tail_lines(selected_job.get("log_path")) if selected_job else ""
    selected_job_running = bool(
        selected_job
        and (
            selected_job.get("status") == "running"
            or (selected_job.get("status") == "success" and not (metrics and backtest_chart))
        )
    )

    return render_template(
        "backtest.html",
        form_data=form_data,
        model_options=TREND_MODEL_OPTIONS,
        mode_options=TREND_MODE_OPTIONS,
        fusion_options=TREND_FUSION_OPTIONS,
        buy_method_options=BACKTEST_BUY_METHOD_OPTIONS,
        sell_method_options=BACKTEST_SELL_METHOD_OPTIONS,
        action_result=action_result,
        jobs=jobs,
        selected_job_id=selected_job_id,
        selected_job=selected_job,
        selected_job_running=selected_job_running,
        metrics=metrics,
        backtest_chart=backtest_chart,
        log_text=log_text,
    )


@app.route("/backtest/job/<job_id>/log")
@role_required("user")
def backtest_job_log(job_id):
    _refresh_backtest_jobs()
    job = BACKTEST_JOBS.get(job_id)
    if not job:
        return {"ok": False, "error": "job not found"}, 404
    payload = {
        "ok": True,
        "job_id": job_id,
        "status": job.get("status"),
        "log": _safe_tail_lines(job.get("log_path")),
        "has_result": bool(job.get("metrics") and job.get("chart")),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return payload


@app.route("/stock_list")
@role_required("user")
def stock_list():
    keyword = request.args.get("keyword", "").strip()
    keyword_lower = keyword.lower()

    records = []
    mapping = load_stock_mapping()
    for code, name in sorted(mapping.items()):
        if keyword and keyword_lower not in code.lower() and keyword_lower not in name.lower():
            continue
        if not (STOCK_DATA_DIR / f"{code}.pqt").exists():
            continue
        records.append({"code": code, "name": name})

    return render_template("stock_list.html", stocks=records, keyword=keyword)


@app.route("/stock_screener")
@role_required("user")
def stock_screener():
    industry_code = str(request.args.get("industry_code", "801011.SI")).strip().upper() or "801011.SI"
    input_trade_date = str(request.args.get("trade_date", "")).strip()
    extra_news_text = str(request.args.get("news_text", "")).strip()

    return render_template(
        "stock_screener.html",
        form_data={
            "industry_code": industry_code,
            "trade_date": input_trade_date,
            "resolved_trade_date": resolve_trade_date(STOCK_DATA_DIR, input_trade_date),
            "news_text": extra_news_text,
        },
        llm_env={
            "provider": os.getenv("LLM_PROVIDER", ""),
            "model": os.getenv("LLM_MODEL", ""),
            "base_url": os.getenv("OPENAI_BASE_URL", ""),
        },
    )


def _run_industry_analysis_job(job_id, payload):
    def on_update(stage, value):
        with INDUSTRY_ANALYSIS_LOCK:
            job = INDUSTRY_ANALYSIS_JOBS.get(job_id)
            if not job:
                return
            job["stages"][stage] = {"status": "done", "value": value}
            job["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        result = run_industry_debate_with_progress(
            industry_code=payload["industry_code"],
            trade_date=payload["trade_date"],
            level_file=INDUSTRY_LEVEL_FILE,
            feature_root=STOCK_DATA_DIR,
            news_root=NEWS_DATA_DIR,
            news_csv_root=NEWS_CSV_DIR,
            news_text=payload["news_text"],
            news_limit=3,
            on_update=on_update,
        )
        with INDUSTRY_ANALYSIS_LOCK:
            job = INDUSTRY_ANALYSIS_JOBS.get(job_id)
            if job:
                job["status"] = "done"
                job["meta"] = {
                    "industry_code": result.get("industry_code", ""),
                    "industry_name": result.get("industry_name", ""),
                    "trade_date": result.get("trade_date", ""),
                    "technical_indicators": _extract_technical_indicators_from_snapshot(result.get("market_snapshot", "")),
                    "news_records": _normalize_analysis_news_records(result.get("news_records", [])),
                }
                job["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except Exception as exc:
        with INDUSTRY_ANALYSIS_LOCK:
            job = INDUSTRY_ANALYSIS_JOBS.get(job_id)
            if job:
                job["status"] = "failed"
                job["error"] = str(exc)
                job["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@app.route("/stock_screener/start", methods=["POST"])
@role_required("user")
def stock_screener_start():
    payload = {
        "industry_code": str(request.form.get("industry_code", "801011.SI")).strip().upper() or "801011.SI",
        "trade_date": str(request.form.get("trade_date", "")).strip(),
        "news_text": str(request.form.get("news_text", "")).strip(),
    }
    resolved_trade_date = resolve_trade_date(STOCK_DATA_DIR, payload["trade_date"])
    industry_name = load_industry_name(INDUSTRY_LEVEL_FILE, payload["industry_code"])

    technical_indicators = []
    news_records = []
    try:
        market_snapshot = build_market_snapshot(
            feature_root=STOCK_DATA_DIR,
            industry_code=payload["industry_code"],
            industry_name=industry_name,
            trade_date=resolved_trade_date,
        )
        technical_indicators = _extract_technical_indicators_from_snapshot(market_snapshot)
    except Exception:
        technical_indicators = []

    try:
        _, raw_news_records = build_news_snapshot(
            news_root=NEWS_DATA_DIR,
            news_csv_root=NEWS_CSV_DIR,
            industry_code=payload["industry_code"],
            industry_name=industry_name,
            trade_date=resolved_trade_date,
            extra_news_text=payload["news_text"],
            limit=3,
        )
        news_records = _normalize_analysis_news_records(raw_news_records)
    except Exception:
        news_records = []

    job_id = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:8]
    init_stages = {
        "technical_report": {"status": "running", "value": ""},
        "news_report": {"status": "running", "value": ""},
        "bull_case": {"status": "pending", "value": ""},
        "bear_case": {"status": "pending", "value": ""},
        "final_decision": {"status": "pending", "value": {}},
    }
    with INDUSTRY_ANALYSIS_LOCK:
        INDUSTRY_ANALYSIS_JOBS[job_id] = {
            "status": "running",
            "payload": payload,
            "stages": init_stages,
            "meta": {
                "industry_code": payload["industry_code"],
                "industry_name": industry_name,
                "trade_date": resolved_trade_date,
                "technical_indicators": technical_indicators,
                "news_records": news_records,
            },
            "error": "",
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    th = threading.Thread(target=_run_industry_analysis_job, args=(job_id, payload), daemon=True)
    th.start()
    return {"ok": True, "job_id": job_id}


@app.route("/stock_screener/status/<job_id>")
@role_required("user")
def stock_screener_status(job_id):
    with INDUSTRY_ANALYSIS_LOCK:
        job = INDUSTRY_ANALYSIS_JOBS.get(job_id)
        if not job:
            return {"ok": False, "error": "job not found"}, 404
        return {
            "ok": True,
            "job_id": job_id,
            "status": job.get("status"),
            "meta": job.get("meta", {}),
            "stages": job.get("stages", {}),
            "error": job.get("error", ""),
            "updated_at": job.get("updated_at", ""),
        }


@app.route("/stock_screener/export")
@role_required("user")
def stock_screener_export():
    return {"ok": False, "message": "选股导出已下线，请使用行业智能分析页面。"}, 410


@app.route("/stock_detail/<stock_code>")
@role_required("user")
def stock_detail(stock_code):
    view_type = request.args.get("view", "history").strip().lower()
    if view_type not in {"history", "factors", "news"}:
        view_type = "history"

    df = load_stock_data(stock_code)
    if df is None or df.empty:
        flash(f"未找到股票 {stock_code} 对应的数据文件。", "error")
        return redirect(url_for("stock_list"))

    stock_name = load_stock_mapping().get(stock_code.upper(), stock_code.upper())

    history_cols = list(HISTORY_COLUMN_LABELS.keys())
    factor_cols = list(FACTOR_COLUMN_LABELS.keys())
    selected_cols = history_cols if view_type == "history" else factor_cols
    existing_cols = [c for c in selected_cols if c in df.columns]
    label_map = HISTORY_COLUMN_LABELS if view_type == "history" else FACTOR_COLUMN_LABELS
    detail_columns = [{"key": c, "label": label_map.get(c, c)} for c in existing_cols]

    if view_type == "history":
        data_view = df.copy()
    elif view_type == "factors":
        data_view = df.tail(60).copy()
    else:
        data_view = df.iloc[0:0].copy()

    data_view = data_view.sort_values("_date_sort", ascending=False)
    rows = data_view[existing_cols].replace({np.nan: None}).to_dict(orient="records")
    for row in rows:
        for key, value in row.items():
            if key != "date_display":
                row[key] = _format_one_decimal(value)

    factor_chart = None
    news_items = []
    if view_type == "factors":
        chart_df = df.sort_values("_date_sort").tail(120).copy()
        labels = chart_df["date_display"].tolist()
        factor_chart = {
            "labels": labels,
            "window_size": 60,
            "price": {
                "open": _to_rounded_list(chart_df["open"].tolist()) if "open" in chart_df.columns else [],
                "close": _to_rounded_list(chart_df["close"].tolist()) if "close" in chart_df.columns else [],
                "high": _to_rounded_list(chart_df["tradePriceMax"].tolist()) if "tradePriceMax" in chart_df.columns else [],
                "low": _to_rounded_list(chart_df["tradePriceMin"].tolist()) if "tradePriceMin" in chart_df.columns else [],
                "volume": _to_rounded_list(chart_df["mktTradeQty"].tolist()) if "mktTradeQty" in chart_df.columns else [],
            },
            "indicators": {
                "MACD_dif": _to_rounded_list(chart_df["MACD_dif"].tolist()) if "MACD_dif" in chart_df.columns else [],
                "MACD_dea": _to_rounded_list(chart_df["MACD_dea"].tolist()) if "MACD_dea" in chart_df.columns else [],
                "MACD_hist": _to_rounded_list(chart_df["MACD_hist"].tolist()) if "MACD_hist" in chart_df.columns else [],
                "KDJ_K": _to_rounded_list(chart_df["KDJ_K"].tolist()) if "KDJ_K" in chart_df.columns else [],
                "KDJ_D": _to_rounded_list(chart_df["KDJ_D"].tolist()) if "KDJ_D" in chart_df.columns else [],
                "KDJ_J": _to_rounded_list(chart_df["KDJ_J"].tolist()) if "KDJ_J" in chart_df.columns else [],
                "RSI_14": _to_rounded_list(chart_df["RSI_14"].tolist()) if "RSI_14" in chart_df.columns else [],
                "BOLL_upper": _to_rounded_list(chart_df["BOLL_upper"].tolist()) if "BOLL_upper" in chart_df.columns else [],
                "BOLL_mid": _to_rounded_list(chart_df["BOLL_mid"].tolist()) if "BOLL_mid" in chart_df.columns else [],
                "BOLL_lower": _to_rounded_list(chart_df["BOLL_lower"].tolist()) if "BOLL_lower" in chart_df.columns else [],
            },
        }
    elif view_type == "news":
        news_items = load_related_news(stock_code)

    return render_template(
        "stock_detail.html",
        stock_code=stock_code.upper(),
        stock_name=stock_name,
        view_type=view_type,
        detail_columns=detail_columns,
        rows=rows,
        factor_chart=factor_chart,
        news_items=news_items,
    )


if __name__ == "__main__":
    app.run(debug=True)
