"""
横截面动量（cross-sectional momentum）引擎。

口径与第一版时序动量引擎保持一致:
  * t 日收盘信号, t+1 开盘成交
  * 单边 7bp (5bp 佣金 + 2bp 滑点)
  * 月度调仓, 目标波动 10%, 杠杆上限 1.0, 现金 0 收益
  * auto_adjust=True 的复权 OHLC

本文件只做计算, 不写报告; 产物由 run_all.py 生成。
"""

import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"

UNIVERSE = ["SPY", "QQQ", "IEF", "TLT", "GLD", "DBC", "VNQ", "EFA", "EEM", "UUP"]
COST_RATE = 0.0005 + 0.0002
VOL_DAYS = 20
VOL60_DAYS = 60
TARGET_VOL = 0.10
TRADING_DAYS = 252
SKIP_DAYS = 21  # 12-1 动量跳过最近 1 个月
CHINA_TZ = timezone(timedelta(hours=8))


# ---------------------------------------------------------------- 数据

def load_data() -> dict[str, pd.DataFrame]:
    out = {}
    for t in UNIVERSE:
        df = pd.read_csv(DATA_DIR / f"{t}.csv", parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.sort_values("date").drop_duplicates("date")
        out[t] = df
    return out


def build_panels(data: dict[str, pd.DataFrame]):
    calendar = pd.DatetimeIndex(sorted(set().union(*[set(df["date"]) for df in data.values()])))
    opens, closes = {}, {}
    for t, df in data.items():
        s = df.set_index("date")
        opens[t] = s["open"].reindex(calendar)
        closes[t] = s["close"].reindex(calendar)
    return pd.DataFrame(opens, index=calendar), pd.DataFrame(closes, index=calendar)


def month_end_flags(calendar: pd.DatetimeIndex) -> np.ndarray:
    flags = np.zeros(len(calendar), dtype=bool)
    for i in range(len(calendar) - 1):
        flags[i] = calendar[i].month != calendar[i + 1].month
    flags[-1] = True
    return flags


def common_start(close_df: pd.DataFrame, mom_days: int = 252) -> pd.Timestamp:
    """全池具备 mom_days + 1 个有效收盘的首日, 与第一版同起点。"""
    ready = close_df.notna().cumsum().min(axis=1) >= (mom_days + 1)
    return close_df.index[ready][0]


# ---------------------------------------------------------------- 目标权重

def make_target(close_df, ret_f, i, k, mom_days, use_filter, weight, target_vol):
    """在 i 日收盘计算目标权重; 返回 (weights, 诊断 dict)。"""
    cols = list(close_df.columns)
    empty = pd.Series(0.0, index=cols)
    if i - mom_days < 0:
        return empty, {"reason": "insufficient_history", "selected": [],
                       "est_vol": None, "scale": 0.0, "sum_weight": 0.0}

    cur = close_df.iloc[i]
    skip_px = close_df.iloc[i - SKIP_DAYS]
    base_px = close_df.iloc[i - mom_days]
    mom = skip_px / base_px - 1.0
    vol60 = ret_f.iloc[i - VOL60_DAYS + 1: i + 1].std(ddof=1)
    valid = mom.notna() & base_px.notna() & (base_px > 0) & (vol60 > 0)

    if not valid.any():
        return empty, {"reason": "no_valid_asset", "selected": [],
                       "est_vol": None, "scale": 0.0, "sum_weight": 0.0}

    # 横截面排序: 动量降序, 同分用代码字典序保证确定性
    ranked = sorted([c for c in cols if bool(valid[c])],
                    key=lambda c: (-float(mom[c]), c))
    picked = ranked[:k]
    if use_filter:
        picked = [c for c in picked if float(mom[c]) > 0]
    if not picked:
        return empty, {"reason": "filtered_all_out", "selected": [],
                       "est_vol": None, "scale": 0.0, "sum_weight": 0.0}

    if weight == "equal":
        w = pd.Series({c: 1.0 / len(picked) for c in picked})
    else:  # inv_vol
        iv = pd.Series({c: 1.0 / float(vol60[c]) for c in picked})
        w = iv / iv.sum()

    win = ret_f.iloc[i - VOL_DAYS + 1: i + 1][picked]
    cov = win.cov().to_numpy() * TRADING_DAYS
    wv = w.reindex(picked).to_numpy(dtype=float)
    var = float(wv @ cov @ wv)
    est = math.sqrt(max(var, 0.0)) if np.isfinite(var) else float("nan")
    if not np.isfinite(est) or est <= 1e-8:
        single = float((win.std(ddof=1) * math.sqrt(TRADING_DAYS)).mean())
        est = single if np.isfinite(single) else target_vol
    scale = min(target_vol / est, 1.0) if est > 0 else 1.0

    weights = pd.Series(0.0, index=cols)
    weights[picked] = w.reindex(picked).to_numpy(dtype=float) * scale
    return weights, {
        "reason": "ok", "selected": picked, "est_vol": float(est),
        "scale": float(scale), "sum_weight": float(weights.sum()),
    }


# ---------------------------------------------------------------- 回测

@dataclass
class Result:
    nav: pd.DataFrame
    trades: pd.DataFrame
    signals: pd.DataFrame
    metrics: dict = field(default_factory=dict)


def run(data, k: int = 3, mom_days: int = 252, use_filter: bool = False,
        weight: str = "equal", target_vol: float = TARGET_VOL,
        start=None, end=None, exec_mode: str = "next_open") -> Result:
    """exec_mode:
      next_open —— 正常口径: t 收盘信号, t+1 开盘成交
      close     —— 变异测试用: t 收盘信号, t 收盘成交(前视), 断言必须变红
    """
    open_df, close_df = build_panels(data)
    cal = close_df.index
    close_ff = close_df.ffill()
    ret_f = close_df.pct_change(fill_method=None).ffill()
    cols = list(close_df.columns)

    def snap(ts, method="pad"):
        idx = int(cal.get_indexer([pd.Timestamp(ts)], method=method)[0])
        return max(0, idx)

    start_idx = snap(start) if start else 0
    end_idx = snap(end) if end else len(cal) - 1
    flags = month_end_flags(cal)

    cash, shares = 100.0, pd.Series(0.0, index=cols)
    last_target = pd.Series(0.0, index=cols)
    pending = None
    rows, trades, signals = [], [], []

    def execute(target_w, d, px, px_source, signal_date):
        nonlocal cash, shares, last_target
        cur_val = pd.Series({t: shares[t] * px[t] for t in cols})
        tradable = px.notna() & (px > 0)
        w = target_w.where(tradable, 0.0)
        pv = cash + float(cur_val.sum())
        cost, target_val = 0.0, pd.Series(0.0, index=cols)
        for _ in range(4):
            target_val = w * (pv - cost)
            cost = float((target_val - cur_val).abs().sum()) * COST_RATE
        trade_val = target_val - cur_val
        prev_held = last_target > 1e-9
        new_held = w > 1e-9
        entry = float(trade_val[~prev_held & new_held].abs().sum())
        exit_ = float(trade_val[prev_held & ~new_held].abs().sum())
        adjust = float(trade_val[prev_held & new_held].abs().sum())
        cash = cash - float(trade_val.sum()) - cost
        for t in cols:
            shares[t] = target_val[t] / px[t] if bool(tradable[t]) else 0.0
        trades.append({
            "signal_date": pd.Timestamp(signal_date).date(),
            "exec_date": pd.Timestamp(d).date(),
            "px_source": px_source,
            "gross_traded": float(trade_val.abs().sum()),
            "cost": float(cost),
            "entry_traded": entry, "exit_traded": exit_, "adjust_traded": adjust,
            "nav_before": float(pv),
            "n_holdings_after": int((w > 1e-9).sum()),
        })
        last_target = w.copy()

    for i in range(start_idx, end_idx + 1):
        d = cal[i]

        if pending is not None:
            sig_date, target = pending
            op = open_df.loc[d]
            op = op.where(op.notna() & (op > 0), close_ff.loc[d])
            execute(target, d, op, "open", sig_date)
            pending = None

        px_close = close_ff.loc[d]
        nav = cash + float((shares * px_close).sum())
        rows.append({"date": d, "nav": nav, "cash": cash,
                     "exposure": float((shares * px_close).sum()) / nav if nav > 0 else 0.0,
                     "n_holdings": int((shares > 1e-9).sum())})

        if flags[i] and i + 1 <= end_idx:
            target, diag = make_target(close_df, ret_f, i, k, mom_days,
                                       use_filter, weight, target_vol)
            row = {"date": d, "selected": ",".join(diag["selected"]),
                   "n_selected": len(diag["selected"]), "reason": diag["reason"],
                   "est_vol": diag["est_vol"], "scale": diag["scale"],
                   "sum_weight": float(target.sum())}
            row.update({f"w_{c}": float(target[c]) for c in cols})
            signals.append(row)

            if exec_mode == "close":
                execute(target, d, close_ff.loc[d], "close", d)
            else:
                pending = (d, target)

    nav_df = pd.DataFrame(rows).set_index("date")
    nav_df["nav"] = nav_df["nav"] / nav_df["nav"].iloc[0] * 100.0
    tr = pd.DataFrame(trades)
    sig = pd.DataFrame(signals)
    return Result(nav=nav_df, trades=tr, signals=sig,
                  metrics=compute_metrics(nav_df, tr))


# ---------------------------------------------------------------- 指标

def compute_metrics(nav_df: pd.DataFrame, trades: pd.DataFrame) -> dict:
    nav = nav_df["nav"]
    ret = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1 if years > 0 else float("nan")
    vol = ret.std(ddof=1) * math.sqrt(TRADING_DAYS)
    sharpe = ret.mean() / ret.std(ddof=1) * math.sqrt(TRADING_DAYS) if ret.std(ddof=1) > 0 else float("nan")
    mdd = (nav / nav.cummax() - 1).min()
    monthly = nav.resample("ME").last().pct_change().dropna()
    win_rate = float((monthly > 0).mean()) if len(monthly) else float("nan")

    mean_nav = float(nav.mean())
    if len(trades):
        gross = float(trades["gross_traded"].sum())
        cost = float(trades["cost"].sum())
        entry = float(trades["entry_traded"].sum())
        exit_ = float(trades["exit_traded"].sum())
        adjust = float(trades["adjust_traded"].sum())
    else:
        gross = cost = entry = exit_ = adjust = 0.0
    turnover = gross / mean_nav / years if years > 0 else float("nan")

    return {
        "start": str(nav.index[0].date()), "end": str(nav.index[-1].date()),
        "years": round(float(years), 4), "trading_days": int(len(ret)),
        "cagr": float(cagr), "ann_vol": float(vol), "sharpe": float(sharpe),
        "max_drawdown": float(mdd), "monthly_win_rate": win_rate,
        "annual_turnover": float(turnover),
        "turnover_entry": entry / mean_nav / years if years > 0 else float("nan"),
        "turnover_exit": exit_ / mean_nav / years if years > 0 else float("nan"),
        "turnover_adjust": adjust / mean_nav / years if years > 0 else float("nan"),
        "cost_total_bp": cost / mean_nav * 1e4 if mean_nav > 0 else float("nan"),
        "cost_annual_bp": cost / mean_nav * 1e4 / years if years > 0 else float("nan"),
        "final_nav": float(nav.iloc[-1]),
        "n_trades": int(len(trades)),
    }


def run_benchmark(data, start, end) -> pd.Series:
    """等权买入持有, 与第一版相同: 样本第 2 个交易日开盘一次性买入, 收 7bp。"""
    open_df, close_df = build_panels(data)
    cal = close_df.loc[(close_df.index >= pd.Timestamp(start)) &
                       (close_df.index <= pd.Timestamp(end))].index
    op = open_df.loc[cal].ffill()
    px = close_df.loc[cal].ffill()
    exec_i = 1 if len(cal) > 1 else 0
    exec_date = cal[exec_i]
    n = len(UNIVERSE)
    pv0 = 100.0
    target = pv0 / n
    cost = target * n * COST_RATE
    cash = pv0 - target * n - cost
    shares = pd.Series({t: target / op.loc[exec_date, t] for t in UNIVERSE})
    vals = []
    for i, d in enumerate(cal):
        vals.append(pv0 if i < exec_i else cash + float((shares * px.loc[d]).sum()))
    return pd.Series(vals, index=cal, name="bench_nav")


def simple_metrics(nav: pd.Series) -> dict:
    nav = nav.dropna()
    ret = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return {
        "start": str(nav.index[0].date()), "end": str(nav.index[-1].date()),
        "years": round(float(years), 4),
        "cagr": float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1),
        "ann_vol": float(ret.std(ddof=1) * math.sqrt(TRADING_DAYS)),
        "sharpe": float(ret.mean() / ret.std(ddof=1) * math.sqrt(TRADING_DAYS)),
        "max_drawdown": float((nav / nav.cummax() - 1).min()),
        "monthly_win_rate": float((nav.resample("ME").last().pct_change().dropna() > 0).mean()),
        "final_nav": float(nav.iloc[-1]),
    }


def rolling_median_sharpe(nav: pd.Series, t0, t1):
    """训练段内 252 日滚动净夏普, 在月末采样; 返回 (中位数, 最差, 样本数)。"""
    ret = nav.pct_change().dropna()
    roll = ret.rolling(TRADING_DAYS)
    sh = roll.mean() / roll.std(ddof=1) * math.sqrt(TRADING_DAYS)
    # ret 丢掉首日, 所以 flag 也要从第 2 个交易日对齐
    flags = month_end_flags(nav.index)[1:]
    sample = sh[(sh.index > pd.Timestamp(t0)) & (sh.index <= pd.Timestamp(t1)) & flags].dropna()
    if not len(sample):
        return float("nan"), float("nan"), 0
    return float(sample.median()), float(sample.min()), int(len(sample))


# ---------------------------------------------------------------- 元信息

def head_info() -> str:
    try:
        p = ROOT / ".git" / "HEAD"
        raw = p.read_text(encoding="utf-8").strip()
        if raw.startswith("ref:"):
            ref = ROOT / ".git" / raw.split(":", 1)[1].strip()
            return ref.read_text(encoding="utf-8").strip() if ref.exists() else raw
        return raw
    except Exception:
        return "n/a"


def meta_block() -> dict:
    script = Path(__file__).resolve()
    return {
        "builtAt": datetime.now(CHINA_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "head": head_info(),
        "script": script.name,
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest()[:16],
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
