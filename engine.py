"""
下半场实验用横截面动量引擎（独立目录，不修改外部历史产物）。

固定口径: t 收盘信号、t+1 开盘成交、月度调仓、单边 7bp、目标波动 10%、杠杆 ≤1.0。
支持: 排名缓冲、最小持有期、换手门槛、切换成本惩罚、多/空、可扩展资产池。
"""

import hashlib
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import guards

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
COST_RATE = 0.0007
VOL_DAYS = 20
VOL60_DAYS = 60
TARGET_VOL = 0.10
TRADING_DAYS = 252
SKIP_DAYS = 21
CHINA_TZ = timezone(timedelta(hours=8))

BASE10 = ["SPY", "QQQ", "IEF", "TLT", "GLD", "DBC", "VNQ", "EFA", "EEM", "UUP"]
EXTRA = ["HYG", "LQD", "EMB", "TIP", "SLV", "DBA", "IWM"]


def load_data(tickers=None) -> dict[str, pd.DataFrame]:
    out = {}
    for t in (tickers or BASE10):
        p = DATA_DIR / f"{t}.csv"
        if not p.exists():
            raise FileNotFoundError(f"缺少 {p}，先运行 fetch_extra.py")
        df = pd.read_csv(p, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        out[t] = df.sort_values("date").drop_duplicates("date")
    return out


def build_panels(data):
    calendar = pd.DatetimeIndex(sorted(set().union(*[set(df["date"]) for df in data.values()])))
    opens, closes = {}, {}
    for t, df in data.items():
        s = df.set_index("date")
        opens[t] = s["open"].reindex(calendar)
        closes[t] = s["close"].reindex(calendar)
    return pd.DataFrame(opens, index=calendar), pd.DataFrame(closes, index=calendar)


def month_end_flags(calendar):
    flags = np.zeros(len(calendar), dtype=bool)
    for i in range(len(calendar) - 1):
        flags[i] = calendar[i].month != calendar[i + 1].month
    flags[-1] = True
    return flags


def common_start(close_df, mom_days=252):
    ready = close_df.notna().cumsum().min(axis=1) >= (mom_days + 1)
    return close_df.index[ready][0]


# ---------------------------------------------------------------- 选股
def make_target(close_df, ret_f, i, k, mom_days, use_filter, weight, target_vol,
                prev_target=None, hold_counts=None, buffer=0, min_hold=0,
                switch_hurdle=0.0, long_short=False):
    cols = list(close_df.columns)
    empty = pd.Series(0.0, index=cols)
    diag = {"reason": "", "selected": [], "est_vol": None, "scale": 0.0,
            "sum_weight": 0.0, "n_selected": 0}
    if i - mom_days < 0:
        diag["reason"] = "insufficient_history"
        return empty, diag

    prev_target = prev_target if prev_target is not None else pd.Series(0.0, index=cols)
    hold_counts = hold_counts or {}

    mom = close_df.iloc[i - SKIP_DAYS] / close_df.iloc[i - mom_days] - 1.0
    vol60 = ret_f.iloc[i - VOL60_DAYS + 1: i + 1].std(ddof=1)
    valid = mom.notna() & (vol60 > 0)
    if use_filter:
        valid = valid & (mom > 0)
    if not valid.any():
        diag["reason"] = "no_valid_asset"
        return empty, diag

    if long_short:
        ranked = sorted([c for c in cols if bool(valid[c])], key=lambda c: (-float(mom[c]), c))
        if len(ranked) < 2 * k:
            diag["reason"] = "too_few_assets"
            return empty, diag
        longs, shorts = ranked[:k], ranked[-k:]
        w = pd.Series(0.0, index=cols)
        w[longs] = 0.5 / len(longs)
        w[shorts] = -0.5 / len(shorts)
        picked = longs + shorts
        diag["selected"] = picked
    else:
        rank = mom.where(valid).rank(ascending=False, method="min")
        incumbents = [c for c in cols if float(prev_target.get(c, 0.0)) > 1e-9 and bool(valid.get(c, False))]

        if switch_hurdle > 0.0:
            score = mom.astype(float).copy()
            for c in incumbents:
                score[c] = float(score[c]) + switch_hurdle
            ranked = sorted([c for c in cols if bool(valid[c])], key=lambda c: (-float(score[c]), c))
            picked = ranked[:k]
        else:
            retained = []
            incumbents.sort(key=lambda c: float(rank.get(c, np.inf)))
            for c in incumbents:
                if len(retained) >= k:
                    break
                forced = min_hold > 0 and hold_counts.get(c, 0) < min_hold
                if forced or float(rank.get(c, np.inf)) <= k + buffer:
                    retained.append(c)
            picked = list(retained)
            ordered = sorted([c for c in cols if bool(valid[c])], key=lambda c: (float(rank.get(c, np.inf)), c))
            for c in ordered:
                if len(picked) >= k:
                    break
                if c not in picked:
                    picked.append(c)
        if not picked:
            diag["reason"] = "no_pick"
            return empty, diag
        w = pd.Series(0.0, index=cols)
        if weight == "equal":
            w[picked] = 1.0 / len(picked)
        else:
            iv = pd.Series({c: 1.0 / float(vol60[c]) for c in picked})
            w[picked] = (iv / iv.sum()).reindex(picked).to_numpy()

    win = ret_f.iloc[i - VOL_DAYS + 1: i + 1][list(w[w != 0].index)]
    cov = win.cov().to_numpy(dtype=float) * TRADING_DAYS
    wv = w[w != 0].to_numpy(dtype=float)
    var = float(wv @ cov @ wv)
    est = math.sqrt(max(var, 0.0)) if np.isfinite(var) else float("nan")
    if not np.isfinite(est) or est <= 1e-8:
        single = float((win.std(ddof=1) * math.sqrt(TRADING_DAYS)).mean())
        est = single if np.isfinite(single) else target_vol
    scale = min(target_vol / est, 1.0) if est > 0 else 1.0
    w = w * scale

    diag.update({"reason": "ok", "selected": picked, "est_vol": float(est),
                 "scale": float(scale), "sum_weight": float(w.abs().sum()),
                 "n_selected": len(picked)})
    return w, diag


# ---------------------------------------------------------------- 回测
@dataclass
class Result:
    nav: pd.DataFrame
    trades: pd.DataFrame
    signals: pd.DataFrame
    metrics: dict = field(default_factory=dict)


def run(data, k=3, mom_days=252, use_filter=False, weight="equal",
        target_vol=TARGET_VOL, buffer=0, min_hold=0, threshold=0.0,
        switch_hurdle=0.0, long_short=False, start=None, end=None):
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
    hold_counts = {}
    pending = None
    rows, trades, signals = [], [], []

    def execute(target_w, d, px, signal_date):
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
        gross = float(trade_val.abs().sum())
        if long_short:
            entry = exit_ = 0.0
            adjust = gross
        else:
            prev_held = last_target > 1e-9
            new_held = w > 1e-9
            entry = float(trade_val[~prev_held & new_held].abs().sum())
            exit_ = float(trade_val[prev_held & ~new_held].abs().sum())
            adjust = float(trade_val[prev_held & new_held].abs().sum())
        cash = cash - float(trade_val.sum()) - cost
        for t in cols:
            shares[t] = target_val[t] / px[t] if bool(tradable[t]) else 0.0
        trades.append({"signal_date": pd.Timestamp(signal_date).date(),
                       "exec_date": pd.Timestamp(d).date(),
                       "gross_traded": gross, "cost": float(cost),
                       "entry_traded": entry, "exit_traded": exit_,
                       "adjust_traded": adjust, "nav_before": float(pv),
                       "n_holdings_after": int((w != 0).sum())})
        last_target = w.copy()

    for i in range(start_idx, end_idx + 1):
        d = cal[i]
        if pending is not None:
            sig_date, target = pending
            # 与 xsec 同口径：缺开盘价就失败，不拿当日收盘价顶替（前视）
            op_checked = guards.check_open_prices(d.date(), open_df.loc[d].to_dict())
            op = pd.Series(op_checked).reindex(close_df.columns)
            execute(target, d, op, sig_date)
            pending = None

        px_close = close_ff.loc[d]
        nav = cash + float((shares * px_close).sum())
        rows.append({"date": d, "nav": nav, "cash": cash,
                     "exposure": float((shares * px_close).sum()) / nav if nav > 0 else 0.0,
                     "n_holdings": int((shares > 1e-9).sum())})

        if flags[i] and i + 1 <= end_idx:
            target, diag = make_target(
                close_df, ret_f, i, k, mom_days, use_filter, weight, target_vol,
                prev_target=last_target, hold_counts=hold_counts,
                buffer=buffer, min_hold=min_hold,
                switch_hurdle=switch_hurdle, long_short=long_short)

            skipped = False
            gross_before = 0.0
            if threshold > 0 and len(diag["selected"]) and nav > 0:
                cur_w = (shares * px_close) / nav
                gross_before = float((target - cur_w).abs().sum())
                if gross_before < threshold:
                    target = cur_w.copy()
                    skipped = True

            row = {"date": d, "selected": ",".join(diag["selected"]),
                   "n_selected": diag["n_selected"], "reason": diag["reason"],
                   "est_vol": diag["est_vol"], "scale": diag["scale"],
                   "sum_weight": float(target.abs().sum()),
                   "skipped": skipped, "gross_before": gross_before,
                   "buffer": buffer, "min_hold": min_hold,
                   "threshold": threshold, "switch_hurdle": switch_hurdle,
                   "long_short": long_short}
            row.update({f"w_{c}": float(target[c]) for c in cols})
            signals.append(row)

            # 持有期计数: 本次仍持有则 +1, 新进为 1, 卖出清零
            new_counts = {}
            for c in cols:
                if float(target.get(c, 0.0)) != 0.0:
                    new_counts[c] = (hold_counts.get(c, 0) + 1) if float(last_target.get(c, 0.0)) != 0.0 else 1
            hold_counts = new_counts

            if not skipped:
                pending = (d, target)

    nav_df = pd.DataFrame(rows).set_index("date")
    nav_df["nav"] = nav_df["nav"] / nav_df["nav"].iloc[0] * 100.0
    tr = pd.DataFrame(trades)
    sig = pd.DataFrame(signals)
    return Result(nav=nav_df, trades=tr, signals=sig,
                  metrics=compute_metrics(nav_df, tr))


def compute_metrics(nav_df, trades):
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
    return {
        "start": str(nav.index[0].date()), "end": str(nav.index[-1].date()),
        "years": round(float(years), 4), "trading_days": int(len(ret)),
        "cagr": float(cagr), "ann_vol": float(vol), "sharpe": float(sharpe),
        "max_drawdown": float(mdd), "monthly_win_rate": win_rate,
        "annual_turnover": float(gross / mean_nav / years) if years > 0 else float("nan"),
        "turnover_entry": float(entry / mean_nav / years) if years > 0 else float("nan"),
        "turnover_exit": float(exit_ / mean_nav / years) if years > 0 else float("nan"),
        "turnover_adjust": float(adjust / mean_nav / years) if years > 0 else float("nan"),
        "cost_total_bp": float(cost / mean_nav * 1e4) if mean_nav > 0 else float("nan"),
        "cost_annual_bp": float(cost / mean_nav * 1e4 / years) if years > 0 else float("nan"),
        "final_nav": float(nav.iloc[-1]),
        "avg_holdings": float(nav_df["n_holdings"].mean()),
        "n_trades": int(len(trades)),
    }


def run_benchmark(data, start, end, tickers=None) -> pd.Series:
    tickers = tickers or list(data.keys())
    open_df, close_df = build_panels(data)
    cal = close_df.loc[(close_df.index >= pd.Timestamp(start)) &
                       (close_df.index <= pd.Timestamp(end))].index
    px = close_df.loc[cal].ffill()
    exec_i = 1 if len(cal) > 1 else 0
    exec_date = cal[exec_i]
    n = len(tickers)
    target = 100.0 / n
    cost = target * n * COST_RATE
    cash = 100.0 - target * n - cost
    # 与 xsec 同口径：基准建仓价缺失直接失败
    entry = guards.check_open_prices(exec_date.date(), open_df.loc[exec_date].to_dict())
    shares = pd.Series({t: target / entry[t] for t in tickers})
    vals = []
    for i, d in enumerate(cal):
        vals.append(100.0 if i < exec_i else cash + float((shares * px.loc[d]).sum()))
    return pd.Series(vals, index=cal, name="bench_nav")


def simple_metrics(nav):
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


def active_stats(strat_ret, bench_ret):
    """OOS 年化波动、基准波动、波动差、年化跟踪误差、相关性。"""
    df = pd.concat([strat_ret, bench_ret], axis=1, join="inner").dropna()
    s, b = df.iloc[:, 0], df.iloc[:, 1]
    diff = s - b
    return {
        "strategy_vol": float(s.std(ddof=1) * math.sqrt(TRADING_DAYS)),
        "benchmark_vol": float(b.std(ddof=1) * math.sqrt(TRADING_DAYS)),
        "vol_diff": float((s.std(ddof=1) - b.std(ddof=1)) * math.sqrt(TRADING_DAYS)),
        "tracking_error": float(diff.std(ddof=1) * math.sqrt(TRADING_DAYS)),
        "corr": float(s.corr(b)),
        "ann_return_diff": float((s.mean() - b.mean()) * TRADING_DAYS),
    }


def head_info():
    try:
        p = ROOT / ".git" / "HEAD"
        raw = p.read_text(encoding="utf-8").strip()
        if raw.startswith("ref:"):
            ref = ROOT / ".git" / raw.split(":", 1)[1].strip()
            return ref.read_text(encoding="utf-8").strip() if ref.exists() else raw
        return raw
    except Exception:
        return "n/a"


def meta_block(script_path=None):
    script = Path(script_path) if script_path else Path(__file__).resolve()
    return {
        "builtAt": datetime.now(CHINA_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "head": head_info(),
        "script": script.name,
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest()[:16],
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
