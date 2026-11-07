"""实验用引擎变体：在 xsec 之上增加 opt-in 的换手带 / 排名缓冲 / 强制权重。

生产引擎 xsec.py 不动。参数取默认值（band=0, rank_buffer=0, force_weight=None）时
本文件必须与 xsec 逐位等价——实验入口会先用 baseline 变体断言这一点。

锁定口径未动：t 收盘信号、t+1 开盘成交、月度调仓、单边 7bp、
目标波动 10%、杠杆 ≤1.0、现金 0 收益。
"""

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import xsec  # noqa: E402

TRADING_DAYS = xsec.TRADING_DAYS
COST_RATE = xsec.COST_RATE
SKIP_DAYS = xsec.SKIP_DAYS
VOL60_DAYS = xsec.VOL60_DAYS
VOL_DAYS = xsec.VOL_DAYS


def make_target(close_df, ret_f, i, k, mom_days, use_filter, weight, target_vol,
                held=(), rank_buffer=0):
    """在 i 日收盘计算目标权重；rank_buffer>0 时现任只要还在前 K+buffer 就不换。"""
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

    ranked = sorted([c for c in cols if bool(valid[c])],
                    key=lambda c: (-float(mom[c]), c))
    if rank_buffer > 0 and held:
        keep = [c for c in ranked[:k + rank_buffer] if c in held]
        new = [c for c in ranked[:k] if c not in keep]
        picked = (keep + new)[:k]
    else:
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


@dataclass
class Result:
    nav: pd.DataFrame
    trades: pd.DataFrame
    signals: pd.DataFrame
    metrics: dict = field(default_factory=dict)
    n_skips: int = 0


def run(data, k=3, mom_days=252, use_filter=False, weight="equal",
        target_vol=xsec.TARGET_VOL, start=None, end=None, exec_mode="next_open",
        band=0.0, rank_buffer=0, force_weight=None):
    """band: 单只权重最大偏离小于该值就不调仓（0=关闭，基线行为）。
    rank_buffer: 现任掉出 top K 但仍在 top K+buffer 内则保留（0=关闭）。
    force_weight: 执行端强制权重方式（None=按选中的 weight，基线行为）。
    """
    open_df, close_df = xsec.build_panels(data)
    cal = close_df.index
    close_ff = close_df.ffill()
    ret_f = close_df.pct_change(fill_method=None).ffill()
    cols = list(close_df.columns)

    def snap(ts, method="pad"):
        idx = int(cal.get_indexer([pd.Timestamp(ts)], method=method)[0])
        return max(0, idx)

    start_idx = snap(start) if start else 0
    end_idx = snap(end) if end else len(cal) - 1
    flags = xsec.month_end_flags(cal)

    cash, shares = 100.0, pd.Series(0.0, index=cols)
    last_target = pd.Series(0.0, index=cols)
    pending = None
    rows, trades, signals = [], [], []
    state = {"skips": 0}

    def execute(target_w, d, px, px_source, signal_date):
        nonlocal cash, shares, last_target
        cur_val = pd.Series({t: shares[t] * px[t] for t in cols})
        tradable = px.notna() & (px > 0)
        w = target_w.where(tradable, 0.0)
        pv = cash + float(cur_val.sum())
        if band > 0:
            cur_w = (cur_val / pv).reindex(cols).fillna(0.0)
            if float((w - cur_w).abs().max()) < band:
                state["skips"] += 1
                return
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
            held = tuple(c for c in cols if last_target[c] > 1e-9)
            target, diag = make_target(close_df, ret_f, i, k, mom_days, use_filter,
                                       force_weight or weight, target_vol,
                                       held=held, rank_buffer=rank_buffer)
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
                  metrics=xsec.compute_metrics(nav_df, tr), n_skips=state["skips"])
