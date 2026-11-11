"""引擎整合层：数据硬校验、walk-forward、指标汇总、生效配置。

只被 run_monthly.py 调用；不写报告、不建输出目录。
口径锁死: t 日收盘信号、t+1 开盘成交、月度调仓、单边 7bp、目标波动 10%、杠杆 ≤1.0。
"""

import hashlib
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT))
import engine  # noqa: E402
import xsec  # noqa: E402

UNIVERSE = engine.BASE10
MOM_MONTHS = {6: 126, 9: 189, 12: 252}
CANDIDATES = [(k, m, f, w) for k in (2, 3) for m in MOM_MONTHS.values()
              for f in (False, True) for w in ("equal", "inv_vol")]
M_IDX = {126: 0, 189: 1, 252: 2}
FLAT = 0.25

# 代码指纹：固定角色顺序（入口 -> 整合/报告 -> 引擎），不依赖目录枚举顺序。
CODE_FILES = (
    ("run_monthly.py", ROOT / "run_monthly.py"),
    ("run_signal.py", ROOT / "run_signal.py"),
    ("pipeline_core.py", ROOT / "pipeline_core.py"),
    ("monthly_report.py", ROOT / "monthly_report.py"),
    ("engine.py", ROOT / "engine.py"),
    ("xsec.py", ROOT / "xsec.py"),
)


def code_sha256():
    """逐文件 sha256 后按固定顺序拼接，再整体哈希；标签用相对路径，移动目录不变。"""
    lines = []
    for label, p in CODE_FILES:
        if not p.exists():
            raise SystemExit(f"[代码指纹失败] 缺少参与计算的源码 {p}")
        lines.append(f"{label}:{hashlib.sha256(p.read_bytes()).hexdigest()}")
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- 数据（硬校验）

def load_data():
    out = {}
    for t in UNIVERSE:
        p = DATA_DIR / f"{t}.csv"
        if not p.exists():
            raise SystemExit(f"[数据缺失] 缺少文件 {p}，流程终止（不做任何填充）")
        df = pd.read_csv(p, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        if df["date"].duplicated().any():
            dups = df.loc[df["date"].duplicated(), "date"].head(3).dt.date.tolist()
            raise SystemExit(f"[数据异常] {p} 存在重复日期 {dups}，流程终止")
        out[t] = df.sort_values("date").reset_index(drop=True)
    return out


def validate(data, start, end):
    cal = pd.DatetimeIndex(sorted(set().union(*[set(df["date"]) for df in data.values()])))
    problems, checks = [], []
    for t, df in data.items():
        p = DATA_DIR / f"{t}.csv"
        inwin = df[(df["date"] >= start) & (df["date"] <= end)]
        if df["date"].max() < end:
            problems.append(f"{p}: 区间缺失 {df['date'].max().date()} 之后 → {end.date()}（文件末行 "
                            f"{df['date'].max().date()}）")
        miss_days = [d for d in cal if start <= d <= end and d not in set(inwin["date"])]
        if miss_days:
            problems.append(f"{p}: 缺行情 {len(miss_days)} 天，最早 {miss_days[0].date()}，"
                            f"最晚 {miss_days[-1].date()}")
        have_m = set(pd.PeriodIndex(inwin["date"], freq="M").unique())
        miss_m = [str(per) for per in pd.period_range(start, end, freq="M") if per not in have_m]
        if miss_m:
            problems.append(f"{p}: 缺月份 {len(miss_m)} 个 {miss_m[:3]}"
                            + (" ..." if len(miss_m) > 3 else ""))
        bad = inwin[(inwin[["open", "close"]].isna().any(axis=1))
                    | (inwin["open"] <= 0) | (inwin["close"] <= 0)]
        if len(bad):
            problems.append(f"{p}: 非正/空行情 {len(bad)} 行，最早 {bad['date'].min().date()}")
        checks.append({"file": p.name, "rows_in_window": int(len(inwin)),
                       "missing_days": len(miss_days), "missing_months": len(miss_m),
                       "bad_rows": int(len(bad))})
    if problems:
        raise SystemExit("[数据完整性失败，禁止插值/前值填充]\n" + "\n".join(problems))
    return {"passed": True, "checks": checks, "checked_files": len(checks),
            "months": int(len(pd.period_range(start, end, freq="M"))),
            "missing_months": sum(c["missing_months"] for c in checks),
            "missing_days": sum(c["missing_days"] for c in checks),
            "bad_rows": sum(c["bad_rows"] for c in checks)}


# ---------------------------------------------------------------- walk-forward

def walk_forward(data, base_start, end, cfg):
    wf = cfg["walk_forward"]
    t0 = pd.Timestamp(base_start) + pd.DateOffset(months=wf["offset_months"])
    windows, returns = [], []
    while True:
        te = t0 + pd.DateOffset(years=wf["train_years"])
        purge = te - pd.DateOffset(months=wf["purge_months"])
        ve_end = te + pd.DateOffset(years=wf["val_years"])
        if ve_end > pd.Timestamp(end):
            break
        sel = []
        for k, mom_days, use_filter, weight in CANDIDATES:
            r = xsec.run(data, k=k, mom_days=mom_days, use_filter=use_filter,
                         weight=weight, start=t0, end=purge)
            med, _, _ = xsec.rolling_median_sharpe(r.nav["nav"], t0, purge)
            sel.append({"k": k, "mom_days": mom_days, "filter": use_filter,
                        "weight": weight, "med": med})
        for c in sel:
            nb = [o for o in sel if o is not c and o["filter"] == c["filter"]
                  and o["weight"] == c["weight"] and abs(o["k"] - c["k"]) <= 1
                  and abs(M_IDX[o["mom_days"]] - M_IDX[c["mom_days"]]) <= 1]
            c["neighbor_min"] = min((o["med"] for o in nb), default=c["med"])
            c["flat"] = bool(np.isfinite(c["med"]) and c["neighbor_min"] >= c["med"] - FLAT)
        ordered = sorted(sel, key=lambda o: (-o["med"] if np.isfinite(o["med"]) else 9e9))
        pick = next((o for o in ordered if o["flat"]), None)
        fallback = pick is None
        pick = pick or ordered[0]

        val = xsec.run(data, k=pick["k"], mom_days=pick["mom_days"],
                       use_filter=pick["filter"], weight=pick["weight"],
                       start=te, end=ve_end)
        vm = val.metrics
        windows.append({
            "train_start": str(t0.date()), "purge_end": str(pd.Timestamp(purge).date()),
            "val_start": str(pd.Timestamp(te).date()), "val_end": str(pd.Timestamp(ve_end).date()),
            "k": pick["k"], "mom_months": pick["mom_days"] // 21,
            "filter": "有" if pick["filter"] else "无",
            "weight": "等权" if pick["weight"] == "equal" else "波动倒数",
            "sel_median": float(pick["med"]), "flat": bool(pick["flat"]), "fallback": bool(fallback),
            "val_cagr": vm["cagr"], "val_vol": vm["ann_vol"], "val_sharpe": vm["sharpe"],
            "val_mdd": vm["max_drawdown"], "val_turnover": vm["annual_turnover"],
            "val_cost_bp": vm["cost_annual_bp"], "val_cost_total_bp": vm["cost_total_bp"],
            "val_n_trades": int(vm["n_trades"]),
            "val_holdings": float(val.signals["n_selected"].mean()),
            "val_turnover_entry": vm["turnover_entry"],
            "val_turnover_exit": vm["turnover_exit"],
            "val_turnover_adjust": vm["turnover_adjust"],
        })
        seg = val.nav["nav"].pct_change()
        seg.iloc[0] = 0.0
        returns.append(seg)
        t0 = t0 + pd.DateOffset(years=wf["step_years"])
    if not windows:
        raise SystemExit(f"[区间不足] {base_start}→{end} 内没有完整的 {wf['val_years']} 年验证窗口")

    rets = pd.concat(returns).sort_index()
    rets = rets[~rets.index.duplicated(keep="first")]
    oos_nav = (1 + rets).cumprod() * 100.0
    bench_oos = xsec.run_benchmark(data, windows[0]["val_start"], windows[-1]["val_end"])
    cont = xsec.run_benchmark(data, base_start, end).reindex(oos_nav.index)
    cr = cont.pct_change()
    cr.iloc[0] = 0.0
    cont_nav = (1 + cr).cumprod() * 100.0
    nav_df = pd.DataFrame({
        "strategy": oos_nav,
        "bench_same_start": bench_oos.reindex(oos_nav.index),
        "bench_continuous": cont_nav,
    }, index=oos_nav.index)
    return nav_df, windows


# ---------------------------------------------------------------- 指标汇总

def series_metrics(nav):
    m = xsec.simple_metrics(nav.dropna())
    mdd = m["max_drawdown"]
    return {**m, "calmar": float(m["cagr"] / abs(mdd))
            if np.isfinite(mdd) and mdd != 0 else float("nan")}


def turnover_tables(windows, perf):
    s = {
        "annual_turnover": float(np.mean([w["val_turnover"] for w in windows])),
        "cost_annual_bp": float(np.mean([w["val_cost_bp"] for w in windows])),
        "cost_total_bp": float(np.sum([w["val_cost_total_bp"] for w in windows])),
        "n_trades": int(sum(w["val_n_trades"] for w in windows)),
        "avg_holdings": float(np.mean([w["val_holdings"] for w in windows])),
        "turnover_entry": float(np.mean([w["val_turnover_entry"] for w in windows])),
        "turnover_exit": float(np.mean([w["val_turnover_exit"] for w in windows])),
        "turnover_adjust": float(np.mean([w["val_turnover_adjust"] for w in windows])),
    }
    s["cost_to_cagr"] = (s["cost_annual_bp"] / 1e4 / perf["strategy"]["cagr"]
                         if perf["strategy"]["cagr"] > 0 else float("nan"))
    years = perf["strategy"]["years"]
    b = {"annual_turnover": 0.0, "cost_annual_bp": 7.0 / years, "cost_total_bp": 7.0,
         "n_trades": 1, "avg_holdings": float(len(UNIVERSE)), "turnover_entry": 0.0,
         "turnover_exit": 0.0, "turnover_adjust": 0.0,
         "cost_to_cagr": (7.0 / years / 1e4 / perf["bench_same_start"]["cagr"])}
    return {"strategy": s, "benchmark": b, "windows": len(windows)}


def gap_rows(perf):
    s = perf["strategy"]
    rows = []
    for key, name in (("bench_same_start", "同起点重建"), ("bench_continuous", "连续持有")):
        b = perf[key]
        rows.append({"benchmark": name, "strategy_cagr": s["cagr"], "bench_cagr": b["cagr"],
                     "gap_bp": (b["cagr"] - s["cagr"]) * 1e4,
                     "strategy_sharpe": s["sharpe"], "bench_sharpe": b["sharpe"],
                     "sharpe_diff": s["sharpe"] - b["sharpe"], "strategy_mdd": s["max_drawdown"],
                     "bench_mdd": b["max_drawdown"],
                     "mdd_diff_pp": (s["max_drawdown"] - b["max_drawdown"]) * 100})
    return rows


# ---------------------------------------------------------------- 锁死口径（入口校验）

def assert_locked():
    """回测与信号共用：锁死口径被改动时直接退出，不允许带着漂移出报告。

    只校验能从引擎常量/函数签名读取的项。"t 收盘信号 / t+1 开盘 / 月度 /
    杠杆 ≤1.0 / 现金 0 收益"是执行流程本身，由 code_sha256 覆盖，
    不在这里重复实现一遍（重复实现反而会两边一起错）。
    刻意不改 effective_config 的 locked 字典：那会改变 config_sha256，
    进而让历史 result_sha256 失去可比性。
    """
    for name, got, want in (
        ("xsec.COST_RATE", xsec.COST_RATE, 0.0007),
        ("xsec.TARGET_VOL", xsec.TARGET_VOL, 0.10),
        ("xsec.SKIP_DAYS", xsec.SKIP_DAYS, 21),
        ("xsec.TRADING_DAYS", xsec.TRADING_DAYS, 252),
        ("xsec.VOL_DAYS", xsec.VOL_DAYS, 20),
    ):
        if got != want:
            raise SystemExit(f"[口径漂移] {name}={got}，锁死值 {want}，请先确认再跑")
    mode = inspect.signature(xsec.run).parameters["exec_mode"].default
    if mode != "next_open":
        raise SystemExit(f"[口径漂移] xsec.run 默认 exec_mode={mode}，锁死值 next_open")


# ---------------------------------------------------------------- 生效配置

def effective_config(args):
    return {
        "universe": UNIVERSE,
        "walk_forward": {"train_years": 8, "val_years": 1, "step_years": 1,
                         "purge_months": 1, "offset_months": 0},
        "candidate_grid": {"k": sorted({c[0] for c in CANDIDATES}),
                           "mom_months": sorted(MOM_MONTHS),
                           "use_filter": sorted({c[2] for c in CANDIDATES}),
                           "weight": sorted({c[3] for c in CANDIDATES}),
                           "n_candidates": len(CANDIDATES)},
        "selection": {"method": "rolling_252d_sharpe_median_at_month_ends",
                      "flat_neighborhood": FLAT,
                      "fallback": "highest median when no flat candidate"},
        "locked": {"signal": "t_close", "fill": "t+1_open", "rebalance": "monthly",
                   "cost_bp_one_way": 7.0, "target_vol": 0.10, "leverage_cap": 1.0,
                   "cash_return": 0.0, "benchmark": "equal_weight_buy_and_hold_7bp_one_time"},
        # 相对标签而非本机绝对路径：DATA_DIR = ROOT/"data" 会随仓库所在目录变化，
        # 写进 config_sha256 会让「同一份代码 + 同一批数据」在两台机器上得到不同指纹。
        # 绝对路径仍留在 metrics.json.inputs.data_dir 供人核对，但不参与任何哈希。
        "data": {"dir": DATA_DIR.relative_to(ROOT).as_posix(), "end_override": args.end},
    }


def cfg_rows(cfg):
    w, g, s, lk = cfg["walk_forward"], cfg["candidate_grid"], cfg["selection"], cfg["locked"]
    return [
        ["口径（锁死）", "信号 / 成交 / 调仓", f'{lk["signal"]} / {lk["fill"]} / {lk["rebalance"]}'],
        ["口径（锁死）", "单边成本 bp", str(lk["cost_bp_one_way"])],
        ["口径（锁死）", "目标波动 / 杠杆上限", f'{lk["target_vol"]} / {lk["leverage_cap"]}'],
        ["口径（锁死）", "现金收益", str(lk["cash_return"])],
        ["walk-forward", "训练 / 验证 / 步长 / purge（年,年,年,月）",
         f'{w["train_years"]} / {w["val_years"]} / {w["step_years"]} / {w["purge_months"]}'],
        ["walk-forward", "起点错开（月）", str(w["offset_months"])],
        ["选参", "候选数 / K / 动量月 / 权重",
         f'{g["n_candidates"]} / {g["k"]} / {g["mom_months"]} / {g["weight"]}'],
        ["选参", "过滤 / 邻域平坦", f'{g["use_filter"]} / {s["flat_neighborhood"]}'],
        ["数据", "数据目录（相对仓库根，不进指纹） / 截止覆盖",
         f'{cfg["data"]["dir"]} / {cfg["data"]["end_override"] or "无（用最新行）"}'],
    ]
