"""有纪律的优化尝试（一次性实验，不进生产流水线）。

判定只看 OOS（walk-forward 拼接净值 2016-02-29→2026-02-27），
训练窗口表现仅作参考。锁定口径不动：7bp / 目标波动 10% / 杠杆 ≤1.0。

用法:
  python experiments/opt_20261108.py --variants baseline
  python experiments/opt_20261108.py --variants band10 band20 band30 buffer1 invvol

baseline 变体会与 runs/<最新>/metrics.json 的 OOS 指标逐位比对，不一致直接退出。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import pipeline_core as pc  # noqa: E402
import xsec  # noqa: E402
import xsec_opt as opt  # noqa: E402

OUT = ROOT / "experiments" / "out"
FLAT = pc.FLAT
M_IDX = pc.M_IDX

VARIANTS = {
    "baseline": {},
    "band10": {"band": 0.10},
    "band20": {"band": 0.20},
    "band30": {"band": 0.30},
    "buffer1": {"rank_buffer": 1},
    "invvol": {"force_weight": "inv_vol"},
}


def walk_forward(data, base_start, end, variant):
    t0 = pd.Timestamp(base_start)
    windows, returns = [], []
    while True:
        te = t0 + pd.DateOffset(years=8)
        purge = te - pd.DateOffset(months=1)
        ve_end = te + pd.DateOffset(years=1)
        if ve_end > pd.Timestamp(end):
            break
        sel = []
        for k, mom_days, use_filter, weight in pc.CANDIDATES:
            r = opt.run(data, k=k, mom_days=mom_days, use_filter=use_filter,
                        weight=weight, start=t0, end=purge, **variant)
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

        val = opt.run(data, k=pick["k"], mom_days=pick["mom_days"],
                      use_filter=pick["filter"], weight=pick["weight"],
                      start=te, end=ve_end, **variant)
        vm = val.metrics
        windows.append({
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
            "val_skips": int(val.n_skips),
        })
        seg = val.nav["nav"].pct_change()
        seg.iloc[0] = 0.0
        returns.append(seg)
        t0 = t0 + pd.DateOffset(years=1)

    rets = pd.concat(returns).sort_index()
    rets = rets[~rets.index.duplicated(keep="first")]
    oos_nav = (1 + rets).cumprod() * 100.0
    bench_oos = xsec.run_benchmark(data, windows[0]["val_start"], windows[-1]["val_end"])
    cont = xsec.run_benchmark(data, base_start, end).reindex(oos_nav.index)
    cr = cont.pct_change()
    cr.iloc[0] = 0.0
    cont_nav = (1 + cr).cumprod() * 100.0
    nav_df = pd.DataFrame({"strategy": oos_nav,
                           "bench_same_start": bench_oos.reindex(oos_nav.index),
                           "bench_continuous": cont_nav}, index=oos_nav.index)
    return nav_df, windows


def yearly_returns(nav):
    ret = nav.pct_change()
    ret.iloc[0] = 0.0
    g = ret.groupby(ret.index.year).apply(lambda s: float((1 + s).prod() - 1))
    return {str(k): float(v) for k, v in g.items()}


def latest_baseline_metrics():
    runs = sorted((ROOT / "runs").glob("*/metrics.json"))
    if not runs:
        return None, None
    return runs[-1], json.loads(runs[-1].read_text(encoding="utf-8"))


def check_baseline(perf, turn):
    path, ref = latest_baseline_metrics()
    if ref is None:
        raise SystemExit("[基线缺失] runs/ 下没有 metrics.json，先跑 run_monthly.py")
    diffs = []
    for key in ("strategy", "bench_same_start", "bench_continuous"):
        for m in ("cagr", "ann_vol", "sharpe", "max_drawdown", "calmar", "final_nav"):
            a, b = perf[key][m], ref["tables"]["performance"][key][m]
            if abs(a - b) > 1e-12:
                diffs.append(f"{key}.{m}: {a!r} != {b!r}")
    for m in ("annual_turnover", "cost_annual_bp", "cost_total_bp", "n_trades"):
        a, b = turn["strategy"][m], ref["tables"]["turnover"]["strategy"][m]
        if abs(a - b) > 1e-9:
            diffs.append(f"turnover.{m}: {a!r} != {b!r}")
    if diffs:
        raise SystemExit("[基线复现失败] 实验引擎在默认参数下与生产引擎不一致:\n"
                         + "\n".join(diffs[:10]))
    return str(path), ref["run"]["result_sha256"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["baseline"])
    args = ap.parse_args()

    data = pc.load_data()
    _, close_df = xsec.build_panels(data)
    base_start = xsec.common_start(close_df)
    end = close_df.index[-1]
    pc.validate(data, base_start, end)

    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.variants:
        if name not in VARIANTS:
            raise SystemExit(f"[未知变体] {name}，可选 {sorted(VARIANTS)}")
        nav_df, windows = walk_forward(data, base_start, end, VARIANTS[name])
        perf = {key: pc.series_metrics(nav_df[key])
                for key in ("strategy", "bench_same_start", "bench_continuous")}
        turn = pc.turnover_tables(windows, perf)
        s = perf["strategy"]

        payload = {
            "variant": name, "params": VARIANTS[name],
            "oos": {"start": str(nav_df.index[0].date()), "end": str(nav_df.index[-1].date()),
                    "cagr": s["cagr"], "sharpe": s["sharpe"], "mdd": s["max_drawdown"],
                    "calmar": s["calmar"], "ann_vol": s["ann_vol"],
                    "turnover": turn["strategy"]["annual_turnover"],
                    "cost_bp": turn["strategy"]["cost_annual_bp"],
                    "cost_total_bp": turn["strategy"]["cost_total_bp"],
                    "n_trades": turn["strategy"]["n_trades"],
                    "skips": int(sum(w["val_skips"] for w in windows))},
            "bench": {"same_start_cagr": perf["bench_same_start"]["cagr"],
                      "continuous_cagr": perf["bench_continuous"]["cagr"]},
            "yearly": yearly_returns(nav_df["strategy"]),
            "windows": windows,
        }
        if name == "baseline":
            path, ref_sha = check_baseline(perf, turn)
            payload["baseline_check"] = {"matched": path, "result_sha256": ref_sha}
            print(f"[基线复现] {path} result_sha256={ref_sha} 全部指标一致", flush=True)

        (OUT / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
        print(f"{name:9s} CAGR {s['cagr']*100:6.2f}%  Sharpe {s['sharpe']:.3f}  "
              f"MDD {s['max_drawdown']*100:7.2f}%  Calmar {s['calmar']:.3f}  "
              f"turnover {turn['strategy']['annual_turnover']*100:7.2f}%  "
              f"cost {turn['strategy']['cost_annual_bp']:.2f}bp  "
              f"skips {payload['oos']['skips']}", flush=True)


if __name__ == "__main__":
    main()
