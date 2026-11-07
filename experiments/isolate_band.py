"""隔离实验：选参固定为基线路径（用基线窗口的 k/月/过滤/权重），
只在验证段开启换手带，用于剥离「换手带本身」与「选参被带偏」两种效应。

输出每个 band 的 OOS 指标；band=0 必须与基线逐位一致。
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import pipeline_core as pc  # noqa: E402
import xsec  # noqa: E402
import xsec_opt as opt  # noqa: E402

OUT = ROOT / "experiments" / "out"
BANDS = [0.0, 0.10, 0.20, 0.30]


def params_from(w):
    return {"k": w["k"], "mom_days": w["mom_months"] * 21,
            "use_filter": w["filter"] == "有",
            "weight": "equal" if w["weight"] == "等权" else "inv_vol"}


def run_band(data, windows_src, band):
    segs, wins = [], []
    for w in windows_src:
        p = params_from(w)
        r = opt.run(data, start=pd.Timestamp(w["val_start"]), end=pd.Timestamp(w["val_end"]),
                    band=band, **p)
        m = r.metrics
        wins.append({"val_start": w["val_start"], "val_end": w["val_end"],
                     "val_turnover": m["annual_turnover"], "val_cost_bp": m["cost_annual_bp"],
                     "val_cost_total_bp": m["cost_total_bp"], "val_n_trades": m["n_trades"],
                     "val_holdings": float(r.signals["n_selected"].mean()),
                     "val_turnover_entry": m["turnover_entry"],
                     "val_turnover_exit": m["turnover_exit"],
                     "val_turnover_adjust": m["turnover_adjust"],
                     "val_skips": int(r.n_skips)})
        seg = r.nav["nav"].pct_change()
        seg.iloc[0] = 0.0
        segs.append(seg)
    rets = pd.concat(segs).sort_index()
    rets = rets[~rets.index.duplicated(keep="first")]
    nav = (1 + rets).cumprod() * 100.0
    return nav, wins


def yearly(nav):
    ret = nav.pct_change()
    ret.iloc[0] = 0.0
    return {str(k): float(v) for k, v in ret.groupby(ret.index.year).apply(
        lambda s: (1 + s).prod() - 1).items()}


def main():
    base = json.loads((OUT / "baseline.json").read_text(encoding="utf-8"))
    data = pc.load_data()
    bench = xsec.run_benchmark(data, base["windows"][0]["val_start"],
                               base["windows"][-1]["val_end"])
    rows = []
    for b in BANDS:
        nav, wins = run_band(data, base["windows"], b)
        perf = {"strategy": pc.series_metrics(nav),
                "bench_same_start": pc.series_metrics(bench)}
        turn = pc.turnover_tables(wins, perf)["strategy"]
        s = perf["strategy"]
        rows.append({"band": b, "cagr": s["cagr"], "sharpe": s["sharpe"],
                     "mdd": s["max_drawdown"], "calmar": s["calmar"],
                     "turnover": turn["annual_turnover"], "cost_bp": turn["cost_annual_bp"],
                     "skips": sum(w["val_skips"] for w in wins),
                     "yearly": yearly(nav)})
        (OUT / f"fixed_band{int(b*100)}.json").write_text(
            json.dumps({"band": b, "rows": rows[-1], "windows": wins},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"band {b:.2f}  CAGR {s['cagr']*100:6.2f}%  Sharpe {s['sharpe']:.3f}  "
              f"MDD {s['max_drawdown']*100:7.2f}%  Calmar {s['calmar']:.3f}  "
              f"turnover {turn['annual_turnover']*100:7.2f}%  cost {turn['cost_annual_bp']:.2f}bp  "
              f"skips {rows[-1]['skips']}", flush=True)

    b0, b20 = rows[0], rows[2]
    print("\n分年度：固定选参，band=0 vs band=0.20")
    years = sorted(b0["yearly"])
    print("| 年份 | band0 | band20 | 差 |")
    print("|---|---|---|---|")
    for y in years:
        d = b20["yearly"][y] - b0["yearly"][y]
        print(f"| {y} | {b0['yearly'][y]*100:+.2f}% | {b20['yearly'][y]*100:+.2f}% | {d*100:+.2f}pp |")


if __name__ == "__main__":
    main()
