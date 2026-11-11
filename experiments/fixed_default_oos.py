"""补 DEFAULT_PARAMS 的样本外成绩（评审 P1 收口）。

为什么必须单独一份报告：README 里的 1.02 夏普来自「每年用前 8 年重新选参」
的 walk-forward 拼接，而 run_signal.py 默认下单用的是固定的
K=3 / 12-1 / 等权 / 无过滤。两组参数不是一回事，拿前者给后者背书是口径混淆。
本脚本只跑固定参数这一组，输出独立样本外报告，区间取自归档 baseline.json，
保证与 walk-forward 的 OOS 逐字对齐、可直接对照（但**不可互换**）。

复算：python experiments/fixed_default_oos.py
前置：data/ 下 10 只 ETF 的 Yahoo 前复权 CSV（见 README 数据声明），装 requirements.txt。

口径说明（写进产物，免得后人混用）：
· 本脚本是**单次连续运行**，参数全程不变；
· walk-forward 是**每年换参后拼接**，参数逐年变化；
· 两者区间相同、引擎相同，但计算对象不同，指标不可相加或互相背书。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import live_params  # noqa: E402
import pipeline_core as pc  # noqa: E402
import xsec  # noqa: E402

OUT = ROOT / "experiments" / "out" / "fixed_default_oos.json"
BASELINE = ROOT / "experiments" / "out" / "baseline.json"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description="固定参数（DEFAULT_PARAMS）的样本外成绩")
    ap.add_argument("--data-dir", help="行情目录，默认 <repo>/data")
    args = ap.parse_args()
    if args.data_dir:
        pc.DATA_DIR = Path(args.data_dir)

    pc.assert_locked()
    data = pc.load_data()
    _, close_df = xsec.build_panels(data)
    base_start = xsec.common_start(close_df)
    data_end = close_df.index[-1]

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    start, end = baseline["oos"]["start"], baseline["oos"]["end"]

    params = live_params.load_params()
    res = xsec.run(data, k=params["k"], mom_days=params["mom_days"],
                   use_filter=params["use_filter"], weight=params["weight"],
                   target_vol=params["target_vol"], start=start, end=end)
    nav = res.nav["nav"]
    m = res.metrics
    metrics = {k: round(float(m[k]), 8) for k in
               ("cagr", "ann_vol", "sharpe", "max_drawdown", "annual_turnover",
                "cost_annual_bp", "cost_total_bp", "n_trades")}
    metrics["calmar"] = round(float(m["cagr"] / abs(m["max_drawdown"])), 8)

    # 基准与 run_monthly.py 的构造完全一致：同起点重建 / 从样本起点连续持有再截取 OOS。
    bench_same = xsec.run_benchmark(data, start, end).reindex(nav.index)
    cont = xsec.run_benchmark(data, base_start, data_end).reindex(nav.index)
    cr = cont.pct_change()
    cr.iloc[0] = 0.0
    cont_nav = (1 + cr).cumprod() * 100.0
    if bench_same.isna().any() or cont_nav.isna().any():
        raise SystemExit("[区间错误] 基准净值无法对齐策略 OOS 索引，拒绝出报告")
    bench = {
        "same_start": pc.series_metrics(bench_same),
        "continuous": pc.series_metrics(cont_nav),
    }

    wf = baseline["oos"]
    comparison = {
        "reference": "experiments/out/baseline.json（walk-forward 每年重新选参）",
        "sharpe_diff": round(metrics["sharpe"] - wf["sharpe"], 8),
        "cagr_diff_pp": round((metrics["cagr"] - wf["cagr"]) * 100, 8),
        "mdd_diff_pp": round((metrics["max_drawdown"] - wf["mdd"]) * 100, 8),
        "calmar_diff": round(metrics["calmar"] - wf["calmar"], 8),
        "note": "差值 = 固定参数 − walk-forward；正/负只表示两组口径的差异，不是策略改进量",
    }

    report = {
        "generated_by": "experiments/fixed_default_oos.py",
        "command": "python experiments/fixed_default_oos.py",
        "method": "单次连续运行固定参数；不是每年换参的 walk-forward 拼接",
        "params": params,
        "params_sha256": live_params.params_sha256(params),
        "period": {"start": str(nav.index[0].date()), "end": str(nav.index[-1].date()),
                   "n_rows": int(len(nav)), "common_start": str(base_start.date()),
                   "data_end": str(data_end.date()),
                   "aligned_with": "experiments/out/baseline.json → oos.start / oos.end"},
        "metrics": metrics,
        "benchmarks": bench,
        "gap_bp": {k: round((v["cagr"] - metrics["cagr"]) * 1e4, 4)
                   for k, v in bench.items()},
        "comparison_to_walk_forward": comparison,
        "data": {"source": "Yahoo Finance / yfinance auto_adjust=True（见 README 数据声明）",
                 "files": [{"file": f"{t}.csv",
                            "sha256": sha256_file(pc.DATA_DIR / f"{t}.csv")}
                           for t in pc.UNIVERSE]},
        "code": {f"{n}.py": sha256_file(ROOT / f"{n}.py")
                 for n in ("xsec", "pipeline_core", "guards", "engine", "live_params")},
        "environment": {"python": sys.version.split()[0],
                        "pandas": pd.__version__},
        "caveats": [
            "这组参数有独立报告，不等于 README 的 1.02 夏普属于它；两组口径不可互换。",
            "标的池是 10 只存活至今的 ETF，存在幸存者偏差；Yahoo 复权价不是可成交价格。",
            "单边 7bp（5bp 佣金 + 2bp 滑点）对 VNQ / DBC 等偏乐观，见 SUMMARY 已知边界。",
            "数据文件未入仓（Yahoo 条款），重新下载后字节不同，指标会有 1e-6 级漂移。",
        ],
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float),
                   encoding="utf-8")
    print(f"完成 {OUT}")
    print(f"参数 {live_params.describe(params)} | sha256 {report['params_sha256'][:16]}")
    print(f"区间 {report['period']['start']} → {report['period']['end']} | 行数 {len(nav)}")
    print(f"CAGR {metrics['cagr']*100:.2f}% / Sharpe {metrics['sharpe']:.3f} / "
          f"MDD {metrics['max_drawdown']*100:.2f}% / Calmar {metrics['calmar']:.3f}")
    print(f"基准 同起点 {bench['same_start']['cagr']*100:.2f}% / "
          f"连续 {bench['continuous']['cagr']*100:.2f}%")


if __name__ == "__main__":
    main()
