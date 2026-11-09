"""月度复跑流水线（唯一入口）。

  python run_monthly.py

  * 输出到 runs/<YYYYMMDD_HHMMSS>/，不覆盖任何历史目录
  * 四张表 + 一张对比图 + report.md + metrics.json（含 effective config）
  * 同一输入重复运行 result_sha256 必须逐位相同；不同则写进 determinism
  * 数据缺月份/缺行情/非正价格 -> 直接失败并指出文件与区间，禁止插值或前值填充
  * 与上一期对比：哪一项变了、变了多少，写进 report.md 与 metrics.json

口径锁死: t 日收盘信号、t+1 开盘成交、月度调仓、单边 7bp、目标波动 10%、杠杆 ≤1.0。
数据/引擎在 pipeline_core.py，报告/图表在 monthly_report.py。
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
sys.path.insert(0, str(ROOT))
import monthly_report  # noqa: E402
import pipeline_core as pc  # noqa: E402
import xsec  # noqa: E402

CHINA_TZ = timezone(timedelta(hours=8))
SERIES = monthly_report.SERIES
WINDOW_FP_KEYS = ("val_start", "val_end", "k", "mom_months", "filter", "weight",
                  "sel_median", "val_cagr", "val_sharpe", "val_mdd", "val_turnover",
                  "val_cost_bp", "val_cost_total_bp", "val_n_trades", "val_holdings",
                  "val_turnover_entry", "val_turnover_exit", "val_turnover_adjust")


# ---------------------------------------------------------------- 指纹 / 对比

def canonical(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), default=float).encode("utf-8")).hexdigest()


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def field_diffs(a, b, prefix=""):
    """递归找出不一致的字段路径，用于 determinism.differing_fields。"""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            key = f"{prefix}.{k}" if prefix else k
            if k not in a or k not in b:
                out.append(key)
            else:
                out.extend(field_diffs(a[k], b[k], key))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b) or a != b:
            out.append(prefix or "list")
    elif a != b:
        out.append(prefix)
    return out


def find_previous():
    if not RUNS_DIR.exists():
        return None
    found = sorted(RUNS_DIR.glob("*/metrics.json"))
    return found[-1] if found else None


def build_compare(prev_path, cfg, cfg_sha, inp_fp, files, perf, windows):
    if prev_path is None:
        return {"previous_run": None}
    prev = json.loads(prev_path.read_text(encoding="utf-8"))
    a, b = flatten(prev.get("effective_config", {})), flatten(cfg)
    config_changes = []
    for key in sorted(set(a) | set(b)):
        if key.startswith("resolved") or a.get(key, "__miss__") == b.get(key, "__miss__"):
            continue
        av, bv = a.get(key), b.get(key)
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (av, bv))
        config_changes.append({"field": key, "previous": av, "current": bv,
                               "delta": float(bv - av) if numeric else None})

    input_changes = []
    pin = prev.get("inputs", {})
    for item, av, bv in (("common_start", pin.get("common_start"), cfg["resolved"]["common_start"]),
                         ("data_end", pin.get("data_end"), cfg["resolved"]["data_end"]),
                         ("run_end", pin.get("run_end"), cfg["resolved"]["run_end"]),
                         ("验证窗口数", len(prev.get("windows", [])), len(windows))):
        if av != bv:
            input_changes.append({"item": item, "previous": av, "current": bv})
    prev_hash = {f["file"]: f["sha256"] for f in pin.get("files", [])}
    for f in files:
        if prev_hash.get(f["file"]) != f["sha256"]:
            input_changes.append({"item": f"{f['file']} sha256", "previous": prev_hash.get(f["file"]),
                                  "current": f["sha256"]})

    metric_changes = []
    for key, _ in SERIES:
        pm = prev.get("tables", {}).get("performance", {}).get(key, {})
        for metric in ("cagr", "ann_vol", "sharpe", "max_drawdown", "calmar", "final_nav"):
            if metric in pm and perf[key][metric] != pm[metric]:
                metric_changes.append({"series": key, "metric": metric,
                                       "previous": pm[metric], "current": perf[key][metric],
                                       "delta": perf[key][metric] - pm[metric]})

    select_changes = []
    prev_w = {w["val_start"]: w for w in prev.get("windows", [])}
    for w in windows:
        old = prev_w.get(w["val_start"])
        if not old:
            continue
        av = f'K={old["k"]}/{old["mom_months"]}月/{old["filter"]}/{old["weight"]}'
        bv = f'K={w["k"]}/{w["mom_months"]}月/{w["filter"]}/{w["weight"]}'
        if av != bv:
            select_changes.append({"val_start": w["val_start"], "previous": av, "current": bv})
    return {"previous_run": prev["run"]["run_id"],
            "comparable": prev["run"]["config_sha256"] == cfg_sha
            and pin.get("inputs_fingerprint") == inp_fp,
            "config_changes": config_changes, "input_changes": input_changes,
            "metric_changes": metric_changes, "selection_changes": select_changes}


def determinism(prev_path, compare, result_sha, tables, windows):
    if prev_path is None:
        return {"status": "first_run", "current_result_sha256": result_sha}
    prev = json.loads(prev_path.read_text(encoding="utf-8"))
    if not compare["comparable"]:
        return {"status": "not_comparable", "current_result_sha256": result_sha,
                "previous_result_sha256": prev["run"]["result_sha256"]}
    if prev["run"]["result_sha256"] == result_sha:
        return {"status": "identical", "current_result_sha256": result_sha,
                "previous_result_sha256": result_sha}
    diffs = field_diffs({"tables": prev.get("tables", {}), "windows": prev.get("windows", [])},
                        {"tables": tables, "windows": windows})
    return {"status": "differs", "current_result_sha256": result_sha,
            "previous_result_sha256": prev["run"]["result_sha256"],
            "differing_fields": diffs[:20], "n_differing": len(diffs)}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="月度复跑：一条命令出四张表 + 一张图")
    ap.add_argument("--end", help="数据截止 YYYY-MM-DD（用于与上一期同输入对比；默认用最新行）")
    args = ap.parse_args()

    pc.assert_locked()  # 改坏锁死口径直接红，校验点收口在 pipeline_core

    data = pc.load_data()
    _, close_df = xsec.build_panels(data)
    base_start = xsec.common_start(close_df)
    data_end = close_df.index[-1]
    end = pd.Timestamp(args.end) if args.end else data_end
    if end > data_end:
        raise SystemExit(f"[区间错误] 指定截止 {end.date()} 晚于数据末行 {data_end.date()}")
    validation = pc.validate(data, base_start, end)

    cfg = pc.effective_config(args)
    cfg_sha = canonical(cfg)
    nav_df, windows = pc.walk_forward(data, base_start, end, cfg)

    perf = {key: pc.series_metrics(nav_df[key]) for key, _ in SERIES}
    turn = pc.turnover_tables(windows, perf)
    gaps = pc.gap_rows(perf)

    files = []
    for t in pc.UNIVERSE:
        p = pc.DATA_DIR / f"{t}.csv"
        files.append({"ticker": t, "file": p.name, "sha256": sha_file(p),
                      "rows": int(len(data[t])),
                      "first": str(data[t]["date"].iloc[0].date()),
                      "last": str(data[t]["date"].iloc[-1].date())})

    cfg_full = dict(cfg, resolved={"common_start": str(base_start.date()),
                                   "data_end": str(data_end.date()),
                                   "run_end": str(end.date()),
                                   "oos_start": str(nav_df.index[0].date()),
                                   "oos_end": str(nav_df.index[-1].date())})
    inp_fp = canonical({"files": [(f["file"], f["sha256"]) for f in files],
                        "common_start": cfg_full["resolved"]["common_start"],
                        "data_end": cfg_full["resolved"]["data_end"],
                        "run_end": cfg_full["resolved"]["run_end"]})
    payload = {"config_sha256": cfg_sha, "inputs_fingerprint": inp_fp,
               "performance": perf, "turnover": turn, "gaps": gaps,
               "windows": [{k: w[k] for k in WINDOW_FP_KEYS} for w in windows]}
    result_sha = canonical(payload)

    prev_path = find_previous()
    compare = build_compare(prev_path, cfg_full, cfg_sha, inp_fp, files, perf, windows)
    determ = determinism(prev_path, compare, result_sha,
                         {"performance": perf, "turnover": turn, "gaps": gaps}, windows)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(CHINA_TZ).strftime("%Y%m%d_%H%M%S")
    run_dir, i = RUNS_DIR / stamp, 0
    while run_dir.exists():
        i += 1
        run_dir = RUNS_DIR / f"{stamp}_{i:02d}"
    run_dir.mkdir(parents=True)

    scripts = (f"run_monthly.py {sha_file(ROOT / 'run_monthly.py')[:16]} / "
               f"pipeline_core.py {sha_file(ROOT / 'pipeline_core.py')[:16]} / "
               f"monthly_report.py {sha_file(ROOT / 'monthly_report.py')[:16]}")
    meta = {"run_id": run_dir.name,
            "built_at": datetime.now(CHINA_TZ).strftime("%Y-%m-%d %H:%M:%S"),
            "command": " ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]),
            "result_sha256": result_sha, "config_sha256": cfg_sha,
            "code_sha256": pc.code_sha256(), "scripts": scripts,
            "engine_sha256": sha_file(ROOT / "xsec.py")[:16],
            "python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
            **cfg_full["resolved"], "inputs_fingerprint": inp_fp}

    monthly_report.write_all(run_dir, nav_df, perf, turn, gaps, files,
                             pc.cfg_rows(cfg_full), meta, validation, compare, determ)
    metrics = {"run": meta, "effective_config": cfg_full,
               "inputs": {"data_dir": str(pc.DATA_DIR), "files": files,
                          **cfg_full["resolved"], "inputs_fingerprint": inp_fp},
               "validation": validation,
               "tables": {"performance": perf, "turnover": turn, "gaps": gaps},
               "windows": windows, "result_sha256": result_sha,
               "determinism": determ, "compare_vs_previous": compare}
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    s, f, c = perf["strategy"], perf["bench_same_start"], perf["bench_continuous"]
    print(f"\n完成: {run_dir}")
    print(f"result_sha256: {result_sha}")
    print(f"策略 CAGR {s['cagr']*100:.2f}% / 夏普 {s['sharpe']:.3f} / "
          f"MDD {s['max_drawdown']*100:.2f}% / Calmar {s['calmar']:.3f}")
    print(f"基准 同起点 {f['cagr']*100:.2f}% / 连续 {c['cagr']*100:.2f}%；"
          f"缺口 {(c['cagr']-s['cagr'])*1e4:+.0f}bp（基准−策略，正=跑输）")
    print(f"确定性: {determ['status']}")


if __name__ == "__main__":
    main()
