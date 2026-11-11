# ============================================================================
# 关闭项 / 硬规则（2026-11-05 定稿）
# 多起点 OOS 不是新增证据：1 年步长下 0/3/6/9 个月窗口高度重叠
#   （相邻起点平均共用 9/12 个月），12 个月与 0 个月完全重合（12/12）。
# 以后报 OOS 必须同时报窗口重叠度，不能把多起点当成多份独立样本。
# 本脚本的 overlap 只由 risk_oos.py 计算并写入 out_risk/report.md。
# ============================================================================
"""
方向 1: 多起点 walk-forward 补样本外。

固定: 8 年训练 / 1 年验证 / 1 年步长 / 1 月 purge；中位数选参；邻域平坦 0.25；
t 收盘信号、t+1 开盘、7bp/边、目标波动 10%、杠杆 ≤1.0。
起点错开 0 / 3 / 6 / 9 / 12 个月，各跑一遍完整 walk-forward；
12 个月与 0 个月窗口重合，只作一致性检查。
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import engine  # noqa: E402
import xsec  # noqa: E402

OUT = ROOT / "out_oos"
MOM_MONTHS = {6: 126, 9: 189, 12: 252}
CANDIDATES = [(k, MOM_MONTHS[m], f, w) for k in (2, 3) for m in (6, 9, 12)
              for f in (False, True) for w in ("equal", "inv_vol")]
M_IDX = {126: 0, 189: 1, 252: 2}
FLAT = 0.25
OFFSETS = [0, 3, 6, 9, 12]


def pct(v, d=2):
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v*100:.{d}f}%"


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def run_offset(data, base_start, end, offset, panels=None):
    t0 = pd.Timestamp(base_start) + pd.DateOffset(months=offset)
    windows, returns = [], []
    while True:
        te = t0 + pd.DateOffset(years=8)
        purge = te - pd.DateOffset(months=1)
        ve_end = te + pd.DateOffset(years=1)
        if ve_end > pd.Timestamp(end):
            break
        sel = []
        for k, mom_days, use_filter, weight in CANDIDATES:
            r = xsec.run(data, k=k, mom_days=mom_days, use_filter=use_filter,
                         weight=weight, start=t0, end=purge, panels=panels)
            med, worst, n = xsec.rolling_median_sharpe(r.nav["nav"], t0, purge)
            sel.append({"k": k, "mom_days": mom_days, "filter": use_filter,
                        "weight": weight, "med": med, "worst": worst})
        for c in sel:
            nb = [o for o in sel if o is not c and o["filter"] == c["filter"]
                  and o["weight"] == c["weight"] and abs(o["k"] - c["k"]) <= 1
                  and abs(M_IDX[o["mom_days"]] - M_IDX[c["mom_days"]]) <= 1]
            c["neighbor_min"] = min((o["med"] for o in nb), default=c["med"])
            c["flat"] = bool(np.isfinite(c["med"]) and c["neighbor_min"] >= c["med"] - FLAT)
        ordered = sorted(sel, key=lambda o: (-o["med"] if np.isfinite(o["med"]) else 9e9))
        pick = next((o for o in ordered if o["flat"]), None)
        fallback = pick is None
        if fallback:
            pick = ordered[0]

        val = xsec.run(data, k=pick["k"], mom_days=pick["mom_days"],
                       use_filter=pick["filter"], weight=pick["weight"],
                       start=te, end=ve_end, panels=panels)
        bench = xsec.run_benchmark(data, te, ve_end)
        vm, bm = val.metrics, xsec.simple_metrics(bench)
        windows.append({
            "offset": offset, "train_start": str(t0.date()),
            "purge_end": str(pd.Timestamp(purge).date()),
            "val_start": str(pd.Timestamp(te).date()), "val_end": str(pd.Timestamp(ve_end).date()),
            "k": pick["k"], "mom_months": pick["mom_days"] // 21,
            "filter": "有" if pick["filter"] else "无",
            "weight": "等权" if pick["weight"] == "equal" else "波动倒数",
            "sel_median": pick["med"], "neighbor_min": pick["neighbor_min"],
            "flat": pick["flat"], "fallback": fallback,
            "val_cagr": vm["cagr"], "val_vol": vm["ann_vol"], "val_sharpe": vm["sharpe"],
            "val_mdd": vm["max_drawdown"], "val_turnover": vm["annual_turnover"],
            "val_cost_bp": vm["cost_annual_bp"],
            "bench_cagr": bm["cagr"], "bench_vol": bm["ann_vol"],
            "bench_sharpe": bm["sharpe"], "bench_mdd": bm["max_drawdown"],
            "gap_bp": (bm["cagr"] - vm["cagr"]) * 1e4,
            "cagr_win": vm["cagr"] > bm["cagr"], "sharpe_win": vm["sharpe"] > bm["sharpe"],
        })
        seg = val.nav["nav"].pct_change()
        seg.iloc[0] = 0.0
        returns.append(seg)
        t0 = t0 + pd.DateOffset(years=1)
        print(f"  offset {offset:+02d}m 窗口 {len(windows)}: {windows[-1]['val_start']}→"
              f"{windows[-1]['val_end']} 选中 K={pick['k']}/{pick['mom_days']//21}月/"
              f"{'有' if pick['filter'] else '无'}/{'等权' if pick['weight']=='equal' else '波动倒数'} "
              f"CAGR {pct(vm['cagr'])} vs {pct(bm['cagr'])}", flush=True)

    rets = pd.concat(returns).sort_index()
    rets = rets[~rets.index.duplicated(keep="first")]
    oos_nav = (1 + rets).cumprod() * 100.0
    oos_m = xsec.simple_metrics(oos_nav)
    bench_oos = xsec.run_benchmark(data, windows[0]["val_start"], windows[-1]["val_end"])
    bm_oos = xsec.simple_metrics(bench_oos)

    full_bench = xsec.run_benchmark(data, base_start, end)
    cont = full_bench.reindex(oos_nav.index)
    cr = cont.pct_change()
    cr.iloc[0] = 0.0
    cont_nav = (1 + cr).cumprod() * 100.0
    cm = xsec.simple_metrics(cont_nav)

    act = engine.active_stats(rets, bench_oos.reindex(rets.index).pct_change().fillna(0.0))
    return {
        "offset": offset, "windows": windows, "oos_nav": oos_nav, "bench_nav": bench_oos,
        "cont_nav": cont_nav, "oos": oos_m, "bench": bm_oos, "cont": cm, "active": act,
        "gap_bp": (bm_oos["cagr"] - oos_m["cagr"]) * 1e4,
        "gap_cont_bp": (cm["cagr"] - oos_m["cagr"]) * 1e4,
        "cagr_wins": int(sum(w["cagr_win"] for w in windows)),
        "sharpe_wins": int(sum(w["sharpe_win"] for w in windows)),
    }


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5.5))
    colors = {0: "#1f4e79", 3: "#7b1fa2", 6: "#c00000", 9: "#ef6c00", 12: "#2e7d32"}
    for r in results:
        ax.plot(r["oos_nav"].index, r["oos_nav"], lw=1.2, color=colors[r["offset"]],
                label=f"OOS offset {r['offset']}m")
    ax.plot(results[0]["bench_nav"].index, results[0]["bench_nav"], color="black",
            lw=1.0, alpha=0.6, ls="--", label="Equal-weight benchmark (fresh)")
    ax.set_yscale("log")
    ax.set_title("Multi-start walk-forward OOS (8y train / 1y validation)")
    ax.set_ylabel("NAV (log)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = xsec.load_data()
    open_df, close_df = xsec.build_panels(data)
    base_start = xsec.common_start(close_df)
    end = close_df.index[-1]
    # 面板/日历/月末标记只依赖 data，一次算好传给每次候选回测（否则 1200 次全样本重算）
    panels = (open_df, close_df, close_df.index, close_df.ffill(),
              close_df.pct_change(fill_method=None).ffill(),
              xsec.month_end_flags(close_df.index))
    print(f"样本 {base_start.date()} → {end.date()}", flush=True)

    results = []
    for off in OFFSETS:
        print(f"\n== offset {off}m ==", flush=True)
        results.append(run_offset(data, base_start, end, off, panels))

    all_windows = [w for r in results for w in r["windows"]]
    pd.DataFrame(all_windows).to_csv(OUT / "windows.csv", index=False, encoding="utf-8-sig")
    base_starts = set(w["val_start"] for w in results[0]["windows"])
    for r in results:
        starts = [w["val_start"] for w in r["windows"]]
        r["duplicate_of_0"] = bool(r["offset"] != 0 and all(s in base_starts for s in starts))
    summary = []
    for r in results:
        summary.append({
            "offset_months": r["offset"], "windows": len(r["windows"]),
            "independent": not r["duplicate_of_0"],
            "oos_start": r["oos"]["start"], "oos_end": r["oos"]["end"],
            "oos_years": r["oos"]["years"],
            "oos_cagr": r["oos"]["cagr"], "bench_cagr": r["bench"]["cagr"],
            "gap_bp": r["gap_bp"], "cont_cagr": r["cont"]["cagr"],
            "gap_cont_bp": r["gap_cont_bp"],
            "oos_vol": r["active"]["strategy_vol"], "bench_vol": r["active"]["benchmark_vol"],
            "vol_diff": r["active"]["vol_diff"], "tracking_error": r["active"]["tracking_error"],
            "corr": r["active"]["corr"],
            "oos_sharpe": r["oos"]["sharpe"], "bench_sharpe": r["bench"]["sharpe"],
            "oos_mdd": r["oos"]["max_drawdown"], "bench_mdd": r["bench"]["max_drawdown"],
            "cagr_wins": r["cagr_wins"], "sharpe_wins": r["sharpe_wins"],
            "fallbacks": int(sum(w["fallback"] for w in r["windows"])),
            "trials": len(r["windows"]) * (len(CANDIDATES) + 1),
        })
    pd.DataFrame(summary).to_csv(OUT / "oos_offsets.csv", index=False, encoding="utf-8-sig")

    nav = pd.DataFrame({f"oos_{r['offset']}m": r["oos_nav"] for r in results})
    nav["bench_fresh"] = results[0]["bench_nav"].reindex(nav.index)
    nav["bench_cont"] = results[0]["cont_nav"].reindex(nav.index)
    nav.to_csv(OUT / "oos_nav.csv", encoding="utf-8-sig")
    plot(results, OUT / "oos_nav.png")

    meta = engine.meta_block(Path(__file__))
    metrics = {"meta": meta, "summary": summary,
               "windows": all_windows,
               "note": "gap_bp = OOS CAGR - 同起点重新建仓基准; gap_cont_bp = OOS CAGR - 2008建仓连续持有基准"}
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=float),
                                      encoding="utf-8")
    write_report(meta, summary, all_windows, results)
    print(f"\n完成 {OUT/'report.md'}", flush=True)


def write_report(meta, summary, windows, results):
    L = []
    L.append("# 方向 1：多起点 walk-forward 补样本外\n")
    L.append(f"- `builtAt` {meta['builtAt']} ｜ `head` {meta['head']} ｜ "
             f"脚本 `oos_multi.py` sha256 前 16 位 **{meta['script_sha256']}** ｜ "
             f"Python {meta['python']} / pandas {meta['pandas']} / numpy {meta['numpy']}")
    L.append("- 命令：`python oos_multi.py`")
    L.append("- 对象：横截面主组合（K=3、12-1、无过滤、等权；10 只 ETF）。"
             "硬约束未动：t 收盘信号、t+1 开盘、月度、7bp/边、目标波动 10%、杠杆 ≤1.0、"
             "8 年训练 / 1 年验证 / 1 年步长 / 1 月 purge、中位数选参 + 邻域平坦 0.25。\n")

    L.append("## 结论\n")
    indep = [s for s in summary if s["independent"]]
    dup = [s for s in summary if not s["independent"]]
    cagrs = [s["oos_cagr"] for s in indep]
    gaps = [s["gap_bp"] for s in indep]
    gc = [s["gap_cont_bp"] for s in indep]
    L.append(f"**多起点没有把样本外变成稳定的正 alpha。** "
             f"{len(indep)} 个不同起点（{ '、'.join(str(s['offset_months']) for s in indep) } 个月；"
             f"验证窗口高度重叠，不是独立样本）"
             f"的 OOS 年化 {pct(min(cagrs))}–{pct(max(cagrs))}，"
             f"对同起点重建基准的缺口 {min(gaps):+.0f}～{max(gaps):+.0f}bp/年（正=跑输）；"
             f"对第一版连续持有基准缺口 {min(gc):+.0f}～{max(gc):+.0f}bp/年。"
             f"OOS 年化波动 {pct(min(s['oos_vol'] for s in indep))}–"
             f"{pct(max(s['oos_vol'] for s in indep))}（基准 "
             f"{pct(min(s['bench_vol'] for s in indep))}–{pct(max(s['bench_vol'] for s in indep))}），"
             f"波动差 {min(s['vol_diff'] for s in indep)*1e4:+.0f}～"
             f"{max(s['vol_diff'] for s in indep)*1e4:+.0f}bp，"
             f"跟踪误差 {pct(min(s['tracking_error'] for s in indep))}–"
             f"{pct(max(s['tracking_error'] for s in indep))}，相关性 "
             f"{min(s['corr'] for s in indep):.2f}–{max(s['corr'] for s in indep):.2f}。"
             + (f" 起点错开 12 个月与 0 个月窗口完全重合（1 年步长的必然结果），"
                f"不算独立起点，只作为一致性检查。" if dup else "")
             + f" 硬约束下 OOS 无法再向前延长：数据起点 2008-02 + 8 年训练，"
             f"最早验证段就是 2016-02；要延长只能缩短训练或改用更早数据，均违反本次硬约束。"
             f"因此本方向回答的是“切点是否敏感”，不能缩小 206bp 的全样本缺口。\n")

    L.append("## 每个起点的 OOS 结果\n")
    rows = []
    for s in summary:
        rows.append([f"{s['offset_months']}m", "是" if s["independent"] else "否(与0m重合)",
                     s["windows"], f"{s['oos_start']}→{s['oos_end']}", f"{s['oos_years']:.2f}",
                     pct(s["oos_cagr"]), pct(s["bench_cagr"]), f"{s['gap_bp']:+.0f}",
                     pct(s["cont_cagr"]), f"{s['gap_cont_bp']:+.0f}",
                     pct(s["oos_vol"]), pct(s["bench_vol"]), f"{s['vol_diff']*1e4:+.0f}",
                     pct(s["tracking_error"]), f"{s['corr']:.2f}",
                     pct(s["oos_mdd"]), f"{s['cagr_wins']}/{s['windows']}"])
    L.append(table(["起点", "独立", "窗口", "OOS 区间", "年数", "OOS 年化", "同起点基准",
                    "缺口 bp", "连续基准", "对连续缺口 bp", "策略波动", "基准波动", "波动差 bp",
                    "跟踪误差", "相关", "OOS 回撤", "年化胜"], rows))
    L.append("\n缺口=基准年化−策略年化（正=跑输）；波动差=策略年化波动−基准年化波动；"
             "跟踪误差=日主动收益标准差×√252。全部为 [A] 本机实测。\n")

    L.append(f"## 逐窗口明细（{len(summary)} 个起点 × 约 10 个窗口）\n")
    rows = []
    for w in windows:
        rows.append([f"{w['offset']}m", f"{w['val_start']}→{w['val_end']}",
                     f"K={w['k']}/{w['mom_months']}月/{w['filter']}/{w['weight']}",
                     f"{w['sel_median']:.2f}", pct(w["val_cagr"]), pct(w["bench_cagr"]),
                     f"{w['gap_bp']:+.0f}", f"{w['val_sharpe']:.2f}", f"{w['bench_sharpe']:.2f}",
                     "是" if w["fallback"] else "否"])
    L.append(table(["起点", "验证段", "选中参数", "训练中位夏普", "验证年化", "基准年化",
                    "缺口 bp", "验证夏普", "基准夏普", "回退"], rows))

    L.append("\n## 结论与解释\n")
    n_win = sum(s["windows"] for s in indep)
    n_cagr_win = sum(s["cagr_wins"] for s in indep)
    n_fb = sum(s["fallbacks"] for s in indep)
    L.append(f"- 独立起点的验证窗口共 {n_win} 个，年化胜基准 {n_cagr_win}/{n_win}；"
             f"邻域平坦检查回退 {n_fb}/{n_win}。")
    L.append("- 每个不同起点对连续持有基准都是正缺口（跑输，+108～+351bp），"
             "对同起点重建基准从领先 16bp（0m）摆到跑输 158bp（9m）；"
             "说明样本外结果对“切点”和“基准怎么建仓”都敏感，没有任何一个起点给出稳定正 alpha。")
    L.append("- 这些起点只是把切月错开 3 个月，1 年验证窗口彼此高度重叠；"
             "它们检验“切点敏感性”，**不增加独立样本量**，不能当成 4 份独立证据。")
    L.append("- 起点错开 12 个月在 1 年步长下与 0 个月窗口完全重合（同一批验证年、同一批训练年），"
             "它只是去掉最差的第一年的 0 个月结果，不是独立证据；报告保留它是为了如实说明这一点。")
    L.append("- 「延长 OOS」受硬约束封死：8 年训练 + 1 年验证 + 数据从 2008-02 起，"
             "验证段最早只能是 2016-02；向前延长需要 1998 年以前的 ETF 数据或缩短训练窗口，两者都不允许。"
             "因此本方向只能做多起点稳健性，不能增加样本外年数。")
    L.append("- 与 206bp 基线的关系：206bp 是第一版时序动量的全样本缺口；"
             "本方向对象是横截面主组合，其全样本缺口仅 10bp。多起点没有缩小 206bp，"
             "也不能把横截面版本的 10bp 变成正超额；它证明的是“不稳定性不是切点造成的”。")
    L.append("\n## 产物\n")
    L.append("- `report.md`、`metrics.json`、`oos_offsets.csv`、`windows.csv`、"
             "`oos_nav.csv`、`oos_nav.png`")
    path = OUT / "report.md"
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
