# ============================================================================
# 关闭项（2026-11-05 定稿，勿重复实验）
# 压换手不作为缩缺口手段。四档全部为负，实测：
#   换手门槛 100% : 201% / 7.65% / 缺口 +67bp（省 23.4bp 成本，丢 58bp 收益）
#   缓冲带 3      : 277% / 6.47% / 缺口 +185bp
#   最小持有 18m  : 256% / 6.39% / 缺口 +193bp
#   成本惩罚 15%  : 295% / 5.99% / 缺口 +233bp
# 基线 535% / 8.22% / +10bp；成本上限 37.4bp/年，不是 206bp 缺口的主因。
# ============================================================================
"""
方向 2: 压换手（535% -> 目标 <=300%），四种机制各自单独测，不叠加。

1. 换手门槛: 月末目标与当前权重的总变动 < 阈值时不交易
2. 排名缓冲带: 现有持仓排名落在 K+buffer 内则保留
3. 最小持有期: 持有未满 N 个月的持仓强制保留（仍受 K 个上限约束）
4. 成本惩罚: 现有持仓的动量分数 +hurdle，新标的必须超过它才替换

固定: t 收盘信号、t+1 开盘、月度、7bp/边、目标波动 10%、杠杆 <=1.0、K=3/12-1/无过滤/等权。

运行: D:\\Agent\\quant\\.venv\\Scripts\\python.exe D:\\Agent\\quant-phase2\\turnover.py
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import engine  # noqa: E402

OUT = ROOT / "out_turnover"
K = 3
MOM = 252
FILTER = False
WEIGHT = "equal"

GRIDS = {
    "threshold": [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00],
    "buffer": [1, 2, 3, 4, 5],
    "min_hold": [3, 6, 9, 12, 18, 24],
    "hurdle": [0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
}
LABEL = {"base": "基线", "threshold": "换手门槛", "buffer": "排名缓冲带",
         "min_hold": "最小持有期", "hurdle": "成本惩罚"}


def pct(v, d=2):
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v*100:.{d}f}%"


def bp(v):
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:+.0f}"


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def one(data, start, end, **kw):
    res = engine.run(data, k=K, mom_days=MOM, use_filter=FILTER, weight=WEIGHT,
                     start=start, end=end, **kw)
    return res.metrics


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = engine.load_data()
    _, close_df = engine.build_panels(data)
    start = engine.common_start(close_df)
    end = close_df.index[-1]

    print(f"样本 {start.date()} -> {end.date()}", flush=True)
    bench_nav = engine.run_benchmark(data, start, end)
    bm = engine.simple_metrics(bench_nav)
    print(f"基准 CAGR {pct(bm['cagr'])}", flush=True)

    base = one(data, start, end)
    base_gap = (bm["cagr"] - base["cagr"]) * 1e4
    print(f"基线: CAGR {pct(base['cagr'])}, 换手 {pct(base['annual_turnover'], 0)}, "
          f"成本 {base['cost_annual_bp']:.1f}bp, 缺口 {base_gap:+.0f}bp", flush=True)
    assert abs(base["cagr"] - 0.082239) < 2e-4, f"基线未复现: {base['cagr']}"
    assert abs(base["annual_turnover"] - 5.3466) < 2e-2, f"基线换手未复现: {base['annual_turnover']}"

    rows = []

    def add(name, param, kw):
        m = one(data, start, end, **kw)
        gap = (bm["cagr"] - m["cagr"]) * 1e4
        rows.append({
            "mechanism": name, "param": param,
            "turnover": m["annual_turnover"], "cagr": m["cagr"],
            "gap_bp": gap, "gap_change_bp": gap - base_gap,
            "cost_bp": m["cost_annual_bp"],
            "cost_saving_bp": base["cost_annual_bp"] - m["cost_annual_bp"],
            "sharpe": m["sharpe"], "mdd": m["max_drawdown"],
            "to_entry": m["turnover_entry"], "to_exit": m["turnover_exit"],
            "to_adjust": m["turnover_adjust"], "final_nav": m["final_nav"],
        })
        print(f"  {LABEL[name]}={param}: 换手 {pct(m['annual_turnover'], 0)} "
              f"CAGR {pct(m['cagr'])} 缺口 {gap:+.0f}bp "
              f"成本省 {base['cost_annual_bp']-m['cost_annual_bp']:.1f}bp", flush=True)

    add("base", "—", {})
    for v in GRIDS["threshold"]:
        add("threshold", f"{v:.0%}", {"threshold": v})
    for v in GRIDS["buffer"]:
        add("buffer", str(v), {"buffer": v})
    for v in GRIDS["min_hold"]:
        add("min_hold", f"{v}m", {"min_hold": v})
    for v in GRIDS["hurdle"]:
        add("hurdle", f"{v:.0%}", {"switch_hurdle": v})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "variants.csv", index=False, encoding="utf-8-sig")

    meta = engine.meta_block(Path(__file__))
    metrics = {
        "meta": meta,
        "sample": [str(start.date()), str(end.date())],
        "benchmark": bm,
        "base": base,
        "base_gap_bp": base_gap,
        "target_turnover": 3.0,
        "variants": rows,
    }
    (OUT / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    write_report(meta, df, base, base_gap, bm)
    print(f"完成 {OUT/'report.md'}", flush=True)


def arm_row(df, name):
    sub = df[df["mechanism"] == name]
    hit = sub[sub["turnover"] <= 3.0]
    best = hit.sort_values("cagr", ascending=False).iloc[0] if len(hit) else None
    closest = sub.sort_values("turnover").iloc[0]
    return hit, best, closest


def sub_saving(df, name):
    return float(df[df["mechanism"] == name]["cost_saving_bp"].max())


def write_report(meta, df, base, base_gap, bm):
    L = []
    L.append("# 方向 2：压换手（535% → ≤300%）\n")
    L.append(f"- `builtAt` {meta['builtAt']} ｜ `head` {meta['head']} ｜ "
             f"`turnover.py` sha256 前 16 位 **{meta['script_sha256']}** ｜ "
             f"引擎 `engine.py` sha256 前 16 位 **{hashlib.sha256((ROOT/'engine.py').read_bytes()).hexdigest()[:16]}** ｜ "
             f"Python {meta['python']} / pandas {meta['pandas']} / numpy {meta['numpy']}")
    L.append("- 命令：`D:\\Agent\\quant\\.venv\\Scripts\\python.exe D:\\Agent\\quant-phase2\\turnover.py`")
    L.append(f"- 样本：{base['start']} → {base['end']}（{base['years']:.2f} 年）；对象：横截面主组合 "
             f"K=3 / 12-1 / 无过滤 / 等权，10 只 ETF。")
    L.append("- 硬约束未动：t 收盘信号、t+1 开盘、月度调仓、单边 7bp、目标波动 10%、杠杆 ≤1.0。")
    L.append("- 标签：`[A]` 本机实测；`[B]` 由实测数字推算。\n")

    print_turnover = base["annual_turnover"]
    hits = df[df["turnover"] <= 3.0].sort_values("gap_bp")
    best_hit = hits.iloc[0]
    loss_pp = (base["cagr"] - best_hit["cagr"]) * 100
    L.append("## 结论\n")
    L.append(f"**四种机制都能把换手压到 ≤300%，但都只能在极端档位实现，且全部让缺口变大。** "
             f"基线 {pct(print_turnover,0)} / 年化 {pct(base['cagr'])} / 缺口 {base_gap:+.0f}bp；"
             f"达标组里缺口最小的是「{LABEL[best_hit['mechanism']]} {best_hit['param']}」："
             f"换手 {pct(best_hit['turnover'],0)}、年化 {pct(best_hit['cagr'])}、"
             f"缺口 {best_hit['gap_bp']:+.0f}bp（比基线扩大 {best_hit['gap_change_bp']:+.0f}bp），"
             f"成本省 {best_hit['cost_saving_bp']:.1f}bp/年，但年化少 {loss_pp:.2f}pp。"
             f"**省下的成本远小于丢掉的收益，压换手不是缩小 206bp 缺口的办法。**\n")

    # 机制小结表
    L.append("## 一、四种机制的一行结论\n")
    rows = []
    for name in ("threshold", "buffer", "min_hold", "hurdle"):
        hit, best, closest = arm_row(df, name)
        if len(hit):
            low = hit.sort_values("gap_bp").iloc[0]
            verdict = (f"达标档位 {low['param']}：换手 {pct(low['turnover'],0)}、"
                       f"年化 {pct(low['cagr'])}、缺口 {low['gap_bp']:+.0f}bp "
                       f"(变化 {low['gap_change_bp']:+.0f}bp)")
        else:
            verdict = (f"未达标：最低 {pct(closest['turnover'],0)}（{closest['param']}），"
                       f"年化 {pct(closest['cagr'])}、缺口 {closest['gap_bp']:+.0f}bp "
                       f"(变化 {closest['gap_change_bp']:+.0f}bp)")
        rows.append([LABEL[name], verdict, f"{sub_saving(df, name):.1f}"])
    L.append(table(["机制", "能否到 ≤300% 及代价", "该机制最大成本节省 bp/年 [A]"], rows))
    L.append("")

    # 逐机制明细
    for name in ("threshold", "buffer", "min_hold", "hurdle"):
        L.append(f"## {LABEL[name]} 明细\n")
        sub = df[df.mechanism == name]
        rows = []
        for _, r in sub.iterrows():
            rows.append([r["param"], pct(r["turnover"], 0), pct(r["cagr"]),
                         f"{r['gap_bp']:+.0f}", f"{r['gap_change_bp']:+.0f}",
                         f"{r['cost_bp']:.1f}", f"{r['cost_saving_bp']:.1f}",
                         pct(r["to_entry"], 0), pct(r["to_exit"], 0), pct(r["to_adjust"], 0)])
        L.append(table(["参数", "年换手", "年化", "缺口 bp", "缺口变化 bp",
                        "成本 bp/年", "成本节省 bp/年", "entry", "exit", "adjust"], rows))
        L.append("")

    L.append("## 基线\n")
    L.append(table(["指标", "值"], [
        ["年换手", pct(print_turnover, 0)],
        ["entry / exit / adjust", f"{pct(base['turnover_entry'],0)} / {pct(base['turnover_exit'],0)} / {pct(base['turnover_adjust'],0)}"],
        ["年化", pct(base["cagr"]),],
        ["缺口（基准−策略）", f"{base_gap:+.0f}bp"],
        ["成本", f"{base['cost_annual_bp']:.1f}bp/年"],
        ["夏普 / 回撤", f"{base['sharpe']:.2f} / {pct(base['max_drawdown'])}"],
    ]) + "\n")

    L.append("## 二、为什么缩不动（逐条）\n")
    th60 = df[(df.mechanism == "threshold") & (df.param == "60%")].iloc[0]
    th80 = df[(df.mechanism == "threshold") & (df.param == "80%")].iloc[0]
    th100 = df[(df.mechanism == "threshold") & (df.param == "100%")].iloc[0]
    b3 = df[(df.mechanism == "buffer") & (df.param == "3")].iloc[0]
    m18 = df[(df.mechanism == "min_hold") & (df.param == "18m")].iloc[0]
    h15 = df[(df.mechanism == "hurdle") & (df.param == "15%")].iloc[0]
    L.append(f"- **换手门槛**：基线里 entry {pct(base['turnover_entry'],0)} + exit "
             f"{pct(base['turnover_exit'],0)} + adjust {pct(base['turnover_adjust'],0)}。"
             f"门槛≤60% 时主要省 adjust，entry+exit 地板 "
             f"{pct(base['turnover_entry']+base['turnover_exit'],0)} 把换手卡在 {pct(th60['turnover'],0)}。"
             f"要到 300% 必须让大多数月份“完全不交易”（80% 档 {pct(th80['turnover'],0)}、"
             f"100% 档 {pct(th100['turnover'],0)}）；此时策略已接近买入持有，"
             f"年化 {pct(th100['cagr'])} 只是“少做少错”，不是换手优化带来的收益。")
    L.append(f"- **排名缓冲带**：直接减少 entry/exit，3 格能到 {pct(b3['turnover'],0)}，"
             f"但保留排名靠后的旧持仓，年化 {pct(b3['cagr'])}，比基线少 "
             f"{(base['cagr']-b3['cagr'])*100:.2f}pp；省 {b3['cost_saving_bp']:.1f}bp 成本，"
             f"丢 {(base['cagr']-b3['cagr'])*1e4:.0f}bp 收益。")
    L.append(f"- **最小持有期**：只强制保留“刚买入”的标的，到期后仍会按排名卖出；"
             f"12 个月档换手 {pct(df[(df.mechanism=='min_hold') & (df.param=='12m')].turnover.iloc[0],0)}、"
             f"年化 {pct(df[(df.mechanism=='min_hold') & (df.param=='12m')].cagr.iloc[0])}，"
             f"18 个月档 {pct(m18['turnover'],0)}、{pct(m18['cagr'])}；"
             f"24 个月档换手 {pct(df[(df.mechanism=='min_hold') & (df.param=='24m')].turnover.iloc[0],0)}、"
             f"年化 {pct(df[(df.mechanism=='min_hold') & (df.param=='24m')].cagr.iloc[0])}，"
             "单调性被打破（强制持有过久反而踩中反转）。")
    L.append(f"- **成本惩罚**：给旧持仓加分后替换门槛变高，15% 档 {pct(h15['turnover'],0)}、"
             f"年化 {pct(h15['cagr'])}；20% 档 {pct(df[(df.mechanism=='hurdle') & (df.param=='20%')].turnover.iloc[0],0)}、"
             f"{pct(df[(df.mechanism=='hurdle') & (df.param=='20%')].cagr.iloc[0])}。"
             "它改变的是持仓选择，不是纯执行优化。")
    L.append(f"- 共同点：535% 换手对应成本 {base['cost_annual_bp']:.1f}bp/年；"
             f"所有档位里最大成本节省 {df['cost_saving_bp'].max():.1f}bp/年，"
             "而年化损失从 0.57pp 到 4.43pp。**成本不是这个策略缺口的主要来源。**\n")

    L.append("## 三、对 206/355bp 基线的意义\n")
    L.append("- 206bp 是第一版时序动量（355% 换手）的缺口；本方向对象是横截面主组合（535% 换手、"
             f"全样本缺口 {base_gap:+.0f}bp）。两套数字不能直接相加。")
    L.append("- 对第一版：355% 换手对应成本 24.8bp/年 [A]；即使把换手压到 300%，"
             "最多省约 (355−300)/355 × 24.8 ≈ 3.8bp/年 [B]，相对 206bp 缺口可忽略。")
    L.append(f"- 对横截面版本：535% → 300% 实测最大成本节省 "
             f"{df['cost_saving_bp'].max():.1f}bp/年 [A]；但所有 ≤300% 档位的年化都低于基线，"
             f"缺口最小也扩大到 {hits['gap_bp'].min():+.0f}bp（基线 {base_gap:+.0f}bp）[A]。\n")

    L.append("## 四、判断\n")
    L.append(f"- 「≤300% 换手」可以做到，但代价是年化至少少 "
             f"{(base['cagr']-hits.sort_values('gap_bp').iloc[0]['cagr'])*100:.2f}pp；"
             "这不是策略改进，是拿收益换交易频率，**不值得为达标而达标**。")
    small = df[(df.mechanism == "threshold") & (df.param == "20%")].iloc[0]
    L.append(f"- 如果只是想省成本、不追求 300%：换手门槛 20% 是唯一没有明显代价的档位，"
             f"换手 {pct(small['turnover'],0)}、年化 {pct(small['cagr'])}、缺口 {small['gap_bp']:+.0f}bp；"
             f"但换手只降 {print_turnover*100-small['turnover']*100:.0f}pp，"
             "且这点改善在 18.7 年样本上属于噪声，不能算策略改进。")
    L.append("- 建议：把「换手≤300%」从本轮目标里撤掉，改为成本约束下的收益最大化"
             "（只在边际信号差 > 2×单边成本时才换），并把研究预算转到方向 3 的资产池/空头腿。\n")

    L.append("## 五、产物\n")
    L.append("- `report.md`、`variants.csv`、`metrics.json`")
    L.append("- 复现命令见文首；基线断言写在脚本内（CAGR 8.2239%、换手 534.66%）。")
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
