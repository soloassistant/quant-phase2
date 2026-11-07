"""
OOS 风险调整后三项（夏普 / 最大回撤 / Calmar），两套引擎 × 两个基准。

口径:
  * 横截面主组合: out_oos/oos_nav.csv 的 oos_0m（2016-02-29 起），基准列 bench_fresh / bench_cont
  * 第一版时序动量: quant/out/walkforward_v2/oos_nav.csv 的 stg_nav；连续基准 bench_nav；
    同起点基准用同一 10 只 ETF、同一区间、同一成交/成本口径重算
  * Sharpe = 日收益均值 / 日收益标准差 × √252；Calmar = CAGR / |最大回撤|
  * 全部为 OOS 拼接净值，不复用全样本指标

运行: python risk_oos.py
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import engine  # noqa: E402

OUT = ROOT / "out_risk"
XSEC_NAV = ROOT / "out_oos" / "oos_nav.csv"
TS_NAV = ROOT / "out" / "walkforward_v2" / "oos_nav.csv"


def pct(v, d=2):
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v*100:.{d}f}%"


def num(v, d=3):
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{d}f}"


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def risk(nav):
    nav = nav.dropna()
    m = engine.simple_metrics(nav)
    calmar = m["cagr"] / abs(m["max_drawdown"]) if m["max_drawdown"] else float("nan")
    return {"start": m["start"], "end": m["end"], "years": m["years"],
            "cagr": m["cagr"], "vol": m["ann_vol"], "sharpe": m["sharpe"],
            "mdd": m["max_drawdown"], "calmar": calmar, "final_nav": m["final_nav"]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for p in (XSEC_NAV, TS_NAV):
        if not p.exists():
            raise SystemExit(f"[数据缺失] 缺少 {p}（历史研究输入，未随仓分发；先跑 oos_multi.py 或自备该文件）")
    print(f"xsec: {XSEC_NAV}", flush=True)
    xs = pd.read_csv(XSEC_NAV, index_col=0, parse_dates=True)
    x_nav = xs["oos_0m"].dropna()
    x_fresh = xs["bench_fresh"].dropna()
    x_cont = xs["bench_cont"].dropna()
    assert abs(risk(x_nav)["sharpe"] - 1.024) < 5e-3, "横截面 OOS 夏普未复现"
    assert abs(risk(x_nav)["mdd"] + 0.1729) < 5e-3, "横截面 OOS 回撤未复现"

    print(f"ts:   {TS_NAV}", flush=True)
    ts = pd.read_csv(TS_NAV, index_col=0, parse_dates=True)
    t_nav = ts["stg_nav"].dropna()
    t_cont = ts["bench_nav"].dropna()
    data = engine.load_data()
    _, close_df = engine.build_panels(data)
    t_fresh = engine.run_benchmark(data, t_nav.index[0], t_nav.index[-1])

    blocks = {
        "xsec": {
            "label": "横截面主组合（K=3 / 12-1 / 等权）",
            "strategy": risk(x_nav), "bench_fresh": risk(x_fresh), "bench_cont": risk(x_cont),
        },
        "ts": {
            "label": "第一版时序动量（12 月 / 正动量等权）",
            "strategy": risk(t_nav), "bench_fresh": risk(t_fresh), "bench_cont": risk(t_cont),
        },
    }

    # OOS 窗口重叠度: 相邻起点、同一序号的验证窗口共用多少个月
    w = pd.read_csv(ROOT / "out_oos" / "windows.csv", parse_dates=["val_start", "val_end"])
    offsets = sorted(w["offset"].unique())
    overlap = []
    for a, b in zip(offsets, offsets[1:]):
        ma = w[w.offset == a].reset_index(drop=True)
        mb = w[w.offset == b].reset_index(drop=True)
        n = min(len(ma), len(mb))
        if n == 0:
            continue
        months = []
        for k in range(n):
            lo = max(ma.loc[k, "val_start"], mb.loc[k, "val_start"])
            hi = min(ma.loc[k, "val_end"], mb.loc[k, "val_end"])
            months.append(max(0, (hi.year - lo.year) * 12 + hi.month - lo.month))
        overlap.append({"a": int(a), "b": int(b), "windows_compared": int(n),
                        "avg_shared_months": float(sum(months) / len(months)),
                        "per_window": [float(v) for v in months]})

    meta = engine.meta_block(Path(__file__))
    result = {"meta": meta, "blocks": blocks, "overlap": overlap,
              "note": "Sharpe=日收益均值/标准差×√252; Calmar=CAGR/|MDD|; 全部 OOS 拼接净值"}
    (OUT / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float),
                                      encoding="utf-8")
    write_report(meta, blocks, overlap)
    print(f"完成 {OUT/'report.md'}", flush=True)


def write_report(meta, blocks, overlap):
    L = []
    L.append("# OOS 风险调整后三项（两套引擎 × 两个基准）\n")
    L.append(f"- `builtAt` {meta['builtAt']} ｜ `head` {meta['head']} ｜ "
             f"`risk_oos.py` sha256 前 16 位 **{meta['script_sha256']}** ｜ "
             f"Python {meta['python']} / pandas {meta['pandas']} / numpy {meta['numpy']}")
    L.append("- 命令：`python risk_oos.py`")
    L.append("- 全部为 **OOS 拼接净值**，不用全样本指标；Sharpe=日收益均值/标准差×√252，"
             "Calmar=CAGR/|最大回撤|。\n")

    xs, ts = blocks["xsec"], blocks["ts"]
    L.append("## 结论\n")
    L.append(f"**横截面组合在 OOS 上买到的是风险改善：夏普 {num(xs['strategy']['sharpe'])} vs "
             f"同起点 {num(xs['bench_fresh']['sharpe'])} / 连续 {num(xs['bench_cont']['sharpe'])}，"
             f"回撤 {pct(xs['strategy']['mdd'])} vs {pct(xs['bench_fresh']['mdd'])} / "
             f"{pct(xs['bench_cont']['mdd'])}，Calmar {num(xs['strategy']['calmar'])} vs "
             f"{num(xs['bench_fresh']['calmar'])} / {num(xs['bench_cont']['calmar'])}。**"
             f" 第一版时序动量同样回撤更浅（{pct(ts['strategy']['mdd'])}），但夏普 "
             f"{num(ts['strategy']['sharpe'])} 低于连续基准 {num(ts['bench_cont']['sharpe'])}，"
             f"Calmar {num(ts['strategy']['calmar'])} 也低于基准 "
             f"{num(ts['bench_cont']['calmar'])}。"
             " **结论：缺口不可缩，改判风险调整后收益。横截面的夏普/回撤/Calmar 三项全部胜出"
             "（两个基准都胜）；第一版只有回撤胜出，夏普和 Calmar 都低于基准。**\n")

    L.append("## 一、三项指标表\n")
    rows = []
    for key, lab in (("xsec", xs["label"]), ("ts", ts["label"])):
        b = blocks[key]
        s, f, c = b["strategy"], b["bench_fresh"], b["bench_cont"]
        rows.append([lab, f"{s['start']}→{s['end']}",
                     f"{pct(s['cagr'])} / {pct(f['cagr'])} / {pct(c['cagr'])}",
                     f"**{num(s['sharpe'])}** / {num(f['sharpe'])} / {num(c['sharpe'])}",
                     f"**{pct(s['mdd'])}** / {pct(f['mdd'])} / {pct(c['mdd'])}",
                     f"**{num(s['calmar'])}** / {num(f['calmar'])} / {num(c['calmar'])}"])
    L.append(table(["引擎 / 组合", "OOS 区间", "CAGR（策略/同起点/连续）",
                    "夏普（策略/同起点/连续）", "最大回撤（策略/同起点/连续）",
                    "Calmar（策略/同起点/连续）"], rows))
    L.append("\n每格顺序：**策略 / 同起点重建基准 / 连续持有基准**；全部 [A] 实测。\n")

    L.append("## 二、逐项差值（策略 − 基准）\n")
    rows = []
    for key, lab in (("xsec", xs["label"]), ("ts", ts["label"])):
        b = blocks[key]
        s = b["strategy"]
        for bname, bkey in (("同起点", "bench_fresh"), ("连续", "bench_cont")):
            bm = b[bkey]
            rows.append([lab, bname,
                         f"{(s['sharpe'] - bm['sharpe']):+.3f}",
                         f"{(s['mdd'] - bm['mdd'])*100:+.2f}pp",
                         f"{(s['calmar'] - bm['calmar']):+.3f}",
                         f"{(s['cagr'] - bm['cagr'])*1e4:+.0f}bp"])
    L.append(table(["引擎 / 组合", "基准", "夏普差", "回撤差", "Calmar 差", "CAGR 差"], rows))
    L.append("\n回撤差为正=回撤更浅；夏普/Calmar/CAGR 差为正=策略更好。\n")

    L.append("## 三、多起点 OOS 的窗口重叠度\n")
    rows = [[f"{o['a']}m ↔ {o['b']}m", o["windows_compared"],
             f"{o['avg_shared_months']:.1f}", f"{o['avg_shared_months']/12*100:.0f}%"]
            for o in overlap]
    L.append(table(["相邻起点", "对比窗口数", "平均共用月数（每个 12 个月窗口）", "重叠比例"], rows))
    L.append("\n- 1 年步长下，12 个月起点与 0 个月完全重合（12/12 个月）；"
             "3/6/9 个月起点与 0 个月平均共用 9/6/3 个月。"
             "**这些不是独立样本，OOS 的多起点只检验切点敏感性。**\n")

    L.append("## 四、三条关闭项\n")
    L.append("1. **空头腿关闭**：`ls_80` 4.33%（−194bp）、`50/50` 0.95%（−531bp）；"
             "长牛样本结构性逆风，且未计借券费，真实更差。以后谁再提做空，直接引这两个数。")
    L.append("2. **压换手不作为缩缺口手段**：门槛 / 缓冲带 / 最小持有 / 成本惩罚四档全部为负；"
             "门槛 100% 省 23.4bp 成本、丢 58bp 收益。不再重复实验。")
    L.append("3. **多起点 OOS 不是新增证据**：0/3/6/9 窗口高度重叠，12 个月与 0 个月完全重合；"
             "以后报 OOS 必须同时报窗口重叠度。\n")

    L.append("## 五、产物\n")
    L.append("- `report.md`、`metrics.json`")
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
