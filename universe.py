# ============================================================================
# 关闭项（2026-11-05 定稿，勿重复实验）
# 空头腿关闭：长牛样本结构性逆风，且未计借券费，真实更差。实测：
#   ls_80（多 80/空 20）: 年化 4.33%，相对只做多基线 -194bp，缺口 399bp
#   ls_dn（多空 50/50）  : 年化 0.95%，相对基线 -531bp，缺口 737bp
# 以后谁再提做空，直接引这两个数。
# ============================================================================
"""
方向 3: 换策略池（加资产类别 / 允许空头 / 去掉幸存者偏差）。

对象: 第一版时序动量（12 个月动量、正动量等权、20 日协方差目标波动 10%、
t 收盘信号、t+1 开盘、月度、单边 7bp、杠杆 ≤1.0）。缺口基线 206bp、换手 355%。

三个子项各自单独测，不叠加:
  A. 加资产类别  -> 需要额外 ETF 数据；本次网络被限流/拒绝，标 BLOCKED，不编造数据
  B. 允许对冲/空头 -> 同一 10 只 ETF，多正动量、空最差负动量，总敞口 <=1.0
  C. 当时可得的池 -> 每个调仓日只允许“上市满 3 年”的标的进入池（point-in-time 可用性）

运行: D:\\Agent\\quant\\.venv\\Scripts\\python.exe D:\\Agent\\quant-phase2\\universe.py
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

OUT = ROOT / "out_universe"
COST = engine.COST_RATE
MOM = 252
TARGET_VOL = 0.10
MIN_HIST_YEARS = 3.0


def pct(v, d=2):
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v*100:.{d}f}%"


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def run_ts(data, long_short=False, short_frac=0.0, short_n=3, min_hist_years=0.0,
           start=None, end=None):
    open_df, close_df = engine.build_panels(data)
    cal = close_df.index
    close_ff = close_df.ffill()
    ret_f = close_df.pct_change(fill_method=None).ffill()
    cols = list(close_df.columns)
    first = {c: close_ff[c].first_valid_index() for c in cols}

    start_idx = 0
    if start is not None:
        start_idx = max(0, int(cal.get_indexer([pd.Timestamp(start)], method="pad")[0]))
    end_idx = len(cal) - 1 if end is None else max(0, int(cal.get_indexer([pd.Timestamp(end)], method="pad")[0]))
    flags = engine.month_end_flags(cal)

    cash, shares = 100.0, pd.Series(0.0, index=cols)
    last_target = pd.Series(0.0, index=cols)
    pending = None
    rows, trades, signals = [], [], []

    def available(c, d):
        if first[c] is None:
            return False
        if min_hist_years <= 0:
            return True
        return pd.Timestamp(d) >= pd.Timestamp(first[c]) + pd.DateOffset(days=int(365.25 * min_hist_years))

    for i in range(start_idx, end_idx + 1):
        d = cal[i]
        if pending is not None:
            sig_date, target = pending
            op = open_df.loc[d]
            op = op.where(op.notna() & (op > 0), close_ff.loc[d])
            cur_val = pd.Series({t: shares[t] * op[t] for t in cols})
            tradable = op.notna() & (op > 0)
            w = target.where(tradable, 0.0)
            pv = cash + float(cur_val.sum())
            cost, target_val = 0.0, pd.Series(0.0, index=cols)
            for _ in range(4):
                target_val = w * (pv - cost)
                cost = float((target_val - cur_val).abs().sum()) * COST
            trade_val = target_val - cur_val
            gross = float(trade_val.abs().sum())
            cash = cash - float(trade_val.sum()) - cost
            for t in cols:
                shares[t] = target_val[t] / op[t] if bool(tradable[t]) else 0.0
            trades.append({"signal_date": pd.Timestamp(sig_date).date(),
                           "exec_date": pd.Timestamp(d).date(),
                           "gross_traded": gross, "cost": float(cost),
                           "entry_traded": 0.0, "exit_traded": 0.0,
                           "adjust_traded": gross, "nav_before": float(pv),
                           "n_holdings_after": int((w != 0).sum())})
            last_target = w.copy()
            pending = None

        px = close_ff.loc[d]
        nav = cash + float((shares * px).sum())
        rows.append({"date": d, "nav": nav, "cash": cash,
                     "exposure": float((shares * px).sum()) / nav if nav > 0 else 0.0,
                     "n_holdings": int((shares.abs() > 1e-9).sum())})

        if flags[i] and i + 1 <= end_idx:
            mom = close_ff.iloc[i] / close_ff.iloc[i - MOM] - 1.0 if i - MOM >= 0 else pd.Series(np.nan, index=cols)
            vol_ok = ret_f.iloc[i - engine.VOL_DAYS + 1: i + 1].std(ddof=1) > 0
            avail = pd.Series({c: available(c, d) for c in cols})
            valid = mom.notna() & (mom > 0) & vol_ok & avail
            longs = [c for c in cols if bool(valid.get(c, False))]
            w = pd.Series(0.0, index=cols)
            if not longs:
                if long_short:
                    neg = [c for c in cols if bool(mom.notna()[c] and mom[c] < 0 and vol_ok[c] and avail[c])]
                    neg = sorted(neg, key=lambda c: float(mom[c]))[:short_n]
                    if neg:
                        w[neg] = -min(short_frac, 1.0) / len(neg)
            else:
                if long_short:
                    neg = [c for c in cols if bool(mom.notna()[c] and mom[c] < 0 and vol_ok[c] and avail[c])]
                    neg = sorted(neg, key=lambda c: float(mom[c]))[:short_n]
                    lg = 1.0 - short_frac if neg else 1.0
                    w[longs] = lg / len(longs)
                    if neg:
                        w[neg] = -short_frac / len(neg)
                else:
                    w[longs] = 1.0 / len(longs)

            picked = [c for c in cols if float(w[c]) != 0.0]
            if picked:
                win = ret_f.iloc[i - engine.VOL_DAYS + 1: i + 1][picked]
                cov = win.cov().to_numpy(dtype=float) * engine.TRADING_DAYS
                wv = w[picked].to_numpy(dtype=float)
                var = float(wv @ cov @ wv)
                est = math.sqrt(max(var, 0.0)) if np.isfinite(var) else float("nan")
                if not np.isfinite(est) or est <= 1e-8:
                    single = float((win.std(ddof=1) * math.sqrt(engine.TRADING_DAYS)).mean())
                    est = single if np.isfinite(single) else TARGET_VOL
                w = w * min(TARGET_VOL / est, 1.0)

            signals.append({"date": d, "n_long": len(longs), "n_short": int((w < 0).sum()),
                            "gross": float(w.abs().sum()), "net": float(w.sum()),
                            "n_avail": int(avail.sum()), "sum_weight": float(w.abs().sum())})
            pending = (d, w)

    nav_df = pd.DataFrame(rows).set_index("date")
    nav_df["nav"] = nav_df["nav"] / nav_df["nav"].iloc[0] * 100.0
    tr = pd.DataFrame(trades)
    sig = pd.DataFrame(signals)
    return engine.Result(nav=nav_df, trades=tr, signals=sig,
                         metrics=engine.compute_metrics(nav_df, tr))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = engine.load_data()
    _, close_df = engine.build_panels(data)
    start = engine.common_start(close_df)
    end = close_df.index[-1]
    base_bench_nav = engine.run_benchmark(data, start, end)
    bm = engine.simple_metrics(base_bench_nav)
    print(f"样本 {start.date()} -> {end.date()} | 基准 CAGR {pct(bm['cagr'])}", flush=True)

    variants = []

    def add(name, desc, **kw):
        r = run_ts(data, start=start, end=end, **kw)
        m = r.metrics
        gap = (bm["cagr"] - m["cagr"]) * 1e4
        variants.append({
            "name": name, "desc": desc, "cagr": m["cagr"], "gap_bp": gap,
            "turnover": m["annual_turnover"], "sharpe": m["sharpe"],
            "mdd": m["max_drawdown"], "cost_bp": m["cost_annual_bp"],
            "final_nav": m["final_nav"],
            "avg_long": float(r.signals["n_long"].mean()) if len(r.signals) else float("nan"),
            "avg_short": float(r.signals["n_short"].mean()) if len(r.signals) else float("nan"),
            "min_avail": int(r.signals["n_avail"].min()) if len(r.signals) else 0,
        })
        print(f"  {name}: CAGR {pct(m['cagr'])}, 缺口 {gap:+.0f}bp, "
              f"换手 {pct(m['annual_turnover'],0)}, MDD {pct(m['max_drawdown'])}", flush=True)

    add("base", "10 只、只做多（第一版口径）")
    add("avail3y", "10 只、只做多、每个调仓日要求上市满 3 年",
        min_hist_years=MIN_HIST_YEARS)
    add("ls_dn", "10 只、多空对冲 50/50（空最差 3 只负动量）",
        long_short=True, short_frac=0.50, short_n=3)
    add("ls_70", "10 只、多 70% / 空 30%（空最差 3 只）",
        long_short=True, short_frac=0.30, short_n=3)
    add("ls_80", "10 只、多 80% / 空 20%（空最差 2 只）",
        long_short=True, short_frac=0.20, short_n=2)

    base = variants[0]
    for v in variants:
        v["delta_bp"] = (v["cagr"] - base["cagr"]) * 1e4
        v["gap_change_bp"] = v["gap_bp"] - base["gap_bp"]

    df = pd.DataFrame(variants)
    df.to_csv(OUT / "variants.csv", index=False, encoding="utf-8-sig")

    meta = engine.meta_block(Path(__file__))
    result = {
        "meta": meta, "sample": [str(start.date()), str(end.date())],
        "benchmark": bm, "base_gap_bp": base["gap_bp"],
        "variants": variants,
        "add_asset_class": {
            "status": "BLOCKED",
            "reason": "yfinance 下载额外 ETF 失败: curl (7) Could not connect to query2.finance.yahoo.com:443; "
                      "query1 返回 403; stooq 返回 200 但为 JS 挑战页, 无 CSV。",
            "tickers_attempted": engine.EXTRA,
        },
    }
    (OUT / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float),
                                      encoding="utf-8")
    write_report(meta, df, base, bm, start, end)
    print(f"完成 {OUT/'report.md'}", flush=True)


def write_report(meta, df, base, bm, start, end):
    L = []
    L.append("# 方向 3：换策略池（资产类别 / 空头 / 幸存者偏差）\n")
    L.append(f"- `builtAt` {meta['builtAt']} ｜ `head` {meta['head']} ｜ "
             f"`universe.py` sha256 前 16 位 **{meta['script_sha256']}** ｜ "
             f"引擎 `engine.py` sha256 前 16 位 **{hashlib.sha256((ROOT/'engine.py').read_bytes()).hexdigest()[:16]}** ｜ "
             f"Python {meta['python']} / pandas {meta['pandas']} / numpy {meta['numpy']}")
    L.append("- 命令：`D:\\Agent\\quant\\.venv\\Scripts\\python.exe D:\\Agent\\quant-phase2\\universe.py`")
    L.append(f"- 样本：{start.date()} → {end.date()}；对象：第一版时序动量"
             "（12 月动量、正动量等权、目标波动 10%、杠杆 ≤1.0），缺口基线 206bp / 换手 355%。")
    L.append("- 硬约束未动：t 收盘信号、t+1 开盘、月度、单边 7bp。标签 `[A]` 实测 / `[B]` 推算。\n")

    L.append("## 结论\n")
    L.append(f"- **加资产类别：BLOCKED。** 额外 7 只 ETF（HYG/LQD/EMB/TIP/SLV/DBA/IWM）下载失败，"
             "真实错误记录在第五节；没有用合成数据或代理资产凑数，因此这一条本轮**无法给出 bp**。")
    ls = df[df["name"].str.startswith("ls")]
    best_ls = ls.sort_values("cagr", ascending=False).iloc[0]
    L.append(f"- **空头腿：能跑，但全部显著恶化。** 最好的空头档位是 {best_ls['name']}："
             f"年化 {pct(best_ls['cagr'])}，比只做多基线 {pct(base['cagr'])} 少 "
             f"{-best_ls['delta_bp']:.0f}bp，缺口扩大到 {best_ls['gap_bp']:+.0f}bp；"
             "空头在 2008–2026 的美股长牛里持续付出反向收益，且未计借券费，真实结果更差。")
    av = df[df.name == 'avail3y'].iloc[0]
    L.append(f"- **当时可得的池：「上市满 3 年」规则实测缩小 {av['delta_bp']:+.0f}bp。** "
             f"年化 {pct(av['cagr'])}（基线 {pct(base['cagr'])}），缺口 {av['gap_bp']:+.0f}bp。"
             "这个改善来自早期剔除了 UUP（2007-03 上市）和 DBC（2006-02 上市）等新基金，"
             "不是“修复了清盘基金缺失”的幸存者偏差；**已清盘的 ETF 仍不在样本内，真正的偏差没消除。**\n")

    L.append("## 一、逐项结果\n")
    rows = []
    for _, r in df.iterrows():
        rows.append([r["name"], r["desc"], pct(r["cagr"]), f"{r['gap_bp']:+.0f}",
                     f"{r['delta_bp']:+.0f}", f"{r['gap_change_bp']:+.0f}",
                     pct(r["turnover"], 0), pct(r["mdd"]), f"{r['cost_bp']:.1f}",
                     f"{r['avg_long']:.1f}/{r['avg_short']:.1f}"])
    L.append(table(["档位", "说明", "年化", "缺口 bp", "年化变化 bp", "缺口变化 bp",
                    "换手", "回撤", "成本 bp/年", "平均多/空只数"], rows))
    L.append("")

    L.append("## 二、每条改动的 bp 账\n")
    L.append(f"- 基线缺口 = 基准 {pct(bm['cagr'])} − 策略 {pct(base['cagr'])} = "
             f"**{base['gap_bp']:+.0f}bp**。")
    L.append(f"- 空头（ls_70）：年化 {pct(df[df.name=='ls_70'].cagr.iloc[0])}，"
             f"缺口 {df[df.name=='ls_70'].gap_bp.iloc[0]:+.0f}bp，"
             f"相对基线 {df[df.name=='ls_70'].delta_bp.iloc[0]:+.0f}bp。")
    L.append(f"- 空头（50/50）：年化 {pct(df[df.name=='ls_dn'].cagr.iloc[0])}，"
             f"缺口 {df[df.name=='ls_dn'].gap_bp.iloc[0]:+.0f}bp，"
             f"相对基线 {df[df.name=='ls_dn'].delta_bp.iloc[0]:+.0f}bp。")
    L.append(f"- 当时可得的池（上市满 3 年）：年化 {pct(df[df.name=='avail3y'].cagr.iloc[0])}，"
             f"缺口 {df[df.name=='avail3y'].gap_bp.iloc[0]:+.0f}bp，"
             f"相对基线 {df[df.name=='avail3y'].delta_bp.iloc[0]:+.0f}bp；"
             "规则在 2009-02 前剔除 DBC、2010-03 前剔除 UUP，所以不是 0bp；"
             "这是“不用太新的基金”的效果，不是清盘偏差修复。")
    L.append("- 加资产类别：BLOCKED，无可分配 bp。")
    L.append("")

    L.append("## 三、剩余缺口归因\n")
    L.append(f"- 成本：基线年化成本 {base['cost_bp']:.1f}bp；这是 206bp 缺口的其中一部分，"
             "换手 355% 下即使降到 300% 也只省约 3.8bp [B]。")
    L.append("- 资产类别缺失：无法量化（数据 BLOCKED）。理论上缺 HYG/LQD/EMB/TIP 等信用与抗通胀资产，"
             "会让 2008、2020、2022 的避险腿更宽；但没有数据就不给数字。")
    L.append("- 空头：本轮实测为负贡献；空头腿在长牛样本里是结构性逆风，借券费未计，真实结果更差 [B]。")
    L.append(f"- 幸存者偏差：“上市满 3 年”修正实测 {av['delta_bp']:+.0f}bp；"
             "清盘基金缺失的偏差仍在，方向未知、量级未知 [B]，不能用本数据下结论。")
    rest = base["gap_bp"] - av["delta_bp"]
    L.append(f"- 归因结论：206bp 里，上市可得性规则消掉 {av['delta_bp']:.0f}bp，剩 {rest:.0f}bp；"
             f"这 {rest:.0f}bp 中可量化的只有成本约 {base['cost_bp']:.0f}bp，"
             f"其余约 {rest - base['cost_bp']:.0f}bp 归因于"
             "“10 只 ETF 的池太窄 + 只做多 + 样本期美股/美元资产顺风”。"
             "空头为负；资产类别拿不到数字。**这些都不是执行细节能补的。**\n")

    L.append("## 四、判断\n")
    L.append("- **值得继续**：把资产池扩到信用/抗通胀/因子 ETF（需要先解决数据源），这是唯一可能改变"
             "“10 只太窄”的结构性改动；空头腿在长牛样本里不值得继续。")
    L.append("- **该放弃**：用空头腿补缺口、用“上市满 3 年”规则声称消除幸存者偏差。前者实测为负；"
             f"后者只缩小 {av['delta_bp']:+.0f}bp，且修不掉清盘基金缺失，不能当成“偏差已消除”。")
    L.append("- 建议下一步：换一个可离线获得的历史 ETF 池（含已清盘代码），或改用 CRSP/彭博"
             "的点时成分数据；在此之前，方向 3 只能作为“机制验证”，不能作为结论。\n")

    L.append("## 五、数据阻塞记录\n")
    L.append("- 尝试 1：`yf.Ticker('HYG').history(start='2006-01-01')` → "
             "`curl (7) Could not connect to query2.finance.yahoo.com:443`。")
    L.append("- 尝试 2：`query1.finance.yahoo.com/v8/finance/chart/HYG` → HTTP 403。")
    L.append("- 尝试 3：`stooq.com/q/d/l/?s=hyg.us&i=d` → HTTP 200 但返回 HTML 挑战页，不是 CSV。")
    L.append("- 结论：不下载、不插值、不用代理资产；`add_asset_class` 保持 BLOCKED。\n")

    L.append("## 六、产物\n")
    L.append("- `report.md`、`variants.csv`、`metrics.json`")
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
