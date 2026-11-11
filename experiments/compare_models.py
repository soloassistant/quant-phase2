"""把本模型与「别人在 GitHub 上的量化模型」拉到同一天平上对比。

公平口径（所有策略共用同一个模拟器）：
  · 同一宇宙：SPY QQQ IEF TLT GLD DBC VNQ EFA EEM UUP（10 只 ETF）
  · 同一数据：本地 data/*.csv（auto_adjust，2008-02→2026-11）
  · 同一执行：t 收盘信号、t+1 开盘成交、单边 7bp、现金 0 收益
  · 月度调仓

对比对象（均为有公开仓库/论文背书的经典开源策略，在本仓库数据上忠实复现）：
  · ours        本仓库主模型：横截面 12-1 动量 top-3、等权、目标波动 10%（vol-target）
  · ours_band20 经济改良版：换手带 20%（单只权重偏离 <20% 就不调仓，降成本）
  · faber5      Faber 战术资产配置（2007）：5 类资产(SPY/EFA/IEF/DBC/VNQ)各 1/5，
                月末收盘 > 10 月均线才持有，否则持现金。GitHub: mmoraschini/faber-taa 等
  · taa10       同上趋势过滤，但套到全部 10 只（更贴近本仓库宇宙）
  · dm          Dual Momentum（Antonacci 2014）：SPY/EFA/IEF 取 12 月动量最强者，
                且绝对动量 > 现金才持有，否则现金。GitHub: andrewjesaitis/dual-momentum 等
  · ew          等权买入持有 10 只（宇宙基准）
  · spy         SPY 买入持有（市场基准）

说明/局限：
  · 本仓库无 T-bill 序列，绝对动量的"现金"按 0 收益处理（比真实 T-bill 保守）。
  · ours 有 vol-target（杠杆≤1、目标波动 10%），外部策略满仓或持现金——这是策略本身的差异，
    用 Sharpe（已按波动归一）与 MDD 横向比更公平，CAGR 仅作参考。

    python experiments/compare_models.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import xsec  # noqa: E402

OUT = ROOT / "experiments" / "out"
COST_RATE = xsec.COST_RATE
COLS = list(xsec.UNIVERSE)
FWD = 21  # 前向约 1 个月（命中率用）


def metrics(nav: pd.Series) -> dict:
    m = xsec.simple_metrics(nav)
    mdd = m["max_drawdown"]
    m["calmar"] = m["cagr"] / abs(mdd) if mdd else float("nan")
    return m


def main():
    data = xsec.load_data()
    open_df, close_df = xsec.build_panels(data)
    cal = close_df.index
    cols = list(close_df.columns)
    close_ff = close_df.ffill()
    ret_f = close_df.pct_change(fill_method=None).ffill()
    open_np = open_df.to_numpy(dtype=float)
    close_np = close_ff.to_numpy(dtype=float)
    flags = xsec.month_end_flags(cal)
    sma = close_df.rolling(210).mean()          # 10 月均线
    fwd_ret = close_df.shift(-FWD) / close_df - 1.0
    base_start = xsec.common_start(close_df)
    end = cal[-1]

    def snap(ts):
        return max(0, int(cal.get_indexer([pd.Timestamp(ts)], method="pad")[0]))

    # ---------------- 统一模拟器（numpy，执行口径与生产一致）----------------
    def simulate(get_target, start_ts, end_ts):
        si, ei = snap(start_ts), snap(end_ts)
        cash, sh = 100.0, np.zeros(len(cols))
        pending, rows, tlog = None, [], []
        n_open_fallback = 0
        for i in range(si, ei + 1):
            if pending is not None:
                px = open_np[i].copy()
                if np.isnan(px).any():  # 理论上样本内不应出现；出现则退回收盘价并计数（这些天极少数）
                    px = np.where(np.isnan(px), close_np[i], px)
                    n_open_fallback += 1
                tradable = ~np.isnan(px) & (px > 0)
                w = np.where(tradable, pending, 0.0)
                cur_val = sh * px
                pv = cash + float(np.nansum(cur_val))
                cost, tv = 0.0, np.zeros(len(cols))
                for _ in range(4):
                    tv = w * (pv - cost)
                    cost = float(np.nansum(np.abs(tv - cur_val))) * COST_RATE
                trade_val = tv - cur_val
                cash = cash - float(np.nansum(trade_val)) - cost
                sh = np.where(tradable, tv / np.where(tradable, px, 1.0), 0.0)
                pending = None
            nav = cash + float(sh @ close_np[i])
            rows.append((cal[i], nav))
            if flags[i] and i + 1 <= ei:
                cur_w = (sh * close_np[i] / nav) if nav > 0 else np.zeros(len(cols))
                tw = get_target(i, cur_w)
                tlog.append((i, tw))
                pending = tw
        nav = pd.Series([v for _, v in rows], index=[d for d, _ in rows])
        return nav, tlog, n_open_fallback

    # ---------------- 各策略的「月末目标权重」函数 ----------------
    def t_ours(i, cur_w):
        w, _ = xsec.make_target(close_df, ret_f, i, 3, 252, False, "equal", 0.10)
        return w.reindex(cols).to_numpy(dtype=float)

    def t_ours_band20(i, cur_w):
        w, _ = xsec.make_target(close_df, ret_f, i, 3, 252, False, "equal", 0.10)
        wv = w.reindex(cols).to_numpy(dtype=float)
        if float(np.max(np.abs(wv - cur_w))) < 0.20:  # 换手带：偏离小就不动
            return cur_w.copy()
        return wv

    def t_ours_trend(i, cur_w):
        # 经济改良：动量选中的标的，只保留仍在 10 月均线之上的；跌破均线的转现金（不补仓）
        w, _ = xsec.make_target(close_df, ret_f, i, 3, 252, False, "equal", 0.10)
        wv = w.reindex(cols).to_numpy(dtype=float).copy()
        for j, c in enumerate(cols):
            s = sma.iloc[i][c]
            if not (pd.notna(s) and close_df.iloc[i][c] > s):
                wv[j] = 0.0
        return wv

    def t_ours_mkt(i, cur_w):
        # 经济改良：市场级择时——SPY 跌破 10 月均线则全现金，否则正常动量选股
        s = sma.iloc[i]["SPY"]
        if pd.notna(s) and close_df.iloc[i]["SPY"] < s:
            return np.zeros(len(cols))
        w, _ = xsec.make_target(close_df, ret_f, i, 3, 252, False, "equal", 0.10)
        return w.reindex(cols).to_numpy(dtype=float)

    def make_faber(universe):
        we = 1.0 / len(universe)
        def f(i, cur_w):
            out = np.zeros(len(cols))
            for c in universe:
                j = cols.index(c)
                s = sma.iloc[i][c]
                if pd.notna(s) and close_df.iloc[i][c] > s:
                    out[j] = we
            return out
        return f

    def t_dm(i, cur_w):
        out = np.zeros(len(cols))
        if i - 252 < 0:
            return out
        cand = ["SPY", "EFA", "IEF"]
        mom = {c: close_df.iloc[i][c] / close_df.iloc[i - 252][c] - 1.0 for c in cand}
        best = max(cand, key=lambda c: mom[c])
        if mom[best] > 0:  # 绝对动量胜现金（0 收益）
            out[cols.index(best)] = 1.0
        return out

    def t_ew(i, cur_w):
        return np.full(len(cols), 1.0 / len(cols))

    def t_spy(i, cur_w):
        out = np.zeros(len(cols)); out[cols.index("SPY")] = 1.0
        return out

    STRATS = {
        "ours":        (t_ours, "本仓库主模型 K=3/12-1/等权/vol10%"),
        "ours_band20": (t_ours_band20, "经济改良：换手带 20%"),
        "ours_trend":  (t_ours_trend, "经济改良：动量选股 + 10月均线趋势过滤"),
        "ours_mkt":    (t_ours_mkt, "经济改良：动量选股 + SPY 10月均线市场避险"),
        "faber5":      (make_faber(["SPY", "EFA", "IEF", "DBC", "VNQ"]), "Faber TAA 5资产/10月均线"),
        "taa10":       (make_faber(list(cols)), "Faber 式趋势过滤 全10只"),
        "dm":          (t_dm, "Dual Momentum 12月"),
        "ew":          (t_ew, "等权 10 只买入持有"),
        "spy":         (t_spy, "SPY 买入持有"),
    }

    # ---------------- 命中率（选中组合下月跑赢等权宇宙） ----------------
    def hit_rate(tlog):
        nh, ph = [], []
        for i, tw in tlog:
            if i + FWD >= len(cal):
                continue
            bench = float(fwd_ret.iloc[i].mean())
            held = [cols[j] for j in range(len(cols)) if tw[j] > 1e-9]
            if not held:
                ph.append(bench < 0.0)  # 持现金：基准跌才算赢
                continue
            for c in held:
                fr = fwd_ret.iloc[i][c]
                if not pd.isna(fr):
                    nh.append(bool(fr > bench))
            prs = [fwd_ret.iloc[i][c] for c in held if not pd.isna(fwd_ret.iloc[i][c])]
            if prs:
                ph.append(bool(float(np.mean(prs)) > bench))
        return (float(np.mean(nh)) if nh else float("nan"),
                float(np.mean(ph)) if ph else float("nan"))

    WINDOWS = {"FULL 2008-02→今": (base_start, end),
               "OOS  2016-02→今": (pd.Timestamp("2016-02-29"), end)}

    result = {"windows": {}, "note": ""}
    for wname, (ws, we) in WINDOWS.items():
        print(f"\n===== {wname} =====", flush=True)
        tbl = []
        for key, (fn, desc) in STRATS.items():
            nav, tlog, fb = simulate(fn, ws, we)
            m = metrics(nav)
            nr, pr = hit_rate(tlog)
            tbl.append({"strategy": key, "desc": desc,
                        "cagr": m["cagr"], "ann_vol": m["ann_vol"], "sharpe": m["sharpe"],
                        "mdd": m["max_drawdown"], "calmar": m["calmar"],
                        "hit_name": nr, "hit_port": pr, "n_open_fallback": fb})
        result["windows"][wname] = tbl
        # 打印
        print(f"{'策略':12s} {'CAGR':>7s} {'波动':>7s} {'Sharpe':>7s} {'MDD':>8s} {'Calmar':>7s} {'命中(名)':>9s} {'命中(组)':>9s}")
        for r in tbl:
            print(f"{r['strategy']:12s} {r['cagr']*100:6.2f}% {r['ann_vol']*100:6.2f}% "
                  f"{r['sharpe']:7.3f} {r['mdd']*100:7.2f}% {r['calmar']:7.3f} "
                  f"{r['hit_name']*100:8.1f}% {r['hit_port']*100:8.1f}%")

    OUT.mkdir(parents=True, exist_ok=True)
    # 交叉校验：模拟器 "ours" 应 ≈ 生产 xsec.run（同口径）
    ro = xsec.run(data, k=3, mom_days=252, use_filter=False, weight="equal",
                  target_vol=0.10, start=pd.Timestamp("2016-02-29"), end=end)
    mo = metrics(ro.nav["nav"])
    print(f"\n[交叉校验] xsec.run OOS: CAGR {mo['cagr']*100:.2f}% Sharpe {mo['sharpe']:.3f} "
          f"MDD {mo['max_drawdown']*100:.2f}%  (应与上表 ours 接近)", flush=True)
    (OUT / "compare_models.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n写出 {OUT/'compare_models.json'}", flush=True)


if __name__ == "__main__":
    main()
