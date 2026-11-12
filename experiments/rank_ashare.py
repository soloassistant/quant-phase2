"""A股「黄金坑」二次筛选：基本面便宜度 + 价格趋势/动量/波动 多因子排名。

问题：用户基本面体系把 59 只都标"极度低估"，阈值偏松；需要模型收窄。
做法（透明、可调）：
  1) 便宜度：按各自口径换算——成长=PEG、品牌=PE分位、金融=PB分位、通用/周期=相对估值下沿折价；
     同类内取百分位（1=最便宜）。
  2) 价格因子（近 3.9 年月线）：趋势=距200日线、6月动量、波动（越低越好），各取全池百分位。
  3) 风险门槛：先剔除 距200日线 < -25%（深跌/飞刀）或 年化波动 > 80%（极端）。
  4) 综合分 = 0.45*便宜 + 0.30*趋势 + 0.15*6月动量 + 0.10*低波；门槛内排名，取前 15 为核心。

⚠ 权重是启发式，非拟合；无历史月度基本面名单，不能做严谨回测——这是决策排序，不是已验证策略。

    python experiments/rank_ashare.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "out"

# 用户"依据"里的基本面：code -> (类型, 值...)
FUND = {
    '000737': ('peg', 0.16), '603799': ('peg', 0.19), '605117': ('peg', 0.20), '300972': ('peg', 0.21),
    '000657': ('peg', 0.25), '601066': ('peg', 0.26), '300037': ('peg', 0.28), '000807': ('peg', 0.30),
    '600026': ('peg', 0.35), '300059': ('peg', 0.36), '300432': ('peg', 0.37), '003022': ('peg', 0.41),
    '002463': ('peg', 0.43), '600415': ('peg', 0.47), '603225': ('peg', 0.48), '688183': ('peg', 0.50),
    '001965': ('peg', 0.55), '300953': ('peg', 0.57), '600522': ('peg', 0.58), '600489': ('peg', 0.61),
    '603338': ('peg', 0.62), '000938': ('peg', 0.70), '300033': ('peg', 0.76), '688297': ('peg', 0.82),
    '603345': ('peg', 0.84), '688099': ('peg', 0.89), '688019': ('peg', 0.94), '300623': ('peg', 0.98),
    '688578': ('pe', 2.0), '300251': ('pe', 10.0), '600519': ('pe', 15.0),
    '600816': ('pb', 0.0), '601136': ('pb', 2.0), '601601': ('pb', 6.0),
    '002821': ('disc', 152.68, 154.77), '601866': ('disc', 2.35, 2.66), '002078': ('disc', 13.28, 17.71),
    '600547': ('disc', 24.91, 30.41), '600085': ('disc', 23.96, 25.18), '002487': ('disc', 32.02, 42.30),
    '601018': ('disc', 3.51, 4.82), '601099': ('disc', 3.37, 4.18), '002044': ('disc', 4.84, 5.93),
    '600008': ('disc', 2.76, 3.73), '002603': ('disc', 15.48, 20.39), '000725': ('disc', 4.41, 4.71),
    '603529': ('disc', 21.34, 22.58), '000937': ('disc', 4.19, 4.25), '600426': ('disc', 18.48, 21.40),
    '600685': ('disc', 25.37, 32.03), '000651': ('disc', 38.89, 61.80), '603786': ('disc', 39.11, 67.74),
    '002294': ('disc', 29.78, 34.03), '000425': ('disc', 6.54, 8.02), '600486': ('disc', 54.38, 64.18),
    '600031': ('disc', 18.27, 23.26), '600862': ('disc', 18.35, 28.22), '600050': ('disc', 4.23, 4.97),
    '002311': ('disc', 47.67, 55.62),
}
W = {"cheap": 0.45, "trend": 0.30, "mom6": 0.15, "lowvol": 0.10}
GATE_DIST = -0.25
GATE_VOL = 0.80
TOP_CORE = 15


def cheap_raw(code):
    f = FUND[code]; t = f[0]
    if t == 'peg':
        return -f[1]
    if t == 'pe':
        return -f[1]
    if t == 'pb':
        return -f[1]
    price, low = f[1], f[2]
    return (low - price) / low  # 相对下沿折价，越大越便宜


def main():
    A = json.loads((OUT / "ashare_screen.json").read_text(encoding="utf-8"))
    rows = {r["code"]: r for r in A["rows"]}

    df = pd.DataFrame([{"code": c, "type": FUND[c][0], "cheap_raw": cheap_raw(c)} for c in FUND])
    df["cheap_pct"] = df.groupby("type")["cheap_raw"].rank(pct=True)
    cheap_pct = dict(zip(df["code"], df["cheap_pct"]))

    for key in ["dist_sma200", "mom_6m"]:
        vals = pd.Series({c: rows[c][key] for c in FUND})
        rk = vals.rank(pct=True)
        for c in FUND:
            rows[c][key + "_pct"] = float(rk[c])
    vol = pd.Series({c: rows[c]["vol20"] for c in FUND})
    vrk = (-vol).rank(pct=True)  # 低波 = 高分
    for c in FUND:
        rows[c]["vol_low_pct"] = float(vrk[c])

    for c in FUND:
        r = rows[c]
        r["cheap_pct"] = float(cheap_pct[c])
        r["score"] = (W["cheap"] * r["cheap_pct"] + W["trend"] * r["dist_sma200_pct"]
                      + W["mom6"] * r["mom_6m_pct"] + W["lowvol"] * r["vol_low_pct"])
        r["gate_ok"] = bool(r["dist_sma200"] is not None and r["dist_sma200"] >= GATE_DIST
                            and r["vol20"] is not None and r["vol20"] <= GATE_VOL)

    gated = sorted([c for c in FUND if rows[c]["gate_ok"]], key=lambda c: -rows[c]["score"])
    rank_map = {c: i + 1 for i, c in enumerate(gated)}
    for c in FUND:
        r = rows[c]
        r["rank"] = rank_map.get(c)  # None = 未过门槛
        if not r["gate_ok"]:
            r["suggest"] = "剔除·深跌/高波"
        elif r["rank"] <= TOP_CORE:
            r["suggest"] = "核心"
        elif r["rank"] <= 2 * TOP_CORE:
            r["suggest"] = "备选"
        else:
            r["suggest"] = "观察"

    out = {"weights": W, "gate": {"dist_sma200_min": GATE_DIST, "vol20_max": GATE_VOL},
           "n_total": len(FUND), "n_gated": len(gated), "n_core": TOP_CORE,
           "ranking": sorted([rows[c] for c in FUND],
                             key=lambda r: (r["rank"] is None, r["rank"] or 999))}
    (OUT / "ashare_rank.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float),
                                          encoding="utf-8")

    print(f"总 59，过风险门槛 {len(gated)}，核心 {TOP_CORE}，备选 {min(TOP_CORE,len(gated)-TOP_CORE)}")
    print(f"{'#':>2}{'代码':7s}{'名称':10s}{'体系':5s}{'便宜':>6s}{'趋势':>6s}{'6月':>7s}{'低波':>6s}{'综合':>6s} 距200线")
    for c in gated[:TOP_CORE]:
        r = rows[c]
        print(f"{r['rank']:>2}{c:7s}{r['name']:10s}{r['system']:5s}{r['cheap_pct']:6.2f}"
              f"{r['dist_sma200_pct']:6.2f}{r['mom_6m']*100:6.1f}%{r['vol_low_pct']:6.2f}"
              f"{r['score']:6.3f} {r['dist_sma200']*100:+.1f}%")
    print("剔除（深跌或高波）:", "、".join(f"{rows[c]['name']}" for c in FUND if not rows[c]['gate_ok']))


if __name__ == "__main__":
    main()
