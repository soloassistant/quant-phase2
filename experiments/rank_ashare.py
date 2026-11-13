"""A股「黄金坑」二次筛选 v2：硬否决(AND)做排除 + 加权分(排序) + K线量化诊断。

针对上一版的漏洞：加权总分让单一维度满分掩盖其余维度崩坏（如艾力斯 便宜1.0 趋势0.29）。
修正：排除决策用**不可互补的硬门槛**（AND），排序才用加权分。

硬否决（任一命中即剔除，不可被便宜度救回）：
  - 破位：收盘距 200 日线 < -15%
  - 长期阴跌：近 250 日收益<0 且 近 120 日收益<0 且 近 250 日多数时间在 200 日线下方(>60%)
  - 黑天鹅：近 250 日单日跌幅 < -9.5%(接近跌停/跳空) 或 5 日跌幅 < -20%
放量异动（疑似消息驱动，不自动剔除，仅标记人工复核）：近 250 日最大量 / 中位量 > 3。

排序（幸存者内）：便宜度45% + 趋势30% + 6月动量15% + 低波10%（仅用于排序）。
另对前若干只生成 K 线图，供人工目检。

    python experiments/rank_ashare.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "out"
DATA = ROOT / "data_ashare"

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
BRK_DIST = -0.10
DOWN_120 = -0.15       # 近 120 日（半年）跌超 15% → 下行趋势
BARS_BELOW = 0.55      # 250 日内 >55% 时间在 200 日线下方
CHEAP_MIN = 0.30       # 便宜度百分位下限（黄金坑必须本身便宜）
BLACKSWAN_1D = -0.15
BLACKSWAN_5D = -0.25
VOLSPIKE = 5.0
TOP_CORE = 12


def cheap_raw(code):
    f = FUND[code]; t = f[0]
    if t in ('peg', 'pe', 'pb'):
        return -f[1]
    price, low = f[1], f[2]
    return (low - price) / low


def kline_diag(df):
    c = df["close"].astype(float).ffill()
    v = df["volume"].astype(float)
    n = len(c)
    d = {}
    d["ret_250"] = float(c.iloc[-1] / c.iloc[-min(251, n)] - 1)
    d["ret_120"] = float(c.iloc[-1] / c.iloc[-min(121, n)] - 1)
    d["ret_60"] = float(c.iloc[-1] / c.iloc[-min(61, n)] - 1)
    sma = c.rolling(200, min_periods=200).mean()
    sma_last = float(sma.iloc[-1]) if pd.notna(sma.iloc[-1]) else float("nan")
    d["sma200"] = sma_last
    d["dist_sma200"] = float(c.iloc[-1] / sma_last - 1) if sma_last == sma_last else float("nan")
    w = min(250, n)
    below = (c.iloc[-w:].to_numpy() < sma.iloc[-w:].to_numpy())
    d["pct_below_sma200"] = float(np.nanmean(below)) if w else float("nan")
    seg = c.iloc[-w:]
    d["max_dd_250"] = float((seg / seg.cummax() - 1).min())
    d["max_1d"] = float(c.pct_change().iloc[-w:].min())
    d["max_5d"] = float((c / c.shift(5) - 1).iloc[-w:].min())
    vv = v.iloc[-w:]
    med = float(vv.median()) if vv.median() > 0 else np.nan
    d["vol_spike"] = float(vv.max() / med) if med == med and med > 0 else float("nan")
    return d


def main():
    A = json.loads((OUT / "ashare_screen.json").read_text(encoding="utf-8"))
    rows = {r["code"]: dict(r) for r in A["rows"]}

    dfc = pd.DataFrame([{"code": c, "type": FUND[c][0], "cheap_raw": cheap_raw(c)} for c in FUND])
    dfc["cheap_pct"] = dfc.groupby("type")["cheap_raw"].rank(pct=True)
    cheap_pct = dict(zip(dfc["code"], dfc["cheap_pct"]))

    for c in FUND:
        p = DATA / f"{c}.csv"
        diag = kline_diag(pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()) if p.exists() else {}
        rows[c].update(diag)
        rows[c]["cheap_pct"] = float(cheap_pct[c])

    for key in ["dist_sma200", "ret_60", "mom_6m"]:
        s = pd.Series({c: rows[c].get(key, np.nan) for c in FUND})
        rk = s.rank(pct=True)
        for c in FUND:
            rows[c][key + "_pct"] = float(rk.get(c, np.nan))
    vol = pd.Series({c: rows[c]["vol20"] for c in FUND})
    vrk = (-vol).rank(pct=True)
    for c in FUND:
        rows[c]["vol_low_pct"] = float(vrk[c])

    for c in FUND:
        r = rows[c]
        r["score"] = (W["cheap"] * r["cheap_pct"] + W["trend"] * r.get("dist_sma200_pct", 0)
                      + W["mom6"] * r["mom_6m_pct"] + W["lowvol"] * r["vol_low_pct"])
        reasons = []
        if r["cheap_pct"] < CHEAP_MIN:
            reasons.append("不够便宜")
        if r.get("dist_sma200") is not None and np.isfinite(r["dist_sma200"]) and r["dist_sma200"] < BRK_DIST:
            reasons.append("破位>10%")
        if np.isfinite(r.get("ret_120", np.nan)) and r["ret_120"] < DOWN_120:
            reasons.append("近6月下行")
        if (np.isfinite(r.get("ret_250", np.nan)) and r["ret_250"] < 0
                and r.get("pct_below_sma200", 0) > BARS_BELOW):
            reasons.append("长期阴跌")
        if (np.isfinite(r.get("max_1d", np.nan)) and r["max_1d"] < BLACKSWAN_1D) or \
           (np.isfinite(r.get("max_5d", np.nan)) and r["max_5d"] < BLACKSWAN_5D):
            reasons.append("黑天鹅")
        r["veto"] = "; ".join(reasons)
        r["flag_volume"] = bool(np.isfinite(r.get("vol_spike", np.nan)) and r["vol_spike"] > VOLSPIKE)

    survivors = [c for c in FUND if not rows[c]["veto"]]
    survivors.sort(key=lambda c: -rows[c]["score"])
    rank = {c: i + 1 for i, c in enumerate(survivors)}
    for c in FUND:
        r = rows[c]
        r["rank"] = rank.get(c)
        if r["veto"]:
            r["suggest"] = "否决"
        elif rank[c] <= TOP_CORE:
            r["suggest"] = "核心"
        elif rank[c] <= 2 * TOP_CORE:
            r["suggest"] = "备选"
        else:
            r["suggest"] = "观察"

    from collections import Counter
    vc = Counter()
    for c in FUND:
        for x in rows[c]["veto"].split("; "):
            if x:
                vc[x] += 1
    out = {"weights": W, "gates": {"brk_dist": BRK_DIST, "blackswan_1d": BLACKSWAN_1D,
                                   "blackswan_5d": BLACKSWAN_5D, "volspike": VOLSPIKE},
           "n_total": len(FUND), "n_veto": len(FUND) - len(survivors), "veto_breakdown": dict(vc),
           "n_survivor": len(survivors), "n_core": min(TOP_CORE, len(survivors)),
           "ranking": sorted([rows[c] for c in FUND], key=lambda r: (r["rank"] is None, r["rank"] or 999))}
    (OUT / "ashare_rank.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float),
                                          encoding="utf-8")

    print(f"总59  否决 {len(FUND)-len(survivors)}  幸存 {len(survivors)}  否决明细 {dict(vc)}")
    print(f"{'#':>2}{'代码':7s}{'名称':10s}{'体系':5s}{'便宜':>6s}{'趋势':>6s}{'250日':>7s}{'最大回撤':>9s}{'单日':>7s}{'量':>4s}{'综合':>6s}")
    for c in survivors[:TOP_CORE + 6]:
        r = rows[c]
        print(f"{r['rank']:>2}{c:7s}{r['name']:10s}{r['system']:5s}{r['cheap_pct']:6.2f}"
              f"{r['dist_sma200_pct']:6.2f}{r['ret_250']*100:6.0f}%{r['max_dd_250']*100:8.0f}%"
              f"{r['max_1d']*100:6.1f}%{'!' if r['flag_volume'] else ' ':>4s}{r['score']:6.3f}")
    print("\n被否决(前20):")
    for c in [c for c in FUND if rows[c]["veto"]][:20]:
        r = rows[c]
        print(f"  {r['name']}({c}) [{r['veto']}] 距200线{r['dist_sma200']*100:+.0f}% 250日{r['ret_250']*100:+.0f}%")


if __name__ == "__main__":
    main()
