"""信号命中率检查：横截面动量选股到底有没有前向预测力。

方法（与策略口径一致：t 收盘信号 / 持有约 1 个月）：
  · 遍历每个月末信号日 i，用 xsec.make_target 取当期选中的 K 只；
  · 前向约 21 个交易日（次月末）的收盘价累计收益，作为该标的下个月收益（close-to-close 近似）；
  · 全宇宙 10 只等权平均收益 = 基准；
  · 标的级命中：选中标的的下月收益 > 基准；
  · 组合级命中（每月）：选中 K 只等权组合下月收益 > 基准；
  · 另报策略自身月度胜率（xsec 全样本回测的 monthly_win_rate）作对照。

不重跑全样本 NAV 模拟来算命中率（只靠 make_target + 收益查表，秒级）；
月度胜率一次性跑一次全样本回测取 monthly_win_rate。

    python check_hit_rate.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import xsec  # noqa: E402
from signal_timing import month_end_indices  # noqa: E402

# 与 live_params.DEFAULT_PARAMS 对齐（默认参数口径）
K = 3
MOM_DAYS = 252
USE_FILTER = False
WEIGHT = "equal"
TARGET_VOL = 0.10
FWD = 21  # 前向交易日数 ≈ 1 个月


def main():
    data = xsec.load_data()
    _, close_df = xsec.build_panels(data)
    base_start = xsec.common_start(close_df)
    cal = close_df.index
    ret_f = close_df.pct_change(fill_method=None).ffill()
    # 前向收益：从 i 到 i+FWD 的累计收益（close-to-close 近似）
    fwd_ret = close_df.shift(-FWD) / close_df - 1.0

    me = month_end_indices(cal)
    name_hits, port_hits = [], []
    year_name = {}
    n_signals = 0
    rows = []
    for i in me:
        if i + FWD >= len(cal):
            continue
        if cal[i] < base_start:
            continue
        w, diag = xsec.make_target(close_df, ret_f, i, K, MOM_DAYS,
                                   USE_FILTER, WEIGHT, TARGET_VOL)
        picked = diag["selected"]
        if not picked:
            continue
        n_signals += 1
        bench = float(fwd_ret.iloc[i].mean())  # 全宇宙等权基准
        for t in picked:
            fr = fwd_ret.iloc[i][t]
            if pd.isna(fr):
                continue
            hit = bool(fr > bench)
            name_hits.append(hit)
            yr = cal[i].year
            year_name.setdefault(yr, [0, 0])
            year_name[yr][0] += 1
            year_name[yr][1] += int(hit)
            rows.append((str(cal[i].date()), t, round(float(fr) * 100, 2),
                         round(bench * 100, 2), hit))
        port_frs = [fwd_ret.iloc[i][t] for t in picked if not pd.isna(fwd_ret.iloc[i][t])]
        port_ret = float(np.mean(port_frs)) if port_frs else float("nan")
        port_hits.append(bool(port_ret > bench))

    name_rate = float(np.mean(name_hits)) if name_hits else float("nan")
    port_rate = float(np.mean(port_hits)) if port_hits else float("nan")

    # 策略月度胜率（一次性全样本回测）
    res = xsec.run(data, k=K, mom_days=MOM_DAYS, use_filter=USE_FILTER, weight=WEIGHT,
                   target_vol=TARGET_VOL, start=base_start, end=cal[-1])
    monthly_win = res.metrics["monthly_win_rate"]

    print(f"信号命中率检查（K={K}, 动量{MOM_DAYS // 21}月, 等权, 前向{FWD}交易日 close-to-close）")
    print(f"有效信号月: {n_signals}")
    print(f"标的级命中率: {name_rate * 100:.2f}%  "
          f"（{sum(name_hits)}/{len(name_hits)} 个(标的,月)对 > 全宇宙等权基准）")
    print(f"组合级命中率: {port_rate * 100:.2f}%  "
          f"（{sum(port_hits)}/{len(port_hits)} 个月选中组合 > 基准）")
    print(f"策略月度胜率(monthly_win_rate): {monthly_win * 100:.2f}%")
    print("\n分年度 标的级命中率:")
    for yr in sorted(year_name):
        tot, h = year_name[yr]
        print(f"  {yr}: {h / tot * 100:.1f}%  ({h}/{tot})")
    print("\n最近 12 个信号月明细 (日期, 标的, 下月收益%, 基准%, 命中):")
    for r in rows[-12:]:
        print(f"  {r[0]} {r[1]:4s} {r[2]:+6.2f}%  bench {r[3]:+6.2f}%  "
              f"{'HIT' if r[4] else 'miss'}")


if __name__ == "__main__":
    main()
