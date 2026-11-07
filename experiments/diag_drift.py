"""诊断：连续持有基准在 OOS 起点（2016-02-29）的实际权重漂移。

连续基准 = 2008-02-29 起等权买入并持有；到 2016-02-29 时权重已被 8 年涨跌拉开。
同起点基准 = 2016-02-29 等权买入。两者 124bp 的年化差，先看是不是权重漂移造成的。
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pipeline_core as pc  # noqa: E402
import xsec  # noqa: E402

BASE_START = "2008-02-29"
OOS_START = "2016-02-29"


def main():
    data = pc.load_data()
    open_df, close_df = xsec.build_panels(data)
    cal = close_df.loc[(close_df.index >= pd.Timestamp(BASE_START))].index
    exec_date = cal[1]
    px0 = open_df.loc[exec_date].ffill()
    shares = pd.Series({t: 1.0 / px0[t] for t in pc.UNIVERSE})

    oos_day = close_df.index[close_df.index >= pd.Timestamp(OOS_START)][0]
    px = close_df.loc[oos_day].ffill()
    val = shares * px
    w = val / val.sum()

    print(f"买入日 {exec_date.date()}（等权，7bp 一次性成本）")
    print(f"观察日 {oos_day.date()}")
    print("\n| 代码 | 漂移权重 | 等权 | 偏离 pp |")
    print("|---|---|---|---|")
    for t in pc.UNIVERSE:
        print(f"| {t} | {w[t]*100:.2f}% | {100/len(pc.UNIVERSE):.2f}% | "
              f"{(w[t]-1/len(pc.UNIVERSE))*100:+.2f} |")
    hhi = float((w ** 2).sum())
    print(f"\nHHI {hhi:.4f}（等权 {1/len(pc.UNIVERSE):.4f}）；"
          f"最大单只 {w.max()*100:.2f}% ({w.idxmax()})；前 3 合计 {w.nlargest(3).sum()*100:.2f}%")

    out = json.loads((ROOT / "experiments" / "out" / "baseline.json").read_text(encoding="utf-8"))
    exps, scales = [], []
    for win in out["windows"]:
        p = {"k": win["k"], "mom_days": win["mom_months"] * 21,
             "use_filter": win["filter"] == "有",
             "weight": "equal" if win["weight"] == "等权" else "inv_vol"}
        r = xsec.run(data, start=pd.Timestamp(win["val_start"]),
                     end=pd.Timestamp(win["val_end"]), **p)
        exps.append(float(r.nav["exposure"].mean()))
        scales.append(float(r.signals["sum_weight"].mean()))
    print(f"\n策略 OOS 平均仓位（市值口径）{sum(exps)/len(exps)*100:.2f}%；"
          f"平均目标权重合计 {sum(scales)/len(scales)*100:.2f}%；"
          f"→ 现金占比约 {100 - sum(exps)/len(exps)*100:.2f}%，现金收益固定 0")


if __name__ == "__main__":
    main()
