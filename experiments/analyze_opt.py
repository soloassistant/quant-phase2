"""读取 experiments/out/*.json，输出分年度与分窗口对照，用于判断改善是否集中在少数年份。"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "out"
NAMES = ["baseline", "band10", "band20", "band30", "buffer1", "invvol"]


def load(name):
    return json.loads((OUT / f"{name}.json").read_text(encoding="utf-8"))


def main():
    data = {n: load(n) for n in NAMES}
    base = data["baseline"]

    print("## 分年度收益（OOS 拼接净值）\n")
    years = sorted(base["yearly"])
    print("| 年份 | 基线 | " + " | ".join(n for n in NAMES if n != "baseline") + " |")
    print("|---" * (len(NAMES) + 1) + "|")
    for y in years:
        row = [f"{base['yearly'].get(y, 0)*100:+.2f}%"]
        for n in NAMES:
            if n == "baseline":
                continue
            row.append(f"{data[n]['yearly'].get(y, 0)*100:+.2f}%")
        print(f"| {y} | " + " | ".join(row) + " |")

    print("\n## 各候选逐年胜出次数与年度超额中位数\n")
    print("| 变体 | 胜出年数 / 总年数 | 年度超额中位数 | 最大单年超额 | 最差单年 |")
    print("|---|---|---|---|---|")
    for n in NAMES:
        if n == "baseline":
            continue
        diffs = [data[n]["yearly"].get(y, 0) - base["yearly"].get(y, 0) for y in years]
        wins = sum(1 for d in diffs if d > 0)
        diffs_sorted = sorted(diffs)
        med = diffs_sorted[len(diffs_sorted) // 2]
        print(f"| {n} | {wins}/{len(years)} | {med*100:+.2f}pp | "
              f"{max(diffs)*100:+.2f}pp | {min(diffs)*100:+.2f}pp |")

    print("\n## 分窗口（验证年）对照：基线 vs band20\n")
    bw = base["windows"]
    cw = data["band20"]["windows"]
    print("| 验证年 | 基线 K/月/过滤/权重 | band20 选择 | 基线 CAGR | band20 CAGR | "
          "基线 Sharpe | band20 Sharpe | 基线换手 | band20 换手 | 跳过次数 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for a, b in zip(bw, cw):
        pa = f'{a["k"]}/{a["mom_months"]}/{a["filter"]}/{a["weight"]}'
        pb = f'{b["k"]}/{b["mom_months"]}/{b["filter"]}/{b["weight"]}'
        print(f'| {a["val_start"][:7]}→{a["val_end"][:7]} | {pa} | {pb} | '
              f'{a["val_cagr"]*100:.2f}% | {b["val_cagr"]*100:.2f}% | '
              f'{a["val_sharpe"]:.3f} | {b["val_sharpe"]:.3f} | '
              f'{a["val_turnover"]*100:.1f}% | {b["val_turnover"]*100:.1f}% | {b["val_skips"]} |')

    print("\n## 选择路径差异\n")
    for a, b in zip(bw, cw):
        pa = (a["k"], a["mom_months"], a["filter"], a["weight"])
        pb = (b["k"], b["mom_months"], b["filter"], b["weight"])
        if pa != pb:
            print(f'- {a["val_start"][:7]}: 基线 {pa} → band20 {pb}')


if __name__ == "__main__":
    main()
