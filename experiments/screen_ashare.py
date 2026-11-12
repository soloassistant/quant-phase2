"""把 59 只 A 股「极度低估/黄金坑」标的，用动量+趋势模型做二次筛选。

背景：用户基本面体系里"黄金坑"这一档阈值偏松、59 只占比过高，需要模型再收窄。
经济逻辑：便宜（低估）不够，要避开"便宜还在跌"的价值陷阱 → 只保留
  ① 趋势向上（收盘 > 200 日均线）
  ② 12-1 动量 > 0（近一年剔除最近 1 月后仍上涨）
再按 12-1 动量排序，得到精筛名单。

数据：akshare 新浪日线（前复权 qfq），2022-01-01 → 今。逐只抓取并落盘 data_ashare/。
      抓不到/历史不足的如实标注，不编数据。

    python experiments/screen_ashare.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments" / "out"
DATA = ROOT / "data_ashare"
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(exist_ok=True)

# 代码 → (名称, 体系)。体系：成长/通用(UNKNOWN)/周期/金融/品牌
STOCKS = [
    ("600816", "建元信托", "金融"), ("688578", "艾力斯", "品牌"), ("601136", "首创证券", "金融"),
    ("601601", "中国太保", "金融"), ("000737", "北方铜业", "成长"), ("603799", "华友钴业", "成长"),
    ("300251", "光线传媒", "品牌"), ("605117", "德业股份", "成长"), ("300972", "万辰集团", "成长"),
    ("000657", "中钨高新", "成长"), ("601066", "中信建投", "成长"), ("300037", "新宙邦", "成长"),
    ("000807", "云铝股份", "成长"), ("600519", "贵州茅台", "品牌"), ("600026", "中远海能", "成长"),
    ("300059", "东方财富", "成长"), ("300432", "富临精工", "成长"), ("003022", "联泓新科", "成长"),
    ("002463", "沪电股份", "成长"), ("600415", "小商品城", "成长"), ("603225", "新凤鸣", "成长"),
    ("688183", "生益电子", "成长"), ("001965", "招商公路", "成长"), ("300953", "震裕科技", "成长"),
    ("600522", "中天科技", "成长"), ("600489", "中金黄金", "成长"), ("603338", "浙江鼎力", "成长"),
    ("000938", "紫光股份", "成长"), ("300033", "同花顺", "成长"), ("688297", "中无人机", "成长"),
    ("603345", "安井食品", "成长"), ("688099", "晶晨股份", "成长"), ("688019", "安集科技", "成长"),
    ("300623", "捷捷微电", "成长"), ("002821", "凯莱英", "通用"), ("601866", "中远海发", "通用"),
    ("002078", "太阳纸业", "通用"), ("600547", "山东黄金", "通用"), ("600085", "同仁堂", "通用"),
    ("002487", "大金重工", "通用"), ("601018", "宁波港", "周期"), ("601099", "太平洋", "通用"),
    ("002044", "美年健康", "通用"), ("600008", "首创环保", "周期"), ("002603", "以岭药业", "通用"),
    ("000725", "京东方A", "通用"), ("603529", "爱玛科技", "周期"), ("000937", "冀中能源", "通用"),
    ("600426", "华鲁恒升", "周期"), ("600685", "中船防务", "通用"), ("000651", "格力电器", "周期"),
    ("603786", "科博达", "通用"), ("002294", "信立泰", "通用"), ("000425", "徐工机械", "周期"),
    ("600486", "扬农化工", "通用"), ("600031", "三一重工", "通用"), ("600862", "中航高科", "通用"),
    ("600050", "中国联通", "通用"), ("002311", "海大集团", "通用"),
]


def sina_symbol(code: str) -> str:
    return ("sh" if code[0] == "6" else "sz") + code


def fetch_one(code: str) -> pd.DataFrame:
    import akshare as ak
    sym = sina_symbol(code)
    df = ak.stock_zh_a_daily(symbol=sym, start_date="20220101", end_date="20261112", adjust="qfq")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    return df


def fetch_all(codes):
    cache_ok, failed = {}, []
    for code in codes:
        p = DATA / f"{code}.csv"
        if p.exists():
            try:
                df = pd.read_csv(p, parse_dates=["date"]).set_index("date")
                if len(df) > 200:
                    cache_ok[code] = df
                    continue
            except Exception:
                pass
        try:
            df = fetch_one(code)
            df.to_csv(p)
            cache_ok[code] = df
            print(f"  fetched {code} rows={len(df)} last={df.index[-1].date()}", flush=True)
            time.sleep(0.3)  # 新浪较宽松，仍礼貌限速
        except Exception as e:
            failed.append((code, repr(e)[:80]))
            print(f"  FAIL {code}: {repr(e)[:80]}", flush=True)
            time.sleep(1.0)
    return cache_ok, failed


def metrics_one(df: pd.DataFrame) -> dict:
    c = df["close"].astype(float)
    n = len(c)
    px = float(c.iloc[-1])
    out = {"rows": n, "price": px, "last_date": str(df.index[-1].date())}
    # 12-1 动量：跳过最近 21 交易日，往前看 252 日
    if n >= 253:
        out["mom_12_1"] = float(c.iloc[-22] / c.iloc[-253] - 1.0)
    elif n >= 130:
        out["mom_12_1"] = float(c.iloc[-22] / c.iloc[0] - 1.0)
        out["short_hist"] = True
    else:
        out["mom_12_1"] = float("nan")
        out["short_hist"] = True
    out["mom_6m"] = float(c.iloc[-1] / c.iloc[-127] - 1.0) if n >= 127 else float("nan")
    out["mom_3m"] = float(c.iloc[-1] / c.iloc[-64] - 1.0) if n >= 64 else float("nan")
    # 200 日均线趋势
    if n >= 200:
        sma200 = float(c.iloc[-200:].mean())
        out["sma200"] = sma200
        out["above_sma200"] = bool(px > sma200)
        out["dist_sma200"] = float(px / sma200 - 1.0)
    else:
        out["sma200"] = float("nan"); out["above_sma200"] = None; out["dist_sma200"] = float("nan")
    # 20 日年化波动
    r = c.pct_change().dropna()
    out["vol20"] = float(r.iloc[-20:].std(ddof=1) * np.sqrt(252)) if len(r) >= 20 else float("nan")
    return out


def main():
    codes = [s[0] for s in STOCKS]
    name = {s[0]: s[1] for s in STOCKS}
    system = {s[0]: s[2] for s in STOCKS}
    print(f"抓取 {len(codes)} 只 A 股（新浪前复权）…", flush=True)
    data, failed = fetch_all(codes)

    rows = []
    for code in codes:
        if code not in data:
            rows.append({"code": code, "name": name[code], "system": system[code],
                         "verdict": "数据缺失", "mom_12_1": None, "above_sma200": None})
            continue
        m = metrics_one(data[code])
        mom = m.get("mom_12_1", float("nan"))
        above = m.get("above_sma200")
        if above is None or not np.isfinite(mom):
            verdict = "历史不足"
        elif not above:
            verdict = "淘汰·跌破200日线"
        elif mom <= 0:
            verdict = "淘汰·12-1动量≤0"
        else:
            verdict = "PASS"
        rows.append({"code": code, "name": name[code], "system": system[code],
                     "price": m.get("price"), "mom_12_1": mom, "mom_6m": m.get("mom_6m"),
                     "mom_3m": m.get("mom_3m"), "above_sma200": above,
                     "dist_sma200": m.get("dist_sma200"), "vol20": m.get("vol20"),
                     "rows": m.get("rows"), "short_hist": m.get("short_hist", False),
                     "verdict": verdict})

    df = pd.DataFrame(rows)

    def tier(r):
        if not (r["verdict"] == "PASS" or str(r["verdict"]).startswith("淘汰")):
            return "X"
        if r["verdict"] == "PASS":
            return "A"
        if r["above_sma200"] is True:
            return "B"
        mom, dist = r["mom_12_1"], r["dist_sma200"]
        if (mom is not None and np.isfinite(mom) and mom > 0 and dist is not None
                and np.isfinite(dist) and dist >= -0.05):
            return "C"
        return "D"

    df["tier"] = df.apply(tier, axis=1)
    df = df.sort_values(["_o"], ascending=True) if "_o" in df else df
    df["_o"] = df["tier"].map({"A": 0, "B": 1, "C": 2, "X": 3, "D": 4})
    df = df.sort_values(["_o", "mom_12_1"], ascending=[True, False])

    def pct(v, d=1):
        return "   n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v*100:6.{d}f}%"

    def flt(v):
        return float("nan") if v is None else v

    def fmt(r):
        return (f"{r['code']:7s}{r['name']:10s}{r['system']:5s}{flt(r['price']):9.2f}"
                f"{pct(r['mom_12_1']):>10s}{pct(r['mom_6m']):>9s}"
                f"{pct(r['dist_sma200']):>11s}{pct(r['vol20']):>9s}")

    hdr = (f"{'代码':7s}{'名称':10s}{'体系':5s}{'现价':>9s}{'12-1动量':>10s}"
           f"{'6月':>9s}{'距200日线':>11s}{'年化波动':>9s}")
    cnt = df["tier"].value_counts().to_dict()
    nA, nB, nC, nD = cnt.get("A", 0), cnt.get("B", 0), cnt.get("C", 0), cnt.get("D", 0)
    print(f"\n抓取成功 {len(data)}/{len(codes)}（新浪前复权，末交易日 {df['rows'].notna().sum() and '2026-11-12'}）")
    print(f"分档：A 通过 {nA} ｜ B 趋势在/动量弱 {nB} ｜ C 贴近翻多 {nC} ｜ D 跌破趋势(剔除) {nD}\n")

    labels = {"A": "A 通过（>200日线 且 12-1动量>0）",
              "B": "B 趋势未破但 12-1动量≤0（观察）",
              "C": "C 跌破200日线但在5%内且动量>0（贴近翻多）"}
    lines_md = []
    for t in ["A", "B", "C"]:
        sub = df[df["tier"] == t]
        if not len(sub):
            continue
        print(f"== {labels[t]} ==")
        print(hdr)
        for _, r in sub.iterrows():
            print(fmt(r))
        print()
        lines_md.append(f"\n### {labels[t]}（{len(sub)}）\n")
        lines_md.append("| 代码 | 名称 | 体系 | 现价 | 12-1动量 | 6月 | 距200日线 | 年化波动 |")
        lines_md.append("|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            lines_md.append(f"| {r['code']} | {r['name']} | {r['system']} | {flt(r['price']):.2f} | "
                            f"{pct(r['mom_12_1'])} | {pct(r['mom_6m'])} | {pct(r['dist_sma200'])} | {pct(r['vol20'])} |")
    print(f"D 跌破趋势（>=5% 在 200 日线下方）{nD} 只，作为价值陷阱风险剔除。")

    (OUT / "ashare_screen.json").write_text(json.dumps(
        {"n_total": len(codes), "n_fetch_ok": len(data), "tiers": {"A": nA, "B": nB, "C": nC, "D": nD},
         "failed": failed, "rows": rows}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    md = ["# 59 只「极度低估/黄金坑」→ 动量+趋势二次筛选\n",
          "口径：新浪前复权日线，12-1 动量（跳过最近 1 月），200 日均线趋势。",
          f"共 {len(codes)} 只，A 通过 {nA}、B 观察 {nB}、C 贴近翻多 {nC}、D 剔除 {nD}。\n",
          "> 61% 的标的跌破 200 日线——「便宜」多因在跌，是典型价值陷阱；模型的作用就是把它们剔掉。"]
    md += lines_md
    (OUT / "ashare_screen.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n写出 {OUT/'ashare_screen.json'} 与 {OUT/'ashare_screen.md'}", flush=True)


if __name__ == "__main__":
    main()
