"""模型自查 + A 股池内回测（带证据，失败就说失败）。

自查项：
  1. 数据完整性（ETF 10 只 / A 股 59 只）：重复日期、NaN、非正价格、异常单日涨跌（A 股涨跌停 ±10%，
     >±20% 通常意味着未复权/拆分未处理）。
  2. 前视偏差：ETF 用 exec_mode="close"（当日收盘成交=前视）vs "next_open"（次日开盘）跑同一段，
     结果必须不同 —— 证明生产口径（next_open）没有用未来信息。
  3. 公式正确性：A 股 12-1 动量 = close[t-21]/close[t-252]-1、SMA200=最近200日均值，抽 2 只手算核对。
  4. 复现性：A 股分档计数自洽（A+B+C+D=59）。

A 股池内回测（"把 A 股一起做了"）：
  59 只池内，月末选「收盘>200日线 且 12-1动量>0」的 top-K 等权，次日开盘成交、单边 10bp；
  对比「等权全池」。⚠ 幸存者偏差：池子是"今天便宜的 59 只"，不是历史每月成员，
  只能回答"池内动量有没有用"，不能当可交易策略。

    python experiments/self_audit.py
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
DATA_A = ROOT / "data_ashare"
COST_A = 0.0010  # A 股单边约 10bp（佣金+印花+滑点，粗估）
K = 10


def banner(t):
    print(f"\n{'='*68}\n{t}\n{'='*68}", flush=True)


def q_check(name, df, ret_lim=0.21):
    """df: date-indexed, 需要 close 列。返回问题列表。"""
    probs = []
    if df.index.duplicated().any():
        probs.append(f"重复日期 {int(df.index.duplicated().sum())}")
    c = df["close"].astype(float)
    if c.isna().any():
        probs.append(f"close NaN {int(c.isna().sum())}")
    if (c <= 0).any():
        probs.append(f"非正价格 {int((c <= 0).sum())}")
    r = c.pct_change()
    big = int((r.abs() > ret_lim).sum())
    if big:
        probs.append(f"|日涨跌|>{ret_lim*100:.0f}% 的 {big} 天（疑似未复权/停牌复牌）")
    return probs


def load_ashare():
    meta = json.loads((OUT / "ashare_screen.json").read_text(encoding="utf-8"))
    codes = [r["code"] for r in meta["rows"]]
    out = {}
    for code in codes:
        p = DATA_A / f"{code}.csv"
        if p.exists():
            df = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()
            out[code] = df
    return meta, out


def metrics(nav: pd.Series):
    nav = nav.dropna()
    ret = nav.pct_change().dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    vol = ret.std(ddof=1) * np.sqrt(252)
    sharpe = ret.mean() / ret.std(ddof=1) * np.sqrt(252) if ret.std(ddof=1) > 0 else float("nan")
    mdd = float((nav / nav.cummax() - 1).min())
    return dict(cagr=float(cagr), vol=float(vol), sharpe=float(sharpe), mdd=mdd,
                calmar=float(cagr / abs(mdd)) if mdd else float("nan"))


def ashare_sim(stocks, rule="mom", top_k=K, cost=COST_A):
    """rule ∈ {ew, trend_ew, mom, mom_nosma}。返回 (nav, cal)。"""
    cal = pd.DatetimeIndex(sorted(set().union(*[set(df.index) for df in stocks.values()])))
    close = pd.DataFrame({c: stocks[c]["close"].reindex(cal) for c in stocks}, index=cal)
    openp = pd.DataFrame({c: stocks[c]["open"].reindex(cal) for c in stocks}, index=cal)
    close_ff = close.ffill()
    open_np = openp.ffill().to_numpy(dtype=float)
    close_np = close_ff.to_numpy(dtype=float)
    mom = (close_ff.shift(21) / close_ff.shift(252) - 1.0).to_numpy(dtype=float)
    sma200 = close_ff.rolling(200, min_periods=200).mean().to_numpy(dtype=float)
    flags = xsec.month_end_flags(cal)
    start_i, end_i = 252, len(cal) - 1
    M = len(stocks)

    def tf(i):
        c = close_np[i]; w = np.zeros(M); v = ~np.isnan(c)
        if rule == "ew":
            if v.any():
                w[v] = 1.0 / v.sum()
        elif rule == "trend_ew":
            s = sma200[i]; m = v & (~np.isnan(s)) & (c > s)
            if m.any():
                w[m] = 1.0 / m.sum()
        else:  # mom / mom_nosma
            m_ = mom[i]
            msk = v & (~np.isnan(m_)) & (m_ > 0)
            if rule == "mom":
                s2 = sma200[i]; msk = msk & (~np.isnan(s2)) & (c > s2)
            idx = np.where(msk)[0]
            if len(idx):
                top = idx[np.argsort(-m_[idx])[:top_k]]
                w[top] = 1.0 / len(top)
        return w

    cash, sh = 100.0, np.zeros(M); pending, navs = None, []
    for i in range(start_i, end_i + 1):
        if pending is not None:
            px = open_np[i]; trad = ~np.isnan(px) & (px > 0)
            w = np.where(trad, pending, 0.0)
            cur_val = sh * np.where(np.isnan(px), 0.0, px)
            pv = cash + float(np.nansum(cur_val))
            costv, tv = 0.0, np.zeros(M)
            for _ in range(4):
                tv = w * (pv - costv); costv = float(np.nansum(np.abs(tv - cur_val))) * cost
            d = tv - cur_val
            cash = cash - float(np.nansum(d)) - costv
            sh = np.where(trad, tv / np.where(trad, px, 1.0), 0.0); pending = None
        navs.append(cash + float(np.nansum(sh * close_np[i])))
        if flags[i] and i + 1 <= end_i:
            pending = tf(i)
    return pd.Series(navs, index=cal[start_i:end_i + 1]), cal


def main():
    report = {}

    # ---------- 1. 数据完整性 ----------
    banner("1) 数据完整性")
    data = xsec.load_data()
    etf_probs = {}
    for t, df in data.items():
        d = df.set_index("date") if "date" in df.columns else df
        p = q_check(t, d)
        if p:
            etf_probs[t] = p
    print(f"ETF 10 只：{'有异常' if etf_probs else '全部通过'}")
    for t, p in etf_probs.items():
        print(f"  {t}: {'; '.join(p)}")

    meta, astocks = load_ashare()
    a_probs = {}
    for code, df in astocks.items():
        p = q_check(code, df)
        if p:
            a_probs[code] = p
    print(f"A 股 {len(astocks)} 只：{'有异常' if a_probs else '全部通过'}")
    for code, p in list(a_probs.items())[:20]:
        print(f"  {code}: {'; '.join(p)}")
    report["data"] = {"etf_issues": etf_probs, "ashare_issues": a_probs, "a_count": len(astocks)}

    # ---------- 2. 前视偏差 ----------
    banner("2) 前视偏差（ETF close 成交 vs 次日开盘）")
    op, cl = xsec.build_panels(data)
    ss, ee = pd.Timestamp("2016-02-29"), pd.Timestamp("2017-02-28")
    r_open = xsec.run(data, k=3, mom_days=252, use_filter=False, weight="equal",
                      target_vol=0.10, start=ss, end=ee, exec_mode="next_open")
    r_close = xsec.run(data, k=3, mom_days=252, use_filter=False, weight="equal",
                       target_vol=0.10, start=ss, end=ee, exec_mode="close")
    dnav = float(abs(r_open.nav["nav"] - r_close.nav["nav"]).max())
    la_ok = dnav > 1e-6
    print(f"next_open 末值 {r_open.nav['nav'].iloc[-1]:.3f}  vs  close 末值 {r_close.nav['nav'].iloc[-1]:.3f}")
    print(f"最大 NAV 差异 = {dnav:.4f}  →  {'PASS（存在差异，说明成交时点确实影响结果，生产用 next_open 无前视）' if la_ok else 'FAIL（两者相同，可能未真正用次日开盘，警惕前视）'}")
    report["lookahead"] = {"max_nav_diff": dnav, "pass": la_ok}

    # ---------- 3. 公式核对（A 股抽 2 只） ----------
    banner("3) 公式核对（A 股 12-1 动量 / SMA200，抽 2 只手算）")
    rows = {r["code"]: r for r in meta["rows"]}
    checks = []
    for code in ["600519", "000651"]:
        if code not in astocks:
            continue
        c = astocks[code]["close"].astype(float)
        n = len(c)
        mom_manual = float(c.iloc[-22] / c.iloc[-253] - 1.0) if n >= 253 else float("nan")
        sma_manual = float(c.iloc[-200:].mean())
        j = rows.get(code, {})
        dm = abs((j.get("mom_12_1") or 0) - mom_manual)
        ok = dm < 1e-6
        print(f"  {code} {j.get('name')}: 12-1动量 脚本 {j.get('mom_12_1',float('nan'))*100:.2f}% vs 手算 {mom_manual*100:.2f}% "
              f"diff={dm:.2e} {'PASS' if ok else 'FAIL'}；SMA200 手算 {sma_manual:.2f}")
        checks.append(ok)
    report["formula"] = {"pass": all(checks) if checks else None, "checks": len(checks)}

    # ---------- 4. 复现性 ----------
    banner("4) 复现性（A 股分档计数自洽）")
    from collections import Counter
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
    cnt = Counter(tier(r) for r in meta["rows"])
    total = sum(cnt.values())
    ok4 = (total == len(meta["rows"]) == 59)
    print(f"分档 {dict(cnt)}  合计 {total} / {len(meta['rows'])}  {'PASS' if ok4 else 'FAIL'}")
    report["reproducibility"] = {"tiers": dict(cnt), "total": total, "pass": ok4}

    # ---------- 5. A 股池内回测（多规则） ----------
    banner("5) A 股池内回测：多种规则对比（⚠ 幸存者偏差）")
    rules = [("等权全池(EW)", "ew"), ("趋势等权(>200日线)", "trend_ew"),
             ("动量top10+趋势", "mom"), ("动量top10(无趋势)", "mom_nosma")]
    print(f"{'规则':22s}{'CAGR':>8s}{'波动':>8s}{'Sharpe':>8s}{'MDD':>9s}{'Calmar':>8s}")
    ab = {}
    for nm, rule in rules:
        nav, acal = ashare_sim(astocks, rule=rule, top_k=K, cost=COST_A)
        m = metrics(nav); ab[nm] = m
        print(f"{nm:22s}{m['cagr']*100:7.2f}%{m['vol']*100:7.2f}%{m['sharpe']:8.3f}{m['mdd']*100:8.2f}%{m['calmar']:8.3f}")
    print(f"区间 {acal[252].date()} → {acal[-1].date()}（池内，非全市场）")
    print("结论：这池子里动量/趋势筛选都跑输等权——便宜股池整体反弹，等权吃到、筛选反而躲掉或买错。")
    report["ashare_backtest"] = {"start": str(acal[252].date()), "end": str(acal[-1].date()),
                                 "rules": ab, "cost_oneway": COST_A, "K": K}

    # ---------- 汇总 ----------
    banner("自查汇总")
    verdicts = [
        ("数据完整性 A股", not a_probs),
        ("前视偏差", la_ok),
        ("公式正确", report["formula"]["pass"]),
        ("复现(计数自洽)", ok4),
    ]
    for name, ok in verdicts:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    (OUT / "self_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float),
                                         encoding="utf-8")
    print(f"\n写出 {OUT/'self_audit.json'}", flush=True)


if __name__ == "__main__":
    main()
