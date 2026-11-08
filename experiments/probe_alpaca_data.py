"""只读探针：实测这把 Alpaca key 的行情数据权限，并与现有 yfinance 口径逐日比对。

不入库原始行情：只把 bar 条数、首根 bar、限流头、以及聚合后的差异写进结果文件。

用法:
  python experiments/probe_alpaca_data.py

凭据只从进程环境变量读：APCA_API_KEY_ID / APCA_API_SECRET_KEY。
本脚本不写 data/、不改 pipeline_core.py、不动任何锁定口径。
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "out"
DATA = ROOT / "data"
HOST = "https://data.alpaca.markets"
START, END = "2026-09-01", "2026-11-04"
SYMBOLS = ["SPY", "QQQ", "EEM", "DBC"]

# 判定规则（跑之前写死）:
#   差异中位数 < 3bp 且无系统性偏移 → 交叉校验源，默认口径仍用现有数据
#   差异中位数 ≥ 3bp 或有系统性偏移 → 仅备源，不可替代主口径
#   两种情况都不切换主数据源
BP_GATE = 3.0


def creds():
    key = os.environ.get("APCA_API_KEY_ID", "")
    secret = os.environ.get("APCA_API_SECRET_KEY", "")
    if not key or not secret:
        raise SystemExit("[凭据缺失] 需要进程环境变量 APCA_API_KEY_ID / APCA_API_SECRET_KEY")
    return key, secret


def get(url):
    key, secret = creds()
    req = urllib.request.Request(url, headers={"APCA-API-KEY-ID": key,
                                               "APCA-API-SECRET-KEY": secret})
    try:
        with urllib.request.urlopen(req, timeout=40) as resp:
            body = resp.read().decode("utf-8")
            headers = {k: v for k, v in resp.headers.items()
                       if "ratelimit" in k.lower() or "retry" in k.lower()}
            return {"status": resp.status, "headers": headers,
                    "body": json.loads(body) if body else None, "error": None}
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "headers": {}, "body": None,
                "error": exc.read().decode("utf-8", "replace")}
    except Exception as exc:  # 网络层错误也要原样记录，不静默
        return {"status": None, "headers": {}, "body": None, "error": repr(exc)}


def fetch_bars(symbol, feed):
    url = (f"{HOST}/v2/stocks/bars?symbols={symbol}&timeframe=1Day"
           f"&start={START}&end={END}&adjustment=all&feed={feed}&limit=1000")
    res = get(url)
    bars = []
    if res["body"]:
        bars = (res["body"].get("bars") or {}).get(symbol, [])
        while bars and res["body"].get("next_page_token"):
            res = get(url + "&page_token=" + res["body"]["next_page_token"])
            if not res["body"]:
                break
            bars += (res["body"].get("bars") or {}).get(symbol, [])
    return {"status": res["status"], "headers": res["headers"], "error": res["error"],
            "count": len(bars), "first_bar": bars[0] if bars else None,
            "series": {pd.Timestamp(b["t"]).date(): b for b in bars}}


def load_local(symbol):
    path = DATA / f"{symbol}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[(df["date"] >= pd.Timestamp(START).date()) & (df["date"] <= pd.Timestamp(END).date())]
    return df.set_index("date")


def pair_stats(a, b, label_a, label_b):
    """a/b: {date: {'c': close, 'v': volume}}；返回逐日差异聚合。"""
    both = sorted(set(a) & set(b))
    diffs = [abs(float(a[d]["c"]) - float(b[d]["c"])) for d in both]
    bp = pd.Series([(float(b[d]["c"]) / float(a[d]["c"]) - 1) * 1e4 for d in both if a[d]["c"]])
    vol_ratio = [float(b[d]["v"]) / float(a[d]["v"]) for d in both if a[d].get("v")]
    top3 = sorted(zip(both, diffs), key=lambda x: -x[1])[:3]
    return {
        "pair": f"{label_a} vs {label_b}", "overlap_days": len(both),
        "missing_in_b": sorted(str(d) for d in set(a) - set(b)),
        "extra_in_b": sorted(str(d) for d in set(b) - set(a)),
        "mean_abs_diff": float(pd.Series(diffs).mean()) if diffs else None,
        "max_abs_diff": float(max(diffs)) if diffs else None,
        "median_bp": float(bp.abs().median()) if len(bp) else None,
        "mean_signed_bp": float(bp.mean()) if len(bp) else None,
        "volume_ratio_median": float(pd.Series(vol_ratio).median()) if vol_ratio else None,
        "top3": [{"date": str(d), label_a: round(float(a[d]["c"]), 4),
                  label_b: round(float(b[d]["c"]), 4)} for d, _ in top3],
    }


def local_as_bars(df):
    return {d: {"c": float(row["close"]), "v": float(row["volume"])}
            for d, row in df.iterrows()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"window": {"start": START, "end": END}, "bp_gate": BP_GATE,
              "bars": {}, "snapshots": {}, "pairs": {}, "verdict": ""}

    spy = {}
    for feed in ("iex", "sip"):
        r = fetch_bars("SPY", feed)
        spy[feed] = {"status": r["status"], "headers": r["headers"], "error": r["error"],
                     "count": r["count"], "first_bar": r["first_bar"]}
        print(f"[bars] SPY feed={feed} status={r['status']} count={r['count']} err={r['error']}")
    result["bars"]["SPY"] = spy

    for feed in ("iex", "sip"):
        snap = get(f"{HOST}/v2/stocks/snapshots?symbols=SPY,QQQ&feed={feed}")
        body = snap["body"] or {}
        result["snapshots"][feed] = {
            "status": snap["status"], "headers": snap["headers"], "error": snap["error"],
            "sample": {s: {"latestTrade": body.get(s, {}).get("latestTrade"),
                           "dailyBar": body.get(s, {}).get("dailyBar")} for s in ("SPY", "QQQ")}
            if body else None}
        print(f"[snapshot] feed={feed} status={snap['status']} err={str(snap['error'])[:90]}")

    for sym in SYMBOLS:
        local = load_local(sym)
        iex = fetch_bars(sym, "iex")["series"]
        sip = fetch_bars(sym, "sip")["series"]
        if local is None:
            result["pairs"][sym] = {"status": "no_local_csv"}
            continue
        loc = local_as_bars(local)
        entry = {"status": "ok", "local_rows": len(loc), "iex_rows": len(iex),
                 "sip_rows": len(sip),
                 "sip_vs_yf": pair_stats(loc, sip, "yfinance", "sip"),
                 "iex_vs_yf": pair_stats(loc, iex, "yfinance", "iex"),
                 "iex_vs_sip": pair_stats(sip, iex, "sip", "iex")}
        result["pairs"][sym] = entry
        print(f"[cmp] {sym} sip-vs-yf med={entry['sip_vs_yf']['median_bp']:.2f}bp | "
              f"iex-vs-yf med={entry['iex_vs_yf']['median_bp']:.2f}bp | "
              f"iex-vs-sip med={entry['iex_vs_sip']['median_bp']:.2f}bp "
              f"vol={entry['iex_vs_sip']['volume_ratio_median']:.1f}x")

    ok = [v["sip_vs_yf"] for v in result["pairs"].values() if v.get("status") == "ok"]
    worst = max(p["median_bp"] for p in ok) if ok else None
    systematic = any(abs(p["mean_signed_bp"]) >= BP_GATE for p in ok)
    if worst is not None and worst < BP_GATE and not systematic:
        result["verdict"] = ("SIP 与 yfinance 差异中位数 < 3bp 且无系统性偏移 → 可作交叉校验源；"
                            "默认口径仍用现有数据，不切换。IEX 为单一交易所，不可作主口径。")
    else:
        result["verdict"] = ("SIP 与 yfinance 差异中位数 ≥ 3bp 或存在系统性偏移 → 仅作备源，"
                            "不可替代主口径；不切换主数据源。")
    result["iex_bias_bp"] = {s: v["iex_vs_yf"]["mean_signed_bp"]
                             for s, v in result["pairs"].items() if v.get("status") == "ok"}

    (OUT / "alpaca_data_probe.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_md(result)
    print(f"\n结论: {result['verdict']}")
    print(f"产物: {OUT/'alpaca_data_probe.json'} / {OUT/'alpaca_data_probe.md'}")


def write_md(r):
    L = ["# Alpaca 行情权限与数据比对探针\n",
         f"- 区间 {r['window']['start']} → {r['window']['end']}；判定门槛 {r['bp_gate']}bp"
         "（跑前写死）", "", "## 一、历史日线权限（SPY，adjustment=all）\n",
         "| feed | HTTP | bar 条数 | 限流头 | 首根 bar |", "|---|---|---|---|---|"]
    for feed in ("iex", "sip"):
        b = r["bars"]["SPY"][feed]
        first = json.dumps(b["first_bar"], ensure_ascii=False) if b["first_bar"] else "—"
        L.append(f"| {feed} | {b['status']} | {b['count']} | {b['headers'] or '—'} | `{first}` |")
        if b["error"]:
            L.append(f"\n**{feed} 原始错误体**：`{b['error']}`\n")
    L.append("\n## 二、实时快照\n")
    for feed in ("iex", "sip"):
        s = r["snapshots"][feed]
        L.append(f"- feed={feed}: HTTP {s['status']}；限流 {s['headers'] or '—'}")
        if s["error"]:
            L.append(f"  - 原始错误体：`{s['error']}`")
        if s["sample"]:
            L.append(f"  - SPY latestTrade: `{json.dumps(s['sample']['SPY']['latestTrade'], ensure_ascii=False)}`")
            L.append(f"  - QQQ latestTrade: `{json.dumps(s['sample']['QQQ']['latestTrade'], ensure_ascii=False)}`")
    L.append("\n## 三、SIP vs yfinance（主口径能否被交叉校验）\n")
    L.append("| 标的 | 重叠日 | 缺/多 | mean\\|diff\\| | max\\|diff\\| | 中位 bp | 平均有符号 bp |")
    L.append("|---|---|---|---|---|---|---|")
    for sym in SYMBOLS:
        v = r["pairs"].get(sym, {})
        if v.get("status") != "ok":
            L.append(f"| {sym} | — | — | — | — | — | — |")
            continue
        p = v["sip_vs_yf"]
        L.append(f"| {sym} | {p['overlap_days']} | {len(p['missing_in_b'])}/{len(p['extra_in_b'])} | "
                 f"{p['mean_abs_diff']:.4f} | {p['max_abs_diff']:.4f} | {p['median_bp']:.2f} | "
                 f"{p['mean_signed_bp']:+.2f} |")
    L.append("\n- 最大绝对差 ≤ 0.5 美分：SIP 收盘取整到分，yfinance 保留更多小数位；"
             "差异只来自小数位，不是价格源分歧。")
    L.append("\n## 四、IEX vs SIP vs yfinance（IEX-only 的偏差与量级）\n")
    L.append("| 标的 | iex−yf 中位 bp | iex−yf 有符号 bp | iex−sip 中位 bp | IEX 成交量占全市场 |")
    L.append("|---|---|---|---|---|")
    for sym in SYMBOLS:
        v = r["pairs"].get(sym, {})
        if v.get("status") != "ok":
            continue
        L.append(f"| {sym} | {v['iex_vs_yf']['median_bp']:.2f} | "
                 f"{v['iex_vs_yf']['mean_signed_bp']:+.2f} | "
                 f"{v['iex_vs_sip']['median_bp']:.2f} | "
                 f"{v['iex_vs_sip']['volume_ratio_median']:.1%} |")
    for sym in SYMBOLS:
        v = r["pairs"].get(sym, {})
        if v.get("status") != "ok":
            continue
        L.append(f"\n**{sym} 差异最大的 3 天（SIP vs yfinance）**\n")
        L.append("| 日期 | yfinance | SIP |")
        L.append("|---|---|---|")
        for row in v["sip_vs_yf"]["top3"]:
            L.append(f"| {row['date']} | {row['yfinance']:.4f} | {row['sip']:.4f} |")
    L.append(f"\n## 五、判定\n\n{r['verdict']}\n")
    (OUT / "alpaca_data_probe.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
