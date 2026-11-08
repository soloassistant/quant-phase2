"""边界探针：SIP 历史可用截止点、复权口径、假日缺口。

只读：不写 data/、不改 pipeline_core.py、不覆盖任何既有产物；
原始行情不入库，只存每天的状态/条数与复权因子的聚合值。

用法:
  python experiments/probe_sip_cutoff.py

凭据只从进程环境变量读：APCA_API_KEY_ID / APCA_API_SECRET_KEY。

判定口径（跑之前写死）:
  C = sip 单日返回 200 且带数据的最新交易日
  N = C 之后到最近一个已收盘交易日（2026-11-06）之间的交易日数
  N 小（<=1）→ 可做“次日复核最新收盘”；N 大（>=5）→ 只作历史校验
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "out"
DATA = ROOT / "data"
HOST = "https://data.alpaca.markets"

TODAY = "2026-11-08"
LAST_CLOSE = "2026-11-06"
# 今天往前 10 个交易日（到 11-06），外加 11-07（周六）作为非交易日对照
PROBE_DATES = ["2026-11-06", "2026-11-05", "2026-11-04", "2026-11-03", "2026-11-02",
               "2026-10-30", "2026-10-29", "2026-10-28", "2026-10-27", "2026-10-26",
               "2026-11-07"]
ADJ_START, ADJ_END = "2026-01-01", "2026-11-04"
# 已发生的休市日：2026-09-07 劳动节（11-26 感恩节在今天之后，未来区间会被 403 挡）
HOL_START, HOL_END = "2026-09-03", "2026-09-08"
HOLIDAY = "2026-09-07"
SYMBOL = "SPY"


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
                "error": exc.read().decode("utf-8", "replace").strip()}
    except Exception as exc:
        return {"status": None, "headers": {}, "body": None, "error": repr(exc)}


def bars(symbol, feed, start, end, adjustment="all"):
    url = (f"{HOST}/v2/stocks/bars?symbols={symbol}&timeframe=1Day"
           f"&start={start}&end={end}&adjustment={adjustment}&feed={feed}&limit=1000")
    out = []
    while url:
        for attempt in range(3):
            res = get(url)
            if res["status"] is not None:
                break
            time.sleep(1 + attempt)
        if res["status"] != 200:
            return {"status": res["status"], "count": 0, "bars": [],
                    "error": res["error"], "headers": res["headers"]}
        body = res["body"] or {}
        out += (body.get("bars") or {}).get(symbol, [])
        token = body.get("next_page_token")
        url = (f"{HOST}/v2/stocks/bars?symbols={symbol}&timeframe=1Day"
               f"&start={start}&end={end}&adjustment={adjustment}&feed={feed}"
               f"&limit=1000&page_token={token}") if token else None
    return {"status": 200, "count": len(out), "bars": out, "error": None,
            "headers": res["headers"]}


def probe_day(day):
    row = {"date": day}
    for feed in ("sip", "iex"):
        r = bars(SYMBOL, feed, f"{day}T00:00:00Z", f"{day}T23:59:59Z")
        row[feed] = {"status": r["status"], "count": r["count"],
                     "first_bar": r["bars"][0] if r["bars"] else None,
                     "error": r["error"]}
    return row


def adjustment_test():
    allr = bars(SYMBOL, "sip", ADJ_START, ADJ_END, "all")
    rawr = bars(SYMBOL, "sip", ADJ_START, ADJ_END, "raw")
    if allr["status"] != 200 or rawr["status"] != 200:
        return {"error": {"all": allr["error"], "raw": rawr["error"],
                          "status": [allr["status"], rawr["status"]]}}
    ad = {pd.Timestamp(b["t"]).date(): b["c"] for b in allr["bars"]}
    rd = {pd.Timestamp(b["t"]).date(): b["c"] for b in rawr["bars"]}
    days = sorted(set(ad) & set(rd))
    ratio = {d: rd[d] / ad[d] for d in days}
    jumps = []
    for prev, cur in zip(days, days[1:]):
        f = ratio[cur] / ratio[prev]
        if abs(f - 1.0) > 0.0005:
            jumps.append({"date": str(cur), "prev": str(prev), "factor": f,
                          "raw": rd[cur], "all": ad[cur],
                          "prev_raw": rd[prev], "prev_all": ad[prev]})
    yf = pd.read_csv(DATA / f"{SYMBOL}.csv")
    yf["d"] = pd.to_datetime(yf["date"]).dt.date
    yf_close = dict(zip(yf["d"], yf["close"]))
    compare = []
    for j in jumps:
        cur = pd.Timestamp(j["date"]).date()
        prev = pd.Timestamp(j["prev"]).date()
        yf_ret = yf_close.get(cur, float("nan")) / yf_close.get(prev, float("nan")) if cur in yf_close else None
        all_ret = ad[cur] / ad[prev]
        compare.append({"event_date": str(cur), "prev_date": str(prev),
                        "alpaca_all_ret": all_ret, "yfinance_ret": yf_ret,
                        "alpaca_all_close": ad[cur], "yfinance_close": yf_close.get(cur),
                        "level_diff": (yf_close.get(cur) - ad[cur]) if cur in yf_close else None,
                        "ret_diff_bp": ((yf_ret - all_ret) * 1e4)
                        if yf_ret is not None else None})
    return {"days": len(days), "first_day": str(days[0]) if days else None,
            "last_day": str(days[-1]) if days else None, "ex_div_events": jumps,
            "latest_event_compare": compare,
            "note": "ratio=raw/all；factor 跳变处即除息日，factor≈prev_close/(prev_close-div)"}


def holiday_test():
    span = bars(SYMBOL, "sip", HOL_START, HOL_END)
    hol = bars(SYMBOL, "sip", f"{HOLIDAY}T00:00:00Z", f"{HOLIDAY}T23:59:59Z")
    dates = [str(pd.Timestamp(b["t"]).date()) for b in span["bars"]]
    fut = bars(SYMBOL, "sip", "2026-11-09T00:00:00Z", "2026-11-09T23:59:59Z")
    return {"span_status": span["status"], "span_dates": dates,
            "span_count": span["count"], "holiday_status": hol["status"],
            "holiday_count": hol["count"],
            "holiday_omitted": HOLIDAY not in dates, "error": span["error"],
            "future_date": "2026-11-09", "future_status": fut["status"],
            "future_error": fut["error"]}


def summarize(rows):
    ok = [r["date"] for r in rows if r["sip"]["status"] == 200 and r["sip"]["count"] > 0]
    cutoff = max(ok) if ok else None
    after = []
    if cutoff:
        dr = pd.bdate_range(start=pd.Timestamp(cutoff) + pd.Timedelta(days=1),
                            end=LAST_CLOSE)
        after = [str(d.date()) for d in dr]
    return {"cutoff": cutoff, "trading_days_after_cutoff": after,
            "N": len(after), "today": TODAY, "last_close": LAST_CLOSE}


def write_md(r):
    lines = ["# SIP 历史截止点 / 复权 / 假日 边界探针", "",
             f"- 今天 {r['summary']['today']}，最近已收盘交易日 {r['summary']['last_close']}",
             f"- **结论：SIP 历史可用截止到 {r['summary']['cutoff']}，N = "
             f"{r['summary']['N']}**（距最近收盘的交易日数）", "",
             "| 日期 | sip 状态 | sip bar 数 | iex 状态 | iex bar 数 |",
             "|---|---|---|---|---|"]
    for row in r["days"]:
        lines.append(f"| {row['date']} | {row['sip']['status']} | {row['sip']['count']} | "
                     f"{row['iex']['status']} | {row['iex']['count']} |")
    errs = [row for row in r["days"] if row["sip"]["error"]]
    if errs:
        lines += ["", "SIP 错误体（原样）:"]
        for row in errs:
            lines.append(f"- {row['date']}: {row['sip']['error']}")
    adj = r["adjustment"]
    lines += ["", "## adjustment=all vs raw（SPY，2026 年内）"]
    if adj.get("error"):
        lines.append(f"- 请求失败: {adj['error']}")
    else:
        lines.append(f"- 重叠交易日 {adj['days']}，检出除息事件 {len(adj['ex_div_events'])} 个")
        for ev in adj["ex_div_events"][-3:]:
            lines.append(f"- {ev['date']}：raw {ev['raw']:.4f} / all {ev['all']:.4f}，"
                         f"因子 {ev['factor']:.6f}")
        for c in adj["latest_event_compare"]:
            lines.append(f"- 事件日 {c['event_date']}：alpaca all 日收益 {c['alpaca_all_ret']:.6f}，"
                         f"yfinance {c['yfinance_ret']:.6f}，差 {c['ret_diff_bp']:.4f}bp，"
                         f"收盘差 {c['level_diff']:.4f}")
    hol = r["holiday_test"]
    lines += ["", f"## 假日缺口（{HOL_START} → {HOL_END}，含 {HOLIDAY} 劳动节）",
              f"- 返回日期: {', '.join(hol['span_dates'])}（{hol['span_count']} 根）",
              f"- 单日 {HOLIDAY}: HTTP {hol['holiday_status']}，{hol['holiday_count']} 根",
              f"- 休市日被跳过（未给空 bar）: {hol['holiday_omitted']}",
              f"- 未来日期 {hol['future_date']}: HTTP {hol['future_status']}"
              + (f"；{hol['future_error']}" if hol["future_error"] else "")]
    path = OUT / "alpaca_sip_cutoff.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    result = {"probed_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
              "symbol": SYMBOL, "days": [probe_day(d) for d in PROBE_DATES]}
    result["summary"] = summarize(result["days"])
    result["adjustment"] = adjustment_test()
    result["holiday_test"] = holiday_test()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "alpaca_sip_cutoff.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    md = write_md(result)
    print((OUT / "alpaca_sip_cutoff.md").read_text(encoding="utf-8"))
    print(f"JSON: {OUT / 'alpaca_sip_cutoff.json'}\nMD:   {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
