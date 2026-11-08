"""Alpaca 纸面账户执行层：生成订单计划（默认只算不下单）。

凭据只从进程环境变量读：APCA_API_KEY_ID / APCA_API_SECRET_KEY / APCA_API_BASE_URL。
本文件不含任何凭据，也不把凭据写进任何输出。

用法:
  python alpaca_exec.py snapshot --out state/snapshots/sig.json
  python alpaca_exec.py plan --signal signals/20261108_014711 --snapshot state/snapshots/sig.json
  python alpaca_exec.py plan ... --execute        # 显式动作，仅 paper 端点、仅开盘时段

纪律: 名义金额单（notional，支持碎股）；偏离 < band（默认 1% of equity）不动；
1x 杠杆，现金不足时缩量、不融资；同一信号只执行一次。
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
PAPER_HOST = "paper-api.alpaca.markets"
DATA_HOST = "https://data.alpaca.markets"

PLAN_COLS = ["symbol", "current_weight", "target_weight", "deviation", "action",
             "notional", "est_qty", "price"]


def creds():
    key = os.environ.get("APCA_API_KEY_ID", "")
    secret = os.environ.get("APCA_API_SECRET_KEY", "")
    if not key or not secret:
        raise SystemExit("[凭据缺失] 需要进程环境变量 APCA_API_KEY_ID / APCA_API_SECRET_KEY")
    return key, secret


def api(method, path, base=None, body=None, timeout=30):
    key, secret = creds()
    base = base or os.environ.get("APCA_API_BASE_URL", "https://" + PAPER_HOST)
    url = base.rstrip("/") + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8")
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"[HTTP {exc.code}] {method} {path}\n{detail}") from None


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cmd_snapshot(out):
    account = api("GET", "/v2/account")
    positions = api("GET", "/v2/positions")
    clock = api("GET", "/v2/clock")
    symbols = sorted({p["symbol"] for p in positions})
    prices = {}
    for p in positions:
        prices[p["symbol"]] = {"price": float(p["current_price"]), "source": "position.current_price"}
    snap = {"fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "base_url": os.environ.get("APCA_API_BASE_URL", "https://" + PAPER_HOST),
            "clock": clock, "account": account, "positions": positions, "prices": prices,
            "extra_symbols": symbols}
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"快照: {path}  sha256={sha256_file(path)}")
    print(f"账户 equity={account['equity']} cash={account['cash']} "
          f"buying_power={account['buying_power']} is_open={clock['is_open']}")


def load_holdings(signal_dir):
    import csv
    path = Path(signal_dir) / "holdings.csv"
    if not path.exists():
        raise SystemExit(f"[数据缺失] 缺少 {path}")
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append({"symbol": row["代码"].strip(),
                         "name": row["名称"].strip(),
                         "target_weight": float(row["实际权重"].strip().rstrip("%")) / 100.0})
    return rows, path


def add_prices(snap, rows):
    """未持仓的目标标的从 data API 取最新成交价（IEX feed），写入快照副本，不改原文件。"""
    missing = [r["symbol"] for r in rows if r["symbol"] not in snap["prices"]]
    if not missing:
        return snap
    key, secret = creds()
    url = f"{DATA_HOST}/v2/stocks/snapshots?symbols={','.join(missing)}&feed=iex"
    req = urllib.request.Request(url, headers={"APCA-API-KEY-ID": key,
                                               "APCA-API-SECRET-KEY": secret})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    out = json.loads(json.dumps(snap))
    for sym, item in data.items():
        px = (item.get("latestTrade") or {}).get("p") or (item.get("dailyBar") or {}).get("c")
        if not px:
            raise SystemExit(f"[价格缺失] {sym} 无可用成交价，拒绝生成计划")
        out["prices"][sym] = {"price": float(px), "source": "data.snapshot.latestTrade.iex"}
    return out


def build_plan(sig_dir, snap_path, band):
    rows, holdings_path = load_holdings(sig_dir)
    snap = json.loads(Path(snap_path).read_text(encoding="utf-8"))
    signal_sha = json.loads((Path(sig_dir) / "metrics.json").read_text(encoding="utf-8"))["signal_sha256"]
    snap = add_prices(snap, rows)

    equity = float(snap["account"]["equity"])
    cash = float(snap["account"]["cash"])
    held = {p["symbol"]: p for p in snap["positions"]}
    lines, buys, sells = [], 0.0, 0.0

    targets = {r["symbol"]: r["target_weight"] for r in rows}
    symbols = sorted(set(held) | set(targets))
    for sym in symbols:
        tw = float(targets.get(sym, 0.0))
        cur_val = float(held[sym]["market_value"]) if sym in held else 0.0
        cw = cur_val / equity
        price = float(snap["prices"][sym]["price"]) if sym in snap["prices"] else \
            (float(held[sym]["current_price"]) if sym in held else 0.0)
        dev = tw - cw
        if abs(dev) < band:
            action, notional = "HOLD", 0.0
        else:
            notional = round(tw * equity - cur_val, 2)
            action = "BUY" if notional > 0 else "SELL"
            if action == "BUY":
                buys += notional
            else:
                sells += -notional
        qty = round(notional / price, 4) if price > 0 else 0.0
        lines.append({"symbol": sym, "current_weight": round(cw, 8),
                      "target_weight": round(tw, 8), "deviation": round(dev, 8),
                      "action": action, "notional": notional, "est_qty": qty,
                      "price": price, "name": next((r["name"] for r in rows if r["symbol"] == sym), "")})

    note = ""
    if buys > cash:
        scale = max(cash, 0.0) / buys
        remaining, buys = max(cash, 0.0), 0.0
        for row in lines:
            if row["action"] == "BUY":
                amt = round(min(remaining, row["notional"] * scale), 2)
                row["notional"] = amt
                remaining = round(remaining - amt, 2)
                row["est_qty"] = round(amt / row["price"], 4) if row["price"] > 0 else 0.0
                buys += amt
        note = f"现金不足，按 1x 缩量至 {scale:.4f}（不融资）"

    cash_after = round(cash - buys + sells, 2)
    sig_path = Path(sig_dir).resolve()
    sig_ref = sig_path.relative_to(ROOT).as_posix() if ROOT in sig_path.parents else sig_path.as_posix()
    plan = {"signal_dir": sig_ref, "signal_sha256": signal_sha,
            "snapshot_sha256": sha256_file(snap_path),
            "band": band, "equity": round(equity, 2), "cash": round(cash, 2),
            "leverage": 1.0,
            "orders": lines, "buy_notional": round(buys, 2), "sell_notional": round(sells, 2),
            "cash_after": cash_after,
            "cash_after_pct": round(cash_after / equity, 8),
            "target_cash_pct": round(1.0 - sum(float(r["target_weight"]) for r in rows), 8)}
    if note:
        plan["cash_guard"] = note
    plan["plan_sha256"] = sha256_text(canonical(plan))
    return plan, holdings_path


def write_plan(plan, sig_dir):
    out = ROOT / "state" / "plans" / Path(sig_dir).name
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
    table = ["| 标的 | 名称 | 当前权重 | 目标权重 | 偏离 | 建议动作 | 名义金额 | 估算股数 | 参考价 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in plan["orders"]:
        table.append("| {symbol} | {name} | {cw:.4%} | {tw:.4%} | {dev:+.4%} | {act} | "
                     "{notional:,.2f} | {qty:,.4f} | {price:,.2f} |".format(
                         symbol=row["symbol"], name=row["name"], cw=row["current_weight"],
                         tw=row["target_weight"], dev=row["deviation"], act=row["action"],
                         notional=row["notional"], qty=row["est_qty"], price=row["price"]))
    head = (f"\n信号 {plan['signal_sha256'][:16]} | 快照 {plan['snapshot_sha256'][:16]} | "
            f"阈值 {plan['band']:.2%} | equity {plan['equity']:,.2f} | 现金 {plan['cash']:,.2f}")
    tail = (f"\n买单合计 {plan['buy_notional']:,.2f} / 卖单合计 {plan['sell_notional']:,.2f} / "
            f"执行后现金 {plan['cash_after']:,.2f}（{plan['cash_after_pct']:.2%}，"
            f"目标现金 {plan['target_cash_pct']:.2%}）")
    if plan.get("cash_guard"):
        tail += f"\n现金约束 {plan['cash_guard']}"
    tail += f"\n计划 sha256 {plan['plan_sha256']}"
    text = head + "\n" + "\n".join(table) + "\n" + tail
    (out / "plan.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return out


def execute(plan, base_url, sig_dir):
    host = urlparse(base_url).hostname
    if host != PAPER_HOST:
        raise SystemExit(f"[拒绝执行] endpoint 主机是 {host}，只允许 {PAPER_HOST}（纸面账户）")
    state = ROOT / "state" / "executed" / f"{plan['signal_sha256']}.json"
    if state.exists():
        prev = json.loads(state.read_text(encoding="utf-8"))
        print(f"该信号已执行，跳过（{prev.get('executed_at')}，{len(prev.get('orders', []))} 笔）")
        return 0
    clock = api("GET", "/v2/clock", base=base_url)
    if not clock.get("is_open"):
        pending = ROOT / "state" / "pending" / f"{plan['signal_sha256']}.json"
        pending.parent.mkdir(parents=True, exist_ok=True)
        pending.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"未开盘（下次开盘 {clock.get('next_open')}），计划已保存未提交: {pending}")
        return 3

    done, failed = [], []
    for row in plan["orders"]:
        if row["action"] == "HOLD" or row["notional"] == 0:
            continue
        body = {"symbol": row["symbol"], "side": row["action"].lower(),
                "type": "market", "time_in_force": "day"}
        if row["action"] == "BUY":
            body["notional"] = f"{row['notional']:.2f}"
        else:
            body["qty"] = f"{row['est_qty']:.4f}"
        try:
            order = api("POST", "/v2/orders", base=base_url, body=body)
            done.append({"symbol": row["symbol"], "side": body["side"],
                         "notional": row["notional"], "id": order["id"],
                         "status": order["status"]})
            print(f"已提交 {row['symbol']} {body['side']} {row['notional']:.2f} id={order['id']}")
        except SystemExit as exc:
            failed.append({"symbol": row["symbol"], "side": body["side"],
                           "notional": row["notional"], "error": str(exc)})
            print(f"[下单失败] {row['symbol']}\n{exc}")
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"signal_sha256": plan["signal_sha256"],
                                 "plan_sha256": plan["plan_sha256"],
                                 "executed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 "orders": done, "failed": failed},
                                ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"成交 {len(done)} 笔 / 失败 {len(failed)} 笔；状态已写 {state}")
    return 0 if not failed else 4


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s0 = sub.add_parser("snapshot")
    s0.add_argument("--out", required=True)
    s1 = sub.add_parser("plan")
    s1.add_argument("--signal", required=True)
    s1.add_argument("--snapshot", required=True)
    s1.add_argument("--band", type=float, default=0.01)
    s1.add_argument("--execute", action="store_true")
    s1.add_argument("--base-url", default=None)
    args = ap.parse_args()

    if args.cmd == "snapshot":
        cmd_snapshot(args.out)
        return 0
    plan, _ = build_plan(args.signal, args.snapshot, args.band)
    write_plan(plan, args.signal)
    if not args.execute:
        print("\n[dry-run] 未提交任何订单；确需执行请加 --execute")
        return 0
    base = args.base_url or os.environ.get("APCA_API_BASE_URL", "https://" + PAPER_HOST)
    return execute(plan, base, args.signal)


if __name__ == "__main__":
    sys.exit(main())
