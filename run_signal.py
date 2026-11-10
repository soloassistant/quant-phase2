"""当期信号导出：这个月该持有什么。

  python run_signal.py

只复用回测同一套引擎/参数（xsec 的 12-1 横截面动量，K=3、等权、目标波动 10%、杠杆 ≤1.0），
不另写排序或权重逻辑；本文件只做「取信号 + 排版 + 校验」。

⚠️ 参数口径（评审 P1）：默认参数**不是** README 里 1.02 夏普那套（那套每年重新选参）。
要用选参结果出信号，传 `--params <json>`；无论哪条路径，参数与 sha256 都会写进
metrics.json 与 report.md，避免「用 A 的成绩证明 B 的持仓」。

产物: signals/<YYYYMMDD_HHMMSS>/holdings.csv, report.md, metrics.json
数据缺失/缺月/缺行情 -> 直接失败并指出文件与区间，禁止插值或前值填充。
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SIGNALS_DIR = ROOT / "signals"
sys.path.insert(0, str(ROOT))
import pipeline_core as pc  # noqa: E402
import xsec  # noqa: E402
import live_params  # noqa: E402
from signal_timing import latest_signal, month_end_indices  # noqa: E402

CHINA_TZ = timezone(timedelta(hours=8))
# 默认值 = DEFAULT_PARAMS；main() 里可被 --params 覆盖（覆盖后会写进 metrics.json）
K = live_params.DEFAULT_PARAMS["k"]
MOM_DAYS = live_params.DEFAULT_PARAMS["mom_days"]
USE_FILTER = live_params.DEFAULT_PARAMS["use_filter"]
WEIGHT = live_params.DEFAULT_PARAMS["weight"]
TARGET_VOL = live_params.DEFAULT_PARAMS["target_vol"]
COST_BP = 7.0

NAMES = {"SPY": "标普500 ETF", "QQQ": "纳指100 ETF", "IEF": "7-10年美债 ETF",
         "TLT": "20年+美债 ETF", "GLD": "黄金 ETF", "DBC": "大宗商品 ETF",
         "VNQ": "美国REIT ETF", "EFA": "发达市场(除美) ETF",
         "EEM": "新兴市场 ETF", "UUP": "美元指数 ETF"}

DISCLAIMER = ("由历史样本规则推导的目标持仓，不构成投资建议，"
              "未计入滑点、税费与实际成交约束")


def canonical(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), default=float).encode("utf-8")).hexdigest()


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ranking(close_df, ret_f, i, mutate=None):
    """复算排序，仅用于门禁；生产权重仍取自引擎信号行。"""
    cur, skip_px = close_df.iloc[i], close_df.iloc[i - xsec.SKIP_DAYS]
    base_px = close_df.iloc[i - MOM_DAYS]
    mom = skip_px / base_px - 1.0
    vol60 = ret_f.iloc[i - xsec.VOL60_DAYS + 1: i + 1].std(ddof=1)
    valid = mom.notna() & base_px.notna() & (base_px > 0) & (vol60 > 0)
    cols = list(close_df.columns)
    ranked = sorted([c for c in cols if bool(valid[c])],
                    key=lambda c: (-float(mom[c]), c))
    if mutate == "reverse":
        return ranked[::-1][:K], mom
    return ranked[:K], mom


def _pct(v, d=2):
    return f"{v*100:.{d}f}%"


def main():
    global K, MOM_DAYS, USE_FILTER, WEIGHT, TARGET_VOL
    ap = argparse.ArgumentParser(description="导出当期目标持仓（不输出回测指标）")
    ap.add_argument("--end", help="数据截止 YYYY-MM-DD（默认用最新行）")
    ap.add_argument(
        "--params",
        help="参数文件（JSON，字段同 live_params.DEFAULT_PARAMS）。"
             "想用选参结果出信号就指到那份 JSON；不传则用默认参数，"
             "并在报告里注明「该组参数没有对应的样本外成绩」。",
    )
    ap.add_argument("--mutate", choices=("top5", "reverse"), help=argparse.SUPPRESS)
    args = ap.parse_args()

    try:
        params = live_params.load_params(args.params)
    except live_params.ParamsError as e:
        raise SystemExit(f"[参数错误] {e}")
    K = params["k"]
    MOM_DAYS = params["mom_days"]
    USE_FILTER = params["use_filter"]
    WEIGHT = params["weight"]
    TARGET_VOL = params["target_vol"]
    params_sha = live_params.params_sha256(params)
    # 参数来源决定「能不能引用 1.02 夏普」：默认参数没有对应报告，就别拿它背书
    oos_ref = live_params.OOS_REFERENCE["default_params" if args.params is None else "walk_forward"]

    if args.mutate == "top5":
        raise SystemExit(f"[门禁失败] 持仓数被改成 K=5，锁定 K={K}；拒绝出信号")

    pc.assert_locked()  # 与回测同一处收口，避免信号侧口径漂移
    data = pc.load_data()
    _, close_df = xsec.build_panels(data)
    base_start = xsec.common_start(close_df)
    data_end = close_df.index[-1]
    end = pd.Timestamp(args.end) if args.end else data_end
    if end > data_end:
        raise SystemExit(f"[区间错误] 指定截止 {end.date()} 晚于数据末行 {data_end.date()}")
    cal = close_df.index[close_df.index <= end]
    validation = pc.validate(data, base_start, end)

    ret_f = close_df.pct_change(fill_method=None).ffill()
    si, exec_available = latest_signal(cal)
    sig_date = cal[si]

    res = xsec.run(data, k=K, mom_days=MOM_DAYS, use_filter=USE_FILTER,
                   weight=WEIGHT, target_vol=TARGET_VOL, start=base_start, end=end)
    sig = res.signals
    row = sig[sig["date"] == sig_date]
    if len(row):
        weights = {t: float(row.iloc[0][f"w_{t}"]) for t in pc.UNIVERSE}
        selected = row.iloc[0]["selected"].split(",") if row.iloc[0]["selected"] else []
        est_vol, scale = float(row.iloc[0]["est_vol"]), float(row.iloc[0]["scale"])
    else:  # 数据末端恰为月末：引擎未生成该信号，用同一 make_target 函数补算
        w, diag = xsec.make_target(close_df, ret_f, si, K, MOM_DAYS,
                                   USE_FILTER, WEIGHT, TARGET_VOL)
        weights = {t: float(w[t]) for t in pc.UNIVERSE}
        selected, est_vol, scale = diag["selected"], diag["est_vol"], diag["scale"]

    # 门禁：排序/持仓数必须与引擎一致，改反或改 K 都会红
    expect, mom = ranking(close_df, ret_f, si, mutate=args.mutate)
    if list(selected) != expect:
        raise SystemExit(f"[门禁失败] 引擎选中 {list(selected)}，独立复算 {expect}"
                         + ("（变异: 排序方向反转）" if args.mutate == "reverse" else ""))
    if len(selected) != K:
        raise SystemExit(f"[门禁失败] 持仓数 {len(selected)} != 锁定 K={K}")
    total = sum(weights.values())
    if total > 1.0 + 1e-9:
        raise SystemExit(f"[门禁失败] 实际权重合计 {total:.6f} > 杠杆上限 1.0")
    if scale > 1.0 + 1e-9:
        raise SystemExit(f"[门禁失败] 缩放系数 {scale:.6f} > 1.0")

    # 上期信号（前一个月末）
    prev_me = [i for i in month_end_indices(cal) if i < si]
    prev_weights, prev_date = {}, None
    if prev_me:
        pi = prev_me[-1]
        prev_date = cal[pi]
        prow = sig[sig["date"] == prev_date]
        if len(prow):
            prev_weights = {t: float(prow.iloc[0][f"w_{t}"]) for t in pc.UNIVERSE}
        else:
            pw, _ = xsec.make_target(close_df, ret_f, pi, K, MOM_DAYS,
                                     USE_FILTER, WEIGHT, TARGET_VOL)
            prev_weights = {t: float(pw[t]) for t in pc.UNIVERSE}

    # 成交与成本：优先取引擎已执行的那笔；否则按目标权重差估算
    trade = res.trades[res.trades["signal_date"] == sig_date.date()]
    if exec_available and len(trade):
        tr = trade.iloc[-1]
        exec_date = str(tr["exec_date"])
        gross_frac = float(tr["gross_traded"]) / float(tr["nav_before"])
        cost_bp_est = float(tr["cost"]) / float(tr["nav_before"]) * 1e4
        cost_source = "engine_executed"
    else:
        exec_date = "次月首个交易日（数据未含，待确认）"
        gross_frac = sum(abs(weights[t] - prev_weights.get(t, 0.0)) for t in pc.UNIVERSE)
        cost_bp_est = gross_frac * COST_BP
        cost_source = "estimated_from_target_weights"

    rows = []
    for t in pc.UNIVERSE:
        cur_w, pre_w = weights[t], prev_weights.get(t, 0.0)
        if cur_w <= 1e-12 and pre_w <= 1e-12:
            continue
        status = "新增" if pre_w <= 1e-12 else ("剔除" if cur_w <= 1e-12 else "保留")
        rows.append({"代码": t, "名称": NAMES.get(t, t),
                     "动量得分": f"{float(mom[t])*100:.2f}%" if pd.notna(mom[t]) else "n/a",
                     "目标权重": _pct(1.0 / K) if cur_w > 1e-12 else "0.00%",
                     "实际权重": _pct(cur_w, 4),
                     "信号日": str(sig_date.date()), "成交日": exec_date,
                     "上期权重": _pct(pre_w, 4), "权重变化": f"{(cur_w - pre_w)*100:+.4f}pp",
                     "状态": status})

    files = [{"ticker": t, "file": f"{t}.csv", "sha256": sha_file(pc.DATA_DIR / f"{t}.csv")}
             for t in pc.UNIVERSE]
    payload = {"signal_date": str(sig_date.date()), "exec_date": exec_date,
               "signal_month": str(sig_date.to_period("M")), "data_end": str(data_end.date()),
               "k": K, "mom_days": MOM_DAYS, "weight": WEIGHT, "target_vol": TARGET_VOL,
               # 参数口径必须跟着信号一起落盘：谁引用这个月的持仓，就能核对用的是哪组参数、
               # 以及这组参数到底有没有对应的样本外成绩（评审 P1 的收口点）
               "params": params, "params_sha256": params_sha,
               "params_source": "default" if args.params is None else str(args.params),
               "params_display": live_params.describe(params),
               "oos_reference": oos_ref,
               "scale": round(scale, 12), "est_vol": round(est_vol, 12),
               "selected": list(selected), "rows": rows,
               "turnover_one_way": round(gross_frac, 12), "cost_bp_est": round(cost_bp_est, 12),
               "files": [(f["file"], f["sha256"]) for f in files]}
    signal_sha = canonical(payload)

    prev_path = None
    if SIGNALS_DIR.exists():
        found = sorted(SIGNALS_DIR.glob("*/metrics.json"))
        prev_path = found[-1] if found else None
    if prev_path is None:
        determ = {"status": "first_run", "current_signal_sha256": signal_sha}
    else:
        prev = json.loads(prev_path.read_text(encoding="utf-8"))
        same = prev.get("signal_sha256") == signal_sha
        determ = {"status": "identical" if same else "differs",
                  "current_signal_sha256": signal_sha,
                  "previous_signal_sha256": prev.get("signal_sha256"),
                  "previous_run": prev.get("run", {}).get("run_id")}

    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(CHINA_TZ).strftime("%Y%m%d_%H%M%S")
    run_dir, i = SIGNALS_DIR / stamp, 0
    while run_dir.exists():
        i += 1
        run_dir = SIGNALS_DIR / f"{stamp}_{i:02d}"
    run_dir.mkdir(parents=True)

    headers = ["代码", "名称", "动量得分", "目标权重", "实际权重", "信号日",
               "成交日", "上期权重", "权重变化", "状态"]
    pd.DataFrame(rows, columns=headers).to_csv(run_dir / "holdings.csv",
                                               index=False, encoding="utf-8-sig")

    L = [f"# 当期目标持仓（信号月 {sig_date.to_period('M')}）\n",
         f"- 命令：`{' '.join([sys.executable, str(Path(__file__)), *sys.argv[1:]])}`",
         f"- 数据截止 **{data_end.date()}**；信号日 **{sig_date.date()}**；成交日 **{exec_date}**",
         f"- 口径锁死：t 收盘信号 / t+1 开盘成交 / 月度 / 单边 {COST_BP:g}bp / 杠杆 ≤1.0",
         f"- 参数：{live_params.describe(params)}（`{params_sha}`，来源 {payload['params_source']}）",
         f"- 参数对应的样本外成绩：{oos_ref['report'] or '**没有**'} —— {oos_ref['note']}",
         ("" if args.params is not None else
          "- ⚠️ **注意**：本报告用的是默认参数，仓库里**没有**这组参数的样本外报告。"
          "README 里的 1.02 夏普来自「每年重新选参」的流程，不能拿来给这份持仓背书。"),
         f"- `signal_sha256` **{signal_sha}** ｜ 脚本 `run_signal.py {sha_file(Path(__file__))[:16]}` ｜ "
         f"引擎 `xsec.py {sha_file(ROOT / 'xsec.py')[:16]}`",
         f"- `code_sha256` **{pc.code_sha256()}**（参与计算的源码按固定顺序合并哈希）",
         f"- **{DISCLAIMER}**\n",
         "## 本期目标持仓\n"]
    L.append("| " + " | ".join(headers) + " |")
    L.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        L.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    L.append(f"\n目标权重=未缩放 1/{K}；实际权重=经目标波动 {TARGET_VOL:.0%}、杠杆 ≤1.0 缩放后的值"
             f"（缩放系数 {scale:.2f}，合计 {total*100:.2f}%，其余为现金）。\n")
    L.append("## 换手与成本预估\n")
    L.append("| 项目 | 值 |\n|---|---|")
    L.append(f"| 单边换手（占净值） | {gross_frac*100:.2f}% |")
    L.append(f"| 预估成本（7bp 单边） | {cost_bp_est:.2f}bp |")
    L.append(f"| 来源 | {cost_source} |")
    L.append(f"| 与上期信号日对比 | {prev_date.date() if prev_date is not None else '无上一期'} |\n")
    L.append("## 上期 → 本期变化\n")
    for st in ("新增", "保留", "剔除"):
        L.append(f"- {st}：" + ("、".join(f"{r['代码']}（{r['权重变化']}）" for r in rows
                                        if r["状态"] == st) or "无"))
    L.append("")
    L.append("## 数据口径\n")
    L.append(f"- 数据截止：{data_end.date()}；信号所属月份：{sig_date.to_period('M')}；"
             f"信号日收盘后计算，下一交易日开盘执行。")
    L.append(f"- 完整性：{validation['checked_files']} 个文件 / {validation['months']} 个月，"
             f"缺月份 {validation['missing_months']}、缺行情 {validation['missing_days']}、"
             f"非法行情 {validation['bad_rows']}；全部为 0 才出信号，不做插值或前值填充。")
    for f in files:
        L.append(f"- `{f['file']}` sha256 {f['sha256']}")
    L.append(f"\n**{DISCLAIMER}**\n")
    L.append(f"GATE PASS signal={sig_date.date()} exec={exec_date} picks={len(selected)} "
             f"sum_w={total:.4f} scale={scale:.4f} turnover={gross_frac*100:.2f}% "
             f"cost={cost_bp_est:.1f}bp sha={signal_sha[:16]}")
    (run_dir / "report.md").write_text("\n".join(L), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(
        {"run": {"run_id": run_dir.name,
                 "built_at": datetime.now(CHINA_TZ).strftime("%Y-%m-%d %H:%M:%S"),
                 "command": " ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]),
                 "code_sha256": pc.code_sha256(),
                 "python": sys.version.split()[0], "pandas": pd.__version__},
         "signal": payload, "signal_sha256": signal_sha, "validation": validation,
         "files": files, "determinism": determ},
        ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n完成: {run_dir}")
    print(f"signal_sha256: {signal_sha}")
    print(f"信号 {sig_date.date()} → 成交 {exec_date}；选中 {','.join(selected)}")
    print(f"单边换手 {gross_frac*100:.2f}% / 成本 {cost_bp_est:.2f}bp；确定性 {determ['status']}")
    print(f"GATE PASS signal={sig_date.date()} picks={len(selected)} sum_w={total:.4f} "
          f"scale={scale:.4f} turnover={gross_frac*100:.2f}% cost={cost_bp_est:.1f}bp "
          f"sha={signal_sha[:16]}")


if __name__ == "__main__":
    main()
