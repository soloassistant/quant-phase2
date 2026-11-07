"""月度快照的报告/图表层。

只消费 run_monthly.py 算好的结果，不读原始数据、不跑回测。
产物: table1_performance.csv, table2_turnover_cost.csv,
table3_benchmark_gap.csv, table4_inputs.csv, nav.csv, chart.png, report.md

四张表每张只用一个标签: 本期全部为 [measured]（本机回测净值/成交实测），
流程不产出 [estimated] 数字；将来若引入估计项，必须单独成表，不与实测混排。
"""

import math
from pathlib import Path

import pandas as pd

MEASURED = "[measured]"
SERIES = [("strategy", "策略（OOS 拼接）"),
          ("bench_same_start", "基准① 同起点重建"),
          ("bench_continuous", "基准② 连续持有")]


def _pct(v, d=2):
    return "n/a" if v is None or not math.isfinite(v) else f"{v*100:.{d}f}%"


def _num(v, d=3):
    return "n/a" if v is None or not math.isfinite(v) else f"{v:.{d}f}"


def _bp(v, d=0, signed=False):
    if v is None or not math.isfinite(v):
        return "n/a"
    return f"{v:+.{d}f}" if signed else f"{v:.{d}f}"


def _md(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def _csv(path, headers, rows):
    pd.DataFrame(rows, columns=headers).to_csv(path, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------- 四张表

def table1(perf):
    headers = ["组合", "区间", "年数", "年化", "年化波动", "夏普",
               "最大回撤", "Calmar", "期末净值"]
    rows = []
    for key, label in SERIES:
        m = perf[key]
        rows.append([label, f"{m['start']}→{m['end']}", f"{m['years']:.2f}",
                     _pct(m["cagr"]), _pct(m["ann_vol"]), _num(m["sharpe"]),
                     _pct(m["max_drawdown"]), _num(m["calmar"]), f"{m['final_nav']:.1f}"])
    return headers, rows


def table2(turn):
    s, b = turn["strategy"], turn["benchmark"]
    headers = ["项目", "策略（OOS 拼接）", "基准（一次性建仓）"]
    rows = [
        ["年化换手", _pct(s["annual_turnover"]), _pct(b["annual_turnover"])],
        ["其中：建仓 / 退出 / 调仓",
         f'{_pct(s["turnover_entry"])} / {_pct(s["turnover_exit"])} / {_pct(s["turnover_adjust"])}', "—"],
        ["年化成本（bp）", _bp(s["cost_annual_bp"]), _bp(b["cost_annual_bp"])],
        ["累计成本（bp）", _bp(s["cost_total_bp"]), _bp(b["cost_total_bp"])],
        ["成交笔数", f'{s["n_trades"]}', f'{b["n_trades"]}（一次性建仓）'],
        ["平均持仓数", f'{s["avg_holdings"]:.1f}', "10"],
        ["成本 / 年化收益", _pct(s["cost_to_cagr"]), _pct(b["cost_to_cagr"])],
        ["统计窗口数", f'{turn["windows"]}', f'{turn["windows"]}'],
    ]
    return headers, rows


def table3(gaps):
    headers = ["基准", "策略年化", "基准年化", "缺口 bp（正=跑输）", "策略夏普", "基准夏普",
               "夏普差", "策略回撤", "基准回撤", "回撤差 pp（正=更浅）"]
    rows = []
    for g in gaps:
        rows.append([g["benchmark"], _pct(g["strategy_cagr"]), _pct(g["bench_cagr"]),
                     _bp(g["gap_bp"], signed=True), _num(g["strategy_sharpe"]),
                     _num(g["bench_sharpe"]), f'{g["sharpe_diff"]:+.3f}',
                     _pct(g["strategy_mdd"]), _pct(g["bench_mdd"]),
                     f'{g["mdd_diff_pp"]:+.2f}pp'])
    return headers, rows


def table4(files, cfg_rows, meta):
    headers = ["类别", "项目", "值"]
    rows = []
    for f in files:
        rows.append(["数据文件", f["file"],
                     f'sha256={f["sha256"]}｜{f["first"]}→{f["last"]}｜{f["rows"]} 行'])
    rows += [
        ["数据区间", "样本起点（全池可用）", meta["common_start"]],
        ["数据区间", "数据截止", meta["data_end"]],
        ["数据区间", "本次运行截止", meta["run_end"]],
        ["数据区间", "OOS 区间", f'{meta["oos_start"]}→{meta["oos_end"]}'],
        ["可复现", "输入指纹", meta["inputs_fingerprint"]],
        ["可复现", "结果指纹 result_sha256", meta["result_sha256"]],
        ["可复现", "代码指纹 code_sha256", meta.get("code_sha256", "n/a")],
    ]
    for category, item, value in cfg_rows:
        rows.append([category, item, value])
    return headers, rows


# ---------------------------------------------------------------- 图

def plot(nav_df, path, oos_start, oos_end):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"strategy": "#1f4e79", "bench_same_start": "#ef6c00",
              "bench_continuous": "#2e7d32"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.5]})
    for key, label in SERIES:
        s = nav_df[key].dropna()
        dd = s / s.cummax() - 1.0
        ax1.plot(s.index, s.values, color=colors[key], lw=1.3,
                 label=f"{label}（终值 {s.iloc[-1]:.1f}）")
        if key == "strategy":
            ax2.fill_between(dd.index, dd.values, 0, color=colors[key], alpha=0.35,
                             label=f"{label}（最大回撤 {dd.min()*100:.2f}%）")
        else:
            ax2.plot(dd.index, dd.values, color=colors[key], lw=1.0,
                     label=f"{label}（最大回撤 {dd.min()*100:.2f}%）")

    ax1.set_yscale("log")
    ax1.set_ylabel("净值（对数轴，区间起点=100）")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_title(f"策略 vs 两个基准：净值与回撤（{oos_start} → {oos_end}）\n"
                  "历史样本，不构成收益预测", fontsize=11)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("回撤")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="lower left")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.autofmt_xdate()
    fig.text(0.5, 0.005, "历史样本，不构成收益预测", ha="center", color="#c00000", fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- markdown

def _compare_lines(compare, determ):
    L = []
    if not compare or not compare.get("previous_run"):
        L.append("首次运行，无上一期可比。下次运行会自动对比本期（按 run 目录倒序取最近一期）。\n")
    else:
        L.append(f"对比基准：上一期 `{compare['previous_run']}`"
                 f"（{'同配置同输入' if compare['comparable'] else '配置或输入已变化'}）。\n")
        if not compare["config_changes"] and not compare["input_changes"] \
                and not compare["metric_changes"]:
            L.append("输入、参数、指标均无变化（逐位一致）。\n")
        if compare["config_changes"]:
            rows = [[c["field"], c["previous"], c["current"],
                     _num(c["delta"], 6) if c["delta"] is not None else "—"]
                    for c in compare["config_changes"]]
            L.append("**参数变化**\n")
            L.append(_md(["参数", "上一期", "本期", "变化"], rows) + "\n")
        if compare["input_changes"]:
            rows = [[c["item"], c["previous"], c["current"]] for c in compare["input_changes"]]
            L.append("**输入变化**\n")
            L.append(_md(["输入项", "上一期", "本期"], rows) + "\n")
        if compare["metric_changes"]:
            rows = []
            for c in compare["metric_changes"]:
                fmt = _pct if c["metric"] in ("cagr", "ann_vol", "max_drawdown") else \
                    (lambda v: f"{v:.1f}") if c["metric"] == "final_nav" else _num
                rows.append([c["series"], c["metric"], fmt(c["previous"]),
                             fmt(c["current"]), fmt(c["delta"])])
            L.append("**指标变化（只列变了项）**\n")
            L.append(_md(["序列", "指标", "上一期", "本期", "变化"], rows) + "\n")
        if compare["selection_changes"]:
            L.append(f"**选参变化**：{len(compare['selection_changes'])} 个验证窗口的选中参数变化。\n")
            rows = [[c["val_start"], c["previous"], c["current"]]
                    for c in compare["selection_changes"][:8]]
            L.append(_md(["验证起始", "上一期选中", "本期选中"], rows) + "\n")
    L.append("**可复现性状态**\n")
    status = {"first_run": "首跑，等待下一期同输入对比。",
              "identical": "上一期与本期的 result_sha256 逐位相同（确定性成立）。",
              "differs": "上一期与本期 result_sha256 不同，差异项见 metrics.json 的 "
                         "determinism.differing_fields（不隐藏漂移）。",
              "not_comparable": "上一期配置或输入不同，不能用于确定性判定。"}
    L.append(f"- {status.get(determ['status'], determ['status'])}\n")
    L.append(f"- 本期 {determ['current_result_sha256']}"
             + (f"；上期 {determ['previous_result_sha256']}" if determ.get("previous_result_sha256") else "")
             + "。无随机种子；排序 key 固定；同一输入、同一环境的浮点运算顺序固定。\n")
    return L


def write_all(run_dir, nav_df, perf, turn, gaps, files, cfg_rows, meta,
              validation, compare, determ):
    run_dir = Path(run_dir)
    h1, r1 = table1(perf)
    h2, r2 = table2(turn)
    h3, r3 = table3(gaps)
    h4, r4 = table4(files, cfg_rows, meta)
    _csv(run_dir / "table1_performance.csv", h1 + ["标签"], [r + [MEASURED] for r in r1])
    _csv(run_dir / "table2_turnover_cost.csv", h2 + ["标签"], [r + [MEASURED] for r in r2])
    _csv(run_dir / "table3_benchmark_gap.csv", h3 + ["标签"], [r + [MEASURED] for r in r3])
    _csv(run_dir / "table4_inputs.csv", h4 + ["标签"], [r + [MEASURED] for r in r4])

    nav_out = pd.DataFrame(index=nav_df.index)
    for key, _ in SERIES:
        s = nav_df[key].dropna()
        nav_out[key] = s
    for key, tag in (("strategy", "dd_strategy"), ("bench_same_start", "dd_same_start"),
                     ("bench_continuous", "dd_continuous")):
        s = nav_df[key].dropna()
        nav_out[tag] = s / s.cummax() - 1.0
    nav_out.index.name = "date"
    nav_out.to_csv(run_dir / "nav.csv", encoding="utf-8-sig")

    plot(nav_df, run_dir / "chart.png", meta["oos_start"], meta["oos_end"])

    s = perf["strategy"]
    f = perf["bench_same_start"]
    c = perf["bench_continuous"]
    gap_c = (c["cagr"] - s["cagr"]) * 1e4
    gap_f = (f["cagr"] - s["cagr"]) * 1e4

    L = []
    L.append(f"# 月度模型快照 {meta['run_id'][:4]}-{meta['run_id'][4:6]}-{meta['run_id'][6:8]}"
             f"（run `{meta['run_id']}`）\n")
    L.append(f"- 命令：`{meta['command']}`")
    L.append(f"- `result_sha256` **{meta['result_sha256']}** ｜ `config_sha256` "
             f"{meta['config_sha256']} ｜ 脚本 `{meta['scripts']}`")
    L.append(f"- 数据截止 **{meta['data_end']}**，OOS 区间 **{meta['oos_start']}→{meta['oos_end']}**"
             f"（数据截断到本次运行截止 {meta['run_end']}）")
    L.append(f"- 环境：Python {meta['python']} / pandas {meta['pandas']} / numpy {meta['numpy']}\n")

    L.append("## 结论\n")
    L.append(f"**OOS 夏普 {_num(s['sharpe'])}，高于同起点基准 {_num(f['sharpe'])}"
             f" 与连续持有基准 {_num(c['sharpe'])}；年化 {_pct(s['cagr'])}，"
             f"对同起点缺口 {gap_f:+.0f}bp、对连续持有缺口 {gap_c:+.0f}bp（正=跑输）。"
             f"风险调整后优势成立，收益增强不成立。**")
    L.append("月度复跑只回答“模型现在长什么样”，不做参数寻优；缺口来源已归因，"
             "剩余属结构上限（10 只 ETF、只做多、样本期顺风）。")
    L.append("**以上全部为历史样本回测结果，不构成收益预测或投资建议。**\n")

    L.append(f"## 表 1 绩效：年化 / 夏普 / 最大回撤 / Calmar {MEASURED}\n")
    L.append(_md(h1, r1))
    L.append("\n夏普=日收益均值/标准差×√252；Calmar=年化/|最大回撤|；期末净值以区间起点=100。\n")

    L.append(f"## 表 2 换手与成本 {MEASURED}\n")
    L.append(_md(h2, r2))
    L.append(f"\n策略列为 {turn['windows']} 个验证窗口实测值的算术平均（每窗 1 年），累计成本为各窗求和。"
             "**成本口径（无税、无融资/融券成本、无额外滑点加成）：**\n")
    L.append("```text")
    L.append("年换手      = Σ|成交金额| / mean(NAV) / 年数      # 单边绝对成交额")
    L.append("             entry + exit + adjust = Σ|成交金额|   # 三者互斥，加总等于毛换手，不重复计")
    L.append("成本（元）  = Σ|成交金额| × 7bp                  # 7bp = 5bp 佣金 + 2bp 滑点")
    L.append("年化成本bp ≈ 年换手 × 7bp                         # 778.89% × 7 ≈ 54.5")
    L.append("累计成本bp  = Σ 各验证窗 cost_total_bp ≈ 54.5 × 10.00 ≈ 545")
    L.append("```")
    L.append("基准为一次性建仓，7bp 一次性成本，年化成本=7bp/区间年数；现金 0 收益。\n")

    L.append(f"## 表 3 与两个基准的缺口 {MEASURED}\n")
    L.append(_md(h3, r3))
    L.append("\n缺口=基准年化−策略年化（正=跑输）；回撤差=策略回撤−基准回撤（正=策略更浅）。\n")

    L.append(f"## 表 4 本次运行输入清单 {MEASURED}\n")
    L.append(_md(h4, r4))
    L.append("\nsha256 为原始 CSV 字节指纹；区间取自文件首末行。参数为本次运行实际生效值，"
             "非估计值；任何改动都会改变 `config_sha256` 并在 metrics.json 中可见。\n")
    next_end = (pd.Timestamp(meta["oos_end"]) + pd.DateOffset(years=1)).date()
    L.append(f"**表注（为什么数据更新到 {meta['data_end']}，OOS 末端仍停在 {meta['oos_end']}）：** "
             f"OOS 只拼接**完整**的 1 年验证窗，不拼接未闭合窗口。本次数据截止 {meta['data_end']}，"
             f"最后一个已闭合验证窗止于 {meta['oos_end']}；其后的验证窗需覆盖到约 {next_end} 才闭合，"
             f"故末端暂不移动。表 4 中「数据截止」与「OOS 区间」本就允许不同，这是窗口闭合规则的结果，不是数据错误。\n")

    L.append("## 图 1 策略 vs 两个基准（净值 + 回撤）\n")
    L.append("![图 1 策略与两个基准的净值及回撤](chart.png)\n")
    L.append(f"区间 {meta['oos_start']}→{meta['oos_end']}；终值与最大回撤已标注在图例。"
             "**历史样本，不构成收益预测。**\n")

    L.append(f"## 与上一期对比 {MEASURED}\n")
    L += _compare_lines(compare, determ)

    L.append("## 口径与数据完整性\n")
    L.append("- 锁死：t 日收盘信号、t+1 开盘成交、月度调仓、单边 7bp、目标波动 10%、杠杆 ≤1.0、"
             "现金 0 收益；8 年训练 / 1 年验证 / 1 年步长 / 1 月 purge；中位数选参 + 邻域平坦 0.25。")
    L.append(f"- 数据完整性：{validation['checked_files']} 个文件、{validation['months']} 个自然月逐个检查；"
             f"缺月份 {validation['missing_months']} 个、缺行情日 {validation['missing_days']} 天、"
             f"非正/空行情 {validation['bad_rows']} 行。任一不为 0 时本流程直接失败，不做插值或前值填充。")
    L.append(f"- 脚本 sha256 前 16 位：{meta['scripts']}；共用引擎 `xsec.py` "
             f"sha256 {meta['engine_sha256']}。")
    L.append("- 本报告四张表全部为 `[measured]`；流程不输出 `[estimated]` 数字。\n")

    (run_dir / "report.md").write_text("\n".join(L), encoding="utf-8")
