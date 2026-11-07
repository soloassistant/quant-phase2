# 量化模型月度复跑

这不是策略优化目录，而是「每月同一口径下，模型现在长什么样」的快照流水线。
策略结论已定、不再调参；本目录只负责可复现地产出四张表、一张图和一份可对比的运行记录。

## 一条命令

```powershell
D:\Agent\quant\.venv\Scripts\python.exe D:\Agent\quant-phase2\run_monthly.py
```

- 输出到 `d:\Agent\quant-phase2\runs\<YYYYMMDD_HHMMSS>\`，每次新建目录，**不覆盖任何历史**。
- 同一天重复运行会追加 `_01`、`_02` 后缀，旧目录原样保留。
- 可选 `--end YYYY-MM-DD`：把本次运行的数据截止钉在指定日，用于和上一期做「同输入、只比参数」。

每次运行产出：

| 文件 | 内容 |
|---|---|
| `report.md` | 结论、四张表、图、与上一期对比、可复现性状态 |
| `table1_performance.csv` | 年化 / 夏普 / 最大回撤 / Calmar `[measured]` |
| `table2_turnover_cost.csv` | 换手与成本 `[measured]` |
| `table3_benchmark_gap.csv` | 与同起点重建、连续持有两个基准的缺口 `[measured]` |
| `table4_inputs.csv` | 数据文件 sha256、区间、生效参数 `[measured]` |
| `chart.png` | 策略与两个基准的净值 + 回撤，标注区间与终值，写明「历史样本，不构成收益预测」 |
| `nav.csv` | 三条净值与各自回撤，按日 |
| `metrics.json` | 全量数字、`effective_config`、输入清单、determinism、对比上一期 |

四张表每张只用一个标签；本流程全部是 `[measured]`（本机回测净值/成交实测），
不产出 `[estimated]` 数字。将来若引入估计项，必须单独成表，不与实测混排。

## 当期信号导出（每月怎么用）

回测报告只回答「过去长什么样」，信号导出回答「这个月该持有什么」。两者共用同一套引擎
`xsec.py` 与同一组锁死口径，不另写排序或权重逻辑。

```powershell
D:\Agent\quant\.venv\Scripts\python.exe D:\Agent\quant-phase2\run_signal.py
```

- 输出到 `d:\Agent\quant-phase2\signals\<YYYYMMDD_HHMMSS>\`，每次新建目录，不覆盖历史。
- `holdings.csv`：代码、名称、动量得分、目标权重（未缩放 1/3）、实际权重（目标波动 10% +
  杠杆 ≤1.0 缩放后）、信号日、成交日、上期权重、权重变化、新增/保留/剔除。
- `report.md`：本期持仓、与上期对比、单边换手与 7bp 成本预估、数据口径、门禁尾行。
- `metrics.json`：`signal_sha256`、输入文件 sha256、完整性检查、与上一次信号的确定性对比。

**什么时候跑**：每月**最后一个交易日收盘后**运行，信号在当日收盘价上计算，**次月首个交易日
开盘执行**，与回测的 t 收盘 / t+1 开盘完全一致。若运行时尚无次月交易数据，成交日显示
「次月首个交易日（数据未含，待确认）」，成本按目标权重差估算；数据补齐后重跑即可得到
真实成交日与成本。

**纪律**：

- 数据缺一个月或缺一天直接失败并指出文件与区间；不插值、不前值填充。
- 同一输入跑两次，`holdings.csv` 逐字节相同，`signal_sha256` 相同；第二期会写
  `determinism.status = identical / differs / first_run`。
- 信号表只讲持仓，不混入年化、夏普、回测缺口；页眉与页脚均写明
  「由历史样本规则推导的目标持仓，不构成投资建议，未计入滑点、税费与实际成交约束」。
- 门禁：「改坏必红」。持仓数改成 5 或排序方向反转都必须报错退出：

```powershell
D:\Agent\quant\.venv\Scripts\python.exe D:\Agent\quant-phase2\run_signal.py --mutate top5
D:\Agent\quant\.venv\Scripts\python.exe D:\Agent\quant-phase2\run_signal.py --mutate reverse
```

说明：信号导出按用户指定的主组合固定参数（K=3 / 12-1 / 等权 / 无过滤）；月度报告里的 OOS
绩效使用每年重新选参的 walk-forward 结果。两者共用引擎与执行口径，但「选参方式」不同，
不可用信号表去解释 OOS 换手。

## 口径锁死

以下六项写在 `pipeline_core.py` / `xsec.py`，运行前会断言，任何改动都会改变
`config_sha256` 并出现在 `metrics.json.effective_config` 与表 4：

- t 日收盘信号
- t+1 开盘成交
- 月度调仓
- 单边 7bp（5bp 佣金 + 2bp 滑点）
- 目标波动 10%
- 杠杆 ≤ 1.0，现金 0 收益

另固定：8 年训练 / 1 年验证 / 1 年步长 / 1 月 purge、24 组候选、中位数选参 + 邻域平坦 0.25、起点偏移 0。
OOS 只拼完整验证年，所以数据每月更新时 OOS 末端可能暂时不动，直到下一个完整验证年闭合；
表 4 同时给出「数据截止」和「OOS 区间」，两者不同不是错误。

## 口径变更（2026-11-06）：535% / 37bp 与 778.89% / 55bp 不是同一对象

历史结论里的「换手 535%/年、成本最多值 37bp」来自 `out_turnover/`（方向 2 压换手）的
**固定参数基线**，不是本流水线的 OOS 拼接结果。两者公式相同，差的是**计算对象**：

| 项 | 旧口径（`out_turnover` 基线） | 新口径（本流水线） |
|---|---|---|
| 对象 | 静态固定参数：K=3 / 12-1 / 无过滤 / 等权 | walk-forward：每年用前 8 年重新选 K、动量、过滤、权重 |
| 区间 | 全样本 2008-02-29 → 2026-11-04（18.68 年） | OOS 2016-02-29 → 2026-02-27（10.00 年，10 个完整验证年） |
| 换手 | 534.7%（报告四舍五入 535%） | 778.89% |
| 年化成本 | 37.4bp | 54.5bp（报告取整 55bp） |
| 累计成本 | — | 545bp |

**旧口径的错不在算法，在把「基线」当成了「模型」。** `turnover.py` 的 `engine.run(...)`
固定 K=3/12-1/无过滤/等权，全样本一次跑完；本流水线的策略参数逐年在变。
用它替本流水线背书，等于用一条从未上线的固定参数线去描述一个每年换参的模型。

差异拆解（同一引擎、同一 7bp、同一月度口径，只换对象）：

| 步骤 | 年换手 | 相对上一行 |
|---|---|---|
| 1. 固定参数，全样本 18.68 年 | 534.7% | — |
| 2. 固定参数，只取 OOS 10 年 | 539.5% | +4.8pp（纯样本区间效应） |
| 3. OOS walk-forward，10 年 | 778.9% | +239.4pp（参数逐年变化的效应） |

剩余约 7.8pp/年来自组合从现金起步、首个验证年一次性建仓，属边界效应，不是重复计数。
第 3 行多数验证窗选中 `inv_vol` 权重或更短的动量窗（单窗换手区间 376.8%–1277.3%），
其中 2019 年窗（K=3/9m/过滤/inv_vol）达 1277.3%，把 10 年均值抬高。
**因此 778.89% 是正确口径；534.7% 是另一条线，二者不可互换，也不可相加。**

成本口径（两处口径一致，无税、无融资成本、无可变滑点加成）：

```text
年换手        = Σ|成交金额| / mean(NAV) / 年数          # 单边绝对成交额
其中 entry + exit + adjust = Σ|成交金额|                 # 三者互斥且加总等于毛换手
成本（元）    = Σ|成交金额| × 7bp                        # 5bp 佣金 + 2bp 滑点
年化成本 bp   ≈ 年换手 × 7bp                             # 534.7×7≈37.4；778.9×7≈54.5
累计成本 bp   = Σ 各验证窗 cost_total_bp ≈ 54.5 × 10.00 ≈ 545
```

即 55bp/545bp **不含税、不含融券/融资、不含额外滑点**，就是「单边 7bp × 换手 × 年数」。
本次口径核对未改动任何算法，`result_sha256` 保持 `00ba98b88cfceac6c1e537f69df850519d8cee117a219d2b39404a2f03fd8108`。

## 改参数、以及怎么对比上一期

**改哪里**：`pipeline_core.py` 顶部的 `CANDIDATES` / `MOM_MONTHS` / `FLAT`，
以及 `effective_config()` 里的 `walk_forward`、`candidate_grid`、`selection`。
锁死项在 `locked` 字典里，不建议改；确需实验时必须同时改断言和本 README。

**改完怎么跑**：

```powershell
# 推荐：钉住与上一期相同的数据截止，把「参数变化」和「新数据」分开
D:\Agent\quant\.venv\Scripts\python.exe D:\Agent\quant-phase2\run_monthly.py --end 2026-11-04
```

**不是重跑就算了，输出必须回答「哪一项变了、变了多少」**：

1. `report.md` 的「与上一期对比 `[measured]`」段会自动取最近一个历史 run：
   - 参数变化表：`参数 | 上一期 | 本期 | 变化`（数值项给出 delta）；
   - 输入变化表：数据文件 sha256、样本起点、数据截止、运行截止、验证窗口数；
   - 指标变化表：只列变了项，按 `序列 | 指标 | 上一期 | 本期 | 变化`；
   - 选参变化：哪些验证窗口的 K / 动量月 / 过滤 / 权重变了。
2. `metrics.json` 的 `compare_vs_previous` 是同一份内容的机器可读版本：
   `config_changes[]`（field / previous / current / delta）、`input_changes[]`、
   `metric_changes[]`（series / metric / previous / current / delta）、`selection_changes[]`。
3. `run.result_sha256` 是结果指纹；配置或输入变了，它就会变。
4. 如果只想比参数、不想让新数据干扰：先用上一期的 `--end` 重跑本期，再改参数重跑；
   两次的 `input_changes` 应为空，`config_changes` 才是纯参数效应。

示例读法：上一期 `config_changes = [{"field":"selection.flat_neighborhood",
"previous":0.25,"current":0.5,"delta":0.25}]`，指标表里 `strategy.sharpe` 从 1.024 到 0.981
（delta −0.043），就是「平坦阈值放宽 0.25，夏普降 0.043」这一条结论，而不是一句「跑完了」。

## 可复现性

- 同一输入、同一 Python/pandas/numpy 环境跑两次，`result_sha256` 必须逐位相同。
- 流程无随机种子、无重采样；排序 key（动量降序 + 代码字典序）固定；浮点求和顺序固定。
- 每次运行都会和最近一期比较：
  - `determinism.status = identical`：上一期与本期同配置同输入，指纹逐位相同；
  - `status = differs`：指纹不同，`differing_fields` 列出具体漂移字段（不隐藏）；
  - `status = not_comparable`：配置或输入已变，不能用于确定性判定；
  - `status = first_run`：首跑，等待下一期同输入对比。
- 环境与脚本指纹：`metrics.json.run.python/pandas/numpy`、`run.scripts`、`run.engine_sha256`。
- 代码指纹 `code_sha256`：对参与计算的源码按**固定顺序**（`run_monthly.py` → `run_signal.py` →
  `pipeline_core.py` → `monthly_report.py` → `engine.py` → `quant-xsec/xsec.py`）逐文件算 sha256，
  拼成 `文件名:哈希` 行后再整体 sha256。顺序按「入口 → 整合/报告 → 引擎」写死，不依赖目录枚举，
  换机器不会漂移；路径用相对标签，移动目录也不变。回测写入 `metrics.json.run.code_sha256`
  与表 4，信号写入 `signals/*/metrics.json.run.code_sha256` 与报告头。

## 数据缺失直接失败

运行前对 10 个 ETF 逐文件检查（窗口 = 全池可用起点 → 本次截止）：

- 文件是否存在、日期是否重复；
- 每个自然月是否至少有一行（缺月份直接失败）；
- 相对全池交易日历是否缺行情日（缺行情直接失败）；
- open/close 是否为空或非正（直接失败）。

失败时退出并打印 `[数据完整性失败] <文件路径>: <缺失区间/月份>`。
**不做插值、不做前值填充、不跳过该月**。补齐数据后重跑即可。

## BLOCKED：加资产类别 / 时点成分池

这条线保持 BLOCKED，不是「以后再说」，而是有明确接入口和前条件。**前置条件全部满足才允许接**：

1. **离线数据文件**：至少 30 个可交易标的、覆盖 ≥5 个资产类别，含**已清盘/已退市代码**；
   日频复权 OHLC，时间覆盖与现有样本重叠 ≥10 年（能回溯到 2003 年以前更好）。
2. **点位时成分表**：每个标的的纳入/剔除日期、退市日期；不得只用「今天还活着的代码」。
3. **完整性**：每个标的在分析窗口内无缺月份、无连续缺行情；每个文件提供 sha256。
4. **授权**：数据许可允许本地回测与内部报告使用。
5. **验收**：
   - 用 `pipeline_core.load_data` 的同一校验逻辑跑一遍，缺月份/缺行情必须报错而不是填充；
   - 在 `run_monthly.py` 的输入清单里新增文件 sha256 与区间；
   - 同一锁死口径（t 收盘信号 / t+1 开盘 / 月度 / 7bp / 10% 目标波动 / 杠杆 ≤1.0）重跑；
   - 输出新 run 目录，并在 `report.md` 的对比段说明：池从几只扩到几只、哪些类别、样本期与
     基准如何重建；同时报告存活偏差检查（退市代码是否真的进入历史排序）。
6. **不做的事**：不用代理数据、不用今天成分回填历史、不对缺失月份插值；拿不到就维持
   「10 只 ETF、只做多、样本期顺风」的结构上限表述。

接入位置：新增数据放 `data_pit/`，在 `pipeline_core.load_data` 注册并扩展 `UNIVERSE`；
`run_monthly.py` 的 manifest 自动记录新文件 sha256。首跑后先看 `determinism.status`
与 `compare_vs_previous`，确认扩展只带来「输入变化」，没有口径漂移。

## 文件分工

| 文件 | 职责 |
|---|---|
| `run_monthly.py` | 编排：配置、指纹、manifest、与上一期对比、determinism、建 run 目录 |
| `pipeline_core.py` | 引擎整合：数据硬校验、walk-forward、指标汇总、生效配置 |
| `monthly_report.py` | 报告/图表：四张表、净值+回撤图、markdown |
| `run_signal.py` | 当期信号：取月末信号、目标持仓表、与上期对比、换手/成本预估、门禁 |
| `xsec.py`（`d:\Agent\quant-xsec`） | 横截面动量引擎（只读共用） |

历史研究产物（`out_oos/`、`out_risk/`、`out_turnover/`、`out_universe/`）与 `SUMMARY.md`
保持原样，不被本流水线覆盖。
