# -*- coding: utf-8 -*-
"""A股黄金坑筛选报表 v4 生成器（静态数值 + 内嵌K线，无公式）。"""
import json
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import FormulaRule

OUT_DIR = Path(r"C:/Users/geral/WorkBuddy/2026-11-09-11-07-54/.tmp-gh-check/quant-live/experiments/out")
XLSX = OUT_DIR / "A股黄金坑筛选报表_v4.xlsx"
CH = OUT_DIR / "charts"
R = json.loads((OUT_DIR / "ashare_rank.json").read_text(encoding="utf-8"))
S = json.loads((OUT_DIR / "self_audit.json").read_text(encoding="utf-8"))
C = json.loads((OUT_DIR / "compare_models.json").read_text(encoding="utf-8"))


def xc(c):
    v = c[1:] if c.startswith("#") else c
    return "FF" + v.upper()


HP, HT, HL, HB = xc("4472C4"), xc("2F5597"), xc("D9E2F3"), xc("BFBFBF")
HG, HGT, HR, HRT = xc("C6EFCE"), xc("006100"), xc("FFC7CE"), xc("9C0006")
HGY, HBL, HO = xc("FFF2CC"), xc("DDEBF7"), xc("ED7D31")
th = Side(style="thin", color=HB)
B = Border(left=th, right=th, top=th, bottom=th)
HF = PatternFill("solid", fgColor=HP)
TF = PatternFill("solid", fgColor=HT)
LF = PatternFill("solid", fgColor=HL)
GF = PatternFill("solid", fgColor=HGY)
GR = PatternFill("solid", fgColor=HG)
RF = PatternFill("solid", fgColor=HR)
OF = PatternFill("solid", fgColor=HO)
HFONT = Font(bold=True, color="FFFFFFFF", size=10)
TFONT = Font(bold=True, color="FFFFFFFF", size=14)
BOLD = Font(bold=True, size=10.5)
N = Font(size=10)
CEN = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT = Alignment(horizontal="left", vertical="center", wrap_text=True)
RIGHT = Alignment(horizontal="right", vertical="center")


def trow(ws, t, n):
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=n)
    c = ws.cell(1, 1, t)
    c.fill = TF
    c.font = TFONT
    c.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 28


def hrow(ws, r, hs, ws_):
    for j, (h, w) in enumerate(zip(hs, ws_), start=1):
        c = ws.cell(r, j, h)
        c.fill = HF
        c.font = HFONT
        c.alignment = CEN
        c.border = B
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.row_dimensions[r].height = 36


def shape(r):
    """形态标签：把'缓涨'和'深坑反弹'分开。250日收益为负=深坑反弹（正是'挣扎最猛'的变体）。"""
    if r.get("ret_250", 0) is not None and r["ret_250"] < 0:
        return "深坑反弹"
    if (r.get("vol20") or 0) > 0.35 or (r.get("max_dd_250") or 0) < -0.28:
        return "高波动"
    return "缓涨"


ranked = R["ranking"]
core = [r for r in ranked if r["suggest"] == "核心"]
G = R["gates"]
snap = R.get("snapshot", {})

wb = Workbook()
# ---------------- Sheet1 筛选结果 ----------------
ws = wb.active
ws.title = "筛选结果"
ws.sheet_view.showGridLines = False
trow(ws, f"A股「黄金坑」筛选 v4 · 绝对便宜度+绝对趋势+跑赢大盘+流动性 · 59→核心{len(core)}", 15)
ws.merge_cells("A2:O2")
ws.cell(2, 1, "v4 修的是 v3 的残留病根：便宜度仍用'组内百分位'（3只组必然产生 0.333/0.667/1.0，"
              "最小 0.333 > 门槛 0.30 → 品牌/金融两类'不够便宜'永不触发）→ 改绝对刻度。").font = Font(size=9, italic=True, color="FF595959")
ws.cell(2, 1).alignment = LEFT
ws.row_dimensions[2].height = 26
ws["A3"] = "硬性门槛（AND，任一不满足即否决）"
ws["A3"].font = Font(bold=True, size=11, color=HT)
ws.merge_cells("A4:O4")
ws["A4"] = (f"① 绝对便宜度≥{G['cheap_min']:.2f}（{G['cheap_scale']}）；"
            f"② 收盘 > 自己200日均线；③ 近120日跑赢沪深300；"
            f"④ 无黑天鹅(单日<-15%/5日<-25%)；⑤ 20日均成交额≥{G['amt_min_yi']:.2f}亿。")
ws["A4"].font = N
ws["A4"].alignment = LEFT
ws["A4"].fill = LF
ws.row_dimensions[4].height = 34
ws["A6"] = f"核心 {len(core)} 只（全部通过五项硬门槛）"
ws["A6"].font = Font(bold=True, size=11)
h = ["排名", "代码", "名称", "体系", "现价", "绝对便宜度", "距200线", "120日", "250日",
     "超沪深300", "超中证500", "250回撤", "成交额(亿)", "放量x", "形态"]
w = [6, 8, 10, 6, 9, 10, 9, 8, 8, 11, 11, 9, 10, 7, 10]
hrow(ws, 7, h, w)
base = 8
for i, r in enumerate(core):
    rr = base + i
    vals = [r.get("rank"), r["code"], r["name"], r["system"], r["price"], r["cheap"],
            r.get("dist_sma200"), r.get("ret_120"), r.get("ret_250"), r.get("excess300"),
            r.get("excess500"), r.get("max_dd_250"), r.get("amt20_yi"), r.get("vol_spike"), shape(r)]
    fs = [None, None, None, None, "#,##0.00", "0.00", "+0.0%;-0.0%", "+0%;-0%", "+0%;-0%",
          "+0%;-0%", "+0%;-0%", "0%", "0.0", "0.0", None]
    for j, (v, f) in enumerate(zip(vals, fs), start=1):
        c = ws.cell(rr, j, v)
        c.border = B
        c.font = N
        if f:
            c.number_format = f
        c.alignment = CEN if j in (1, 2, 4, 15) else (LEFT if j == 3 else RIGHT)
    # 形态上色：深坑反弹=橙、高波动=灰、缓涨=绿
    sp = shape(r)
    cell = ws.cell(rr, 15)
    if sp == "深坑反弹":
        cell.fill = OF
        cell.font = Font(size=10, bold=True, color="FFFFFFFF")
    elif sp == "高波动":
        cell.fill = GF
        cell.font = Font(size=10, color="FF7F6000")
    else:
        cell.fill = GR
        cell.font = Font(size=10, color=HGT)
    if (r.get("vol_spike") or 0) > 5:
        ws.cell(rr, 14).font = Font(size=10, bold=True, color=HRT)
er = base + len(core) + 1
ws.merge_cells(start_row=er, start_column=1, end_row=er, end_column=15)
vc = R["veto_breakdown"]
ws.cell(er, 1, f"其余 {R['n_veto']} 只被否决：" + "、".join(f"{k} {v}" for k, v in vc.items())
        + "（有重叠）。注意：'形态=深坑反弹'的标的虽通过门槛，但 250 日仍是负收益，"
          "属'跌久刚反弹'，与'缓涨'性质不同，需人工判断。").font = Font(size=9, color=HRT)
ws.cell(er, 1).alignment = LEFT
ws.row_dimensions[er].height = 40
er2 = er + 1
ws.merge_cells(start_row=er2, start_column=1, end_row=er2, end_column=15)
ws.cell(er2, 1, f"数据快照：末交易日 {snap.get('last_date_max', '?')} ｜ data_fp {snap.get('data_fp', '')[:16]}… ｜ {snap.get('n_files', 0)} 个文件"
        " ｜ ⚠「超沪深300」与「120日」是同一信息的两种表述（指数收益对所有股票是同一常数，两者排序完全相同），不是两条独立证据。"
        ).font = Font(size=9, color="FF595959")
ws.cell(er2, 1).alignment = LEFT
ws.row_dimensions[er2].height = 30

# ---------------- Sheet2 全池排名 ----------------
ws2 = wb.create_sheet("全池排名")
ws2.sheet_view.showGridLines = False
trow(ws2, "全池 59 只（绝对便宜度 · 五项门槛 · 否决原因）", 13)
h2 = ["建议", "排名", "代码", "名称", "体系", "现价", "绝对便宜度", "距200线", "120日",
      "250日", "超沪深300", "250回撤", "否决原因"]
w2 = [7, 5, 8, 10, 6, 9, 11, 9, 7, 8, 11, 9, 30]
hrow(ws2, 2, h2, w2)
f2 = 3
l2 = f2 + len(ranked) - 1
for i, r in enumerate(ranked):
    rr = f2 + i
    vals = [r["suggest"], r.get("rank") or "—", r["code"], r["name"], r["system"], r["price"],
            r["cheap"], r.get("dist_sma200"), r.get("ret_120"), r.get("ret_250"),
            r.get("excess300"), r.get("max_dd_250"), r.get("veto") or ""]
    fs = [None, None, None, None, None, "#,##0.00", "0.00", "+0.0%;-0.0%", "+0%;-0%",
          "+0%;-0%", "+0%;-0%", "0%", None]
    for j, (v, f) in enumerate(zip(vals, fs), start=1):
        c = ws2.cell(rr, j, v)
        c.border = B
        c.font = N
        if f:
            c.number_format = f
        c.alignment = CEN if j in (1, 2, 3, 5) else (LEFT if j in (4, 13) else RIGHT)
ws2.freeze_panes = "F3"
ws2.auto_filter.ref = f"A2:M{l2}"
ws2.conditional_formatting.add(f"A{f2}:M{l2}", FormulaRule(formula=[f'$A{f2}="核心"'], fill=GR, font=Font(color=HGT, bold=True)))
ws2.conditional_formatting.add(f"A{f2}:M{l2}", FormulaRule(formula=[f'$A{f2}="否决"'], fill=RF))

# ---------------- Sheet3 K线图 ----------------
ws3 = wb.create_sheet("K线图")
ws3.sheet_view.showGridLines = False
trow(ws3, "核心标的 K 线（近180日，红涨绿跌，蓝=SMA200）", 8)
ws3.merge_cells("A2:H2")
ws3.cell(2, 1, "图为价格形态目检。基本面破裂/黑天鹅/消息驱动无数据源，只能靠价格代理（放量已标红），需人工核公告财报。").font = Font(size=9, italic=True, color="FF595959")
ws3.cell(2, 1).alignment = LEFT
try:
    from openpyxl.drawing.image import Image as XImg
    row = 3
    for k, r in enumerate(core):
        p = CH / f"{r['code']}_{r['name']}.png"
        if not p.exists():
            continue
        im = XImg(str(p))
        im.width = 430
        im.height = 240
        ws3.add_image(im, ("A" if k % 2 == 0 else "F") + str(row))
        if k % 2 == 1:
            row += 13
except Exception as e:
    ws3.cell(3, 1, f"图片失败:{e}")

# ---------------- Sheet4 模型自查 ----------------
ws4 = wb.create_sheet("模型自查")
ws4.sheet_view.showGridLines = False
trow(ws4, "模型自查（带证据）· 含本轮新查出的质量问题", 4)
hrow(ws4, 2, ["检查项", "结果", "证据", "性质"], [18, 9, 62, 10])
aud = [
    ("数据完整性", "PASS", "ETF10/A股59 无异常；EEM 2008-10-13 +22.77% 经核为真实危机日", "原有"),
    ("前视偏差", "PASS", f"close vs 次日开盘 最大NAV差 {S['lookahead']['max_nav_diff']:.2f}，次日开盘无前视", "原有"),
    ("公式正确", "PASS", "茅台/格力 12-1动量手算=脚本 diff=0", "原有"),
    ("排除结构", "PASS", "排除用硬否决(AND)，加权分只用于排序；艾力斯(便宜1.0/趋势0.29)已否", "v2"),
    ("池内截面", "PASS", "排除改用绝对趋势(上200日线)+跑赢沪深300，不再用池内百分位", "v3"),
    ("便宜度强制分布", "FIXED", "品牌/金融各3只，组内百分位必为{0.333,0.667,1.0}，最小0.333>0.30 → '不够便宜'永不触发；改绝对刻度", "v4 新查"),
    ("指数数据静默缺失", "FIXED", "data_ashare/ 0文件入git；load_idx() 缺失时曾静默返回空→excess300=NaN→全部被'跑输大盘'否掉且不报错；改抛 FileNotFoundError", "v4 新查"),
    ("流动性无门槛", "FIXED", "原无成交额检查；核心中招商公路 20日均成交额仅1.7亿（最薄）；已加 ≥0.5亿 门槛并列示", "v4 新查"),
    ("快照不可追溯", "FIXED", "前复权每天重算，重抓会改历史价；原 json 无快照标识；已写入 data_fp + 末交易日", "v4 新查"),
    ("pandas concat 隐式排序", "FIXED", "依赖 pandas4 将改默认值的隐式行为（当前3.1.1实测 diff=0，未算错）；已显式 sort=True", "v4 新查·隐患"),
    ("因子共线", "NOTE", "corr(超300, 120日)=1.000：指数收益是常数，两者排序完全相同，非独立证据；排序分未重复计入", "v4 新查"),
]
for i, (k, v, ev, tag) in enumerate(aud):
    rr = 3 + i
    ws4.cell(rr, 1, k).alignment = CEN
    ws4.cell(rr, 2, v).alignment = CEN
    ws4.cell(rr, 3, ev).alignment = LEFT
    ws4.cell(rr, 4, tag).alignment = CEN
    for j in range(1, 5):
        ws4.cell(rr, j).border = B
        ws4.cell(rr, j).font = N
    col = GR if v == "PASS" else (OF if v == "NOTE" else LF)
    ws4.cell(rr, 2).fill = col
    ws4.cell(rr, 2).font = Font(bold=True, color=HGT if v == "PASS" else ("FF7F6000" if v == "NOTE" else HT))
    ws4.row_dimensions[rr].height = 30

# ---------------- Sheet5 ETF模型对比 ----------------
ws5 = wb.create_sheet("ETF模型对比")
ws5.sheet_view.showGridLines = False
trow(ws5, "ETF 模型 vs 其他开源模型（OOS 2016-02→今）", 7)
hrow(ws5, 2, ["策略", "CAGR", "年化波动", "Sharpe", "MDD", "Calmar", "命中(标的)"], [26, 11, 11, 11, 11, 11, 12])
oos = C["windows"]["OOS  2016-02→今"]
for i, it in enumerate(oos):
    rr = 3 + i
    ws5.cell(rr, 1, it["strategy"]).alignment = LEFT
    for j, k, f in [(2, "cagr", "0.00%"), (3, "ann_vol", "0.00%"), (4, "sharpe", "0.000"),
                    (5, "mdd", "0.00%"), (6, "calmar", "0.000"), (7, "hit_name", "0.0%")]:
        c = ws5.cell(rr, j, it[k])
        c.number_format = f
        c.alignment = RIGHT
    for j in range(1, 8):
        ws5.cell(rr, j).border = B
        ws5.cell(rr, j).font = N
    if it["strategy"].startswith("ours"):
        ws5.cell(rr, 1).font = Font(bold=True, color=HT)

wb.properties.title = "A股黄金坑筛选报表v4"
wb.save(str(XLSX))
print("saved", XLSX, "core", len(core))
