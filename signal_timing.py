"""信号时点：从交易日序列里挑出「该出信号的那一天」。

抽出来的原因：这段逻辑原先埋在 `run_signal.py` 里，模块级 import pandas，
导致这条最关键的判定（评审点名的月末 bug）没法被单测覆盖。
这里只依赖 `market_calendar`，入参是任何带 `.month` / `.date()` 的序列
（pandas 的 DatetimeIndex 与 stdlib 的 date 列表都能直接喂进来）。
"""
from __future__ import annotations

from market_calendar import has_later_trading_day_in_month

__all__ = ["month_end_indices", "latest_signal"]


def _as_date(x):
    """pandas.Timestamp 有 .date()，stdlib date 本身就是日期。"""
    return x.date() if hasattr(x, "date") else x


def month_end_indices(cal):
    """已观察到的月末下标（有下一交易日且月份切换）。"""
    return [i for i in range(len(cal) - 1) if cal[i].month != cal[i + 1].month]


def latest_signal(cal):
    """返回 (信号下标, 是否有次日成交数据)。

    数据末端若正好是本月最后一个交易日，按最后一行出信号，成交日待次月数据；
    否则取最近一个「有下一交易日」的月末。

    ⚠️ 2026-11-10 修：原来用 `pd.offsets.BDay(1)` 判「下一天是否换月」，
    它只跳周末、**不认交易所休市日**。典型踩坑是「当月最后一个日历日恰好休市」：
    2021-05-31 是阵亡将士纪念日，5 月最后交易日其实是 5/28，BDay 却把 5/31
    当成下一天、月份没变，于是程序退回去用 4/30 的信号 —— 会发出上个月的持仓。
    现在改用 NYSE 日历判断「本月还有没有后续交易日」。
    """
    last = cal[-1]
    if not has_later_trading_day_in_month(_as_date(last)):
        return len(cal) - 1, False
    me = month_end_indices(cal)
    if not me:
        raise SystemExit(f"[区间不足] {_as_date(cal[0])}→{_as_date(last)} 内没有可识别的月末")
    return me[-1], True
