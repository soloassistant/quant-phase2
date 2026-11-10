"""NYSE 交易日历（仅覆盖判断「本月还有没有后续交易日」所需的粒度）。

为什么不用第三方库：这个仓库要能在干净 clone 下跑测试，而
`pandas_market_calendars` 会再拉一层依赖。这里只需要「某天是不是交易日」
和「本月本月末之前还有没有交易日」，用 stdlib 实现足够，并且可离线测试。

**已知边界（写清楚，别让调用方误以为这是完整交易日历）**：
· 只覆盖 NYSE 的**全天休市**（常规节假日 + 耶稣受难日），不含提前收盘（半日市）；
· 不覆盖临时休市（国葬、飓风等）。这类事件极少，但一旦发生，本模块会把
  那天当成交易日 —— 对「月末判定」的影响是**保守**的（会晚一天确认月末），
  不会提前用上未发生的数据。
"""
from __future__ import annotations

from datetime import date, timedelta

__all__ = [
    "is_trading_day",
    "last_trading_day_of_month",
    "has_later_trading_day_in_month",
]


def _easter(year: int) -> date:
    """Anonymous Gregorian algorithm（复活节星期天）。"""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _observed(d: date) -> date:
    """美国惯例：周六休市提前到周五，周日休市顺延到周一。"""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """当月第 n 个星期 weekday（weekday: 0=周一）。"""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def nyse_holidays(year: int) -> set[date]:
    """该年 NYSE 全天休市的日期集合。"""
    days: set[date] = set()
    days.add(_observed(date(year, 1, 1)))                 # New Year's Day
    days.add(_nth_weekday(year, 1, 0, 3))                 # MLK Day
    days.add(_nth_weekday(year, 2, 0, 3))                 # Washington's Birthday
    days.add(_easter(year) - timedelta(days=2))           # Good Friday
    days.add(_last_weekday(year, 5, 0))                   # Memorial Day
    if year >= 2022:
        days.add(_observed(date(year, 6, 19)))            # Juneteenth（2022 起）
    days.add(_observed(date(year, 7, 4)))                 # Independence Day
    days.add(_nth_weekday(year, 9, 0, 1))                 # Labor Day
    days.add(_nth_weekday(year, 11, 3, 4))                # Thanksgiving
    days.add(_observed(date(year, 12, 25)))               # Christmas
    return days


def is_trading_day(d: date) -> bool:
    if d.weekday() >= 5:
        return False
    return d not in nyse_holidays(d.year)


def last_trading_day_of_month(year: int, month: int) -> date:
    nxt = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    d = nxt - timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def has_later_trading_day_in_month(d: date) -> bool:
    """d 之后（不含 d）本月是否还有交易日。

    返回 False 就表示「d 是本月的最后一个交易日」，可以收盘后出信号。
    """
    last = last_trading_day_of_month(d.year, d.month)
    return d < last
