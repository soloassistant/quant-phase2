"""成交价护栏：缺开盘价时**直接失败**，不再拿当日收盘价顶替。

为什么单独抽出来：评审指出旧实现在 `open` 缺失时会 `op.where(..., close_ff.loc[d])`，
等于「开盘那一刻就看到了当天收盘」，是明确的前视偏差，而且它**静默**发生 ——
回测数字看着更漂亮，却没人知道这一天用的是收盘价。

本模块不 import pandas，方便用 unittest 直接覆盖。
"""
from __future__ import annotations

__all__ = ["DataGapError", "check_open_prices"]


class DataGapError(RuntimeError):
    """行情缺口：该用的价格没有，且不允许用别的东西替代。"""


def check_open_prices(date_label, opens: dict) -> dict:
    """校验某日全部标的的开盘价都可用。

    参数
    ----
    date_label : 可打印的日期（str / date 均可）
    opens      : {ticker: 价格}，缺失用 None，pandas 的 NaN 也可以（NaN != NaN）

    返回校验后的同一份 dict（值转 float）。
    任一标的缺失或非正 → 抛 DataGapError，并把缺失清单列出来。
    """
    bad = []
    clean = {}
    for ticker, value in opens.items():
        try:
            price = float(value)
        except (TypeError, ValueError):
            bad.append(f"{ticker}(非数值)")
            continue
        # NaN 不等于自身：这是不依赖 pandas 的判空方式
        if price != price or price <= 0:
            bad.append(f"{ticker}({'NaN' if price != price else price})")
            continue
        clean[ticker] = price
    if bad:
        raise DataGapError(
            f"[行情缺口] {date_label} 开盘价不可用：{', '.join(bad)}。"
            "拒绝改用当日收盘价成交（那等于提前知道收盘价，属于前视偏差）。"
            "请补齐行情或把该日排除在样本外。"
        )
    return clean
