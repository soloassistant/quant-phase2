"""NYSE 日历的离线测试。

重点用例来自评审：2025-07-03 是 7 月最后一个交易日（7/4 独立日休市），
旧实现用 BDay(1) 判月份切换，会把这天当成「月中」，退回去发 6 月信号。
"""
import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_calendar import (  # noqa: E402
    has_later_trading_day_in_month,
    is_trading_day,
    last_trading_day_of_month,
)


class TradingDayTest(unittest.TestCase):
    def test_weekend_is_closed(self):
        self.assertFalse(is_trading_day(date(2025, 7, 5)))   # 周六
        self.assertFalse(is_trading_day(date(2025, 7, 6)))   # 周日

    def test_regular_weekday_is_open(self):
        self.assertTrue(is_trading_day(date(2025, 7, 2)))

    def test_new_year(self):
        self.assertFalse(is_trading_day(date(2025, 1, 1)))

    def test_good_friday(self):
        self.assertFalse(is_trading_day(date(2024, 3, 29)))

    def test_july_fourth_observed(self):
        self.assertFalse(is_trading_day(date(2025, 7, 4)))

    def test_christmas_observed_on_friday(self):
        # 2021-12-25 是周六 → 提前到 12-24（周五）休市
        self.assertFalse(is_trading_day(date(2021, 12, 24)))

    def test_juneteenth_from_2022(self):
        self.assertFalse(is_trading_day(date(2022, 6, 20)))
        self.assertTrue(is_trading_day(date(2021, 6, 18)))   # 2021 尚未休市

    def test_thanksgiving(self):
        self.assertFalse(is_trading_day(date(2025, 11, 27)))
        self.assertTrue(is_trading_day(date(2025, 11, 28)))


class MonthEndTest(unittest.TestCase):
    def test_memorial_day_is_month_end_2021_05(self):
        """真正的踩坑用例：2021-05-31 是 5 月最后一个日历日，且是阵亡将士纪念日。

        旧实现用 `pd.offsets.BDay(1)` 判月份切换：从 5/28（周五）加一个工作日
        得 5/31（周一，休市），月份没变 → 判定「月中」，退回去发 4 月信号。
        （2015–2026 里这种「月末最后一天恰逢休市」只出现在 2021-05；
        评审举的 2025-07-03 不是月末 —— 7/7 起还有交易日，那个例子不成立。）
        """
        self.assertEqual(last_trading_day_of_month(2021, 5), date(2021, 5, 28))
        self.assertFalse(has_later_trading_day_in_month(date(2021, 5, 28)))

    def test_july_3_2025_is_not_month_end(self):
        """锁定正确认知：7/3 只是「7/4 休市前的最后一个交易日」，不是月末。"""
        self.assertTrue(has_later_trading_day_in_month(date(2025, 7, 3)))
        self.assertEqual(last_trading_day_of_month(2025, 7), date(2025, 7, 31))

    def test_day_before_holiday_month_end_still_has_later(self):
        self.assertTrue(has_later_trading_day_in_month(date(2025, 7, 2)))

    def test_normal_month_end(self):
        self.assertEqual(last_trading_day_of_month(2025, 6), date(2025, 6, 30))
        self.assertFalse(has_later_trading_day_in_month(date(2025, 6, 30)))
        self.assertTrue(has_later_trading_day_in_month(date(2025, 6, 27)))

    def test_year_end(self):
        self.assertEqual(last_trading_day_of_month(2025, 12), date(2025, 12, 31))
        self.assertFalse(has_later_trading_day_in_month(date(2025, 12, 31)))

    def test_month_ending_on_weekend(self):
        # 2025-05-31 是周六，5 月最后交易日应是 5/30（周五）
        self.assertEqual(last_trading_day_of_month(2025, 5), date(2025, 5, 30))


if __name__ == "__main__":
    unittest.main(verbosity=2)
