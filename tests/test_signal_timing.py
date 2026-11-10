"""信号时点端到端测试（不需要 pandas）。

用 stdlib 的 date 列表构造交易日序列——`latest_signal` 只用到 `.month` 与 `.date()`，
所以可以直接覆盖「月末恰逢休市」这条真实路径。
"""
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_calendar import is_trading_day  # noqa: E402
from signal_timing import latest_signal, month_end_indices  # noqa: E402


def sessions(start: date, end: date):
    out, d = [], start
    while d <= end:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


class LatestSignalTest(unittest.TestCase):
    def test_month_end_on_holiday_uses_last_trading_day(self):
        """2021-05-31 休市（阵亡将士纪念日）→ 数据到 5/28 就应出 5 月信号。

        旧实现：BDay(1) 从 5/28 得到 5/31、月份没变 → 判为月中 → 退回 4/30。
        """
        cal = sessions(date(2021, 4, 1), date(2021, 5, 28))
        idx, exec_available = latest_signal(cal)
        self.assertEqual(cal[idx], date(2021, 5, 28))
        self.assertFalse(exec_available)

    def test_mid_month_falls_back_to_previous_month_end(self):
        cal = sessions(date(2021, 4, 1), date(2021, 5, 27))
        idx, exec_available = latest_signal(cal)
        self.assertEqual(cal[idx], date(2021, 4, 30))
        self.assertTrue(exec_available)

    def test_normal_month_end(self):
        cal = sessions(date(2025, 5, 1), date(2025, 6, 30))
        idx, exec_available = latest_signal(cal)
        self.assertEqual(cal[idx], date(2025, 6, 30))
        self.assertFalse(exec_available)

    def test_month_ending_on_weekend(self):
        # 2025-05-31 周六；数据到 5/30（周五）即 5 月最后交易日
        cal = sessions(date(2025, 5, 1), date(2025, 5, 30))
        idx, exec_available = latest_signal(cal)
        self.assertEqual(cal[idx], date(2025, 5, 30))
        self.assertFalse(exec_available)

    def test_no_month_end_in_range(self):
        cal = sessions(date(2025, 6, 2), date(2025, 6, 20))
        with self.assertRaises(SystemExit):
            latest_signal(cal)

    def test_month_end_indices(self):
        cal = sessions(date(2021, 4, 1), date(2021, 5, 28))
        idx = month_end_indices(cal)
        self.assertEqual([cal[i] for i in idx], [date(2021, 4, 30)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
