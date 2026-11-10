"""成交价护栏测试：缺开盘价必须报错，不允许静默用收盘价。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guards import DataGapError, check_open_prices  # noqa: E402


class OpenPriceGuardTest(unittest.TestCase):
    def test_complete_prices_pass(self):
        got = check_open_prices("2025-07-03", {"SPY": 620.5, "QQQ": 551.0})
        self.assertEqual(got["SPY"], 620.5)

    def test_nan_is_rejected(self):
        with self.assertRaises(DataGapError) as ctx:
            check_open_prices("2025-07-03", {"SPY": 620.5, "TLT": float("nan")})
        self.assertIn("TLT", str(ctx.exception))

    def test_none_is_rejected(self):
        with self.assertRaises(DataGapError):
            check_open_prices("2025-07-03", {"SPY": None})

    def test_non_positive_is_rejected(self):
        with self.assertRaises(DataGapError):
            check_open_prices("2025-07-03", {"SPY": 0})
        with self.assertRaises(DataGapError):
            check_open_prices("2025-07-03", {"SPY": -1.0})

    def test_message_explains_why_not_close_fallback(self):
        try:
            check_open_prices("2025-07-03", {"SPY": float("nan")})
        except DataGapError as e:
            self.assertIn("前视", str(e))
        else:
            self.fail("应当抛错")


if __name__ == "__main__":
    unittest.main(verbosity=2)
