"""评审 P4：两套引擎的数值对照（不是源码级检查，是真的跑一遍）。

为什么单独一个文件：CI 主 job 不装第三方依赖，而这里必须 import pandas/numpy，
所以模块级 SkipTest 让主 job 跳过；CI 另有 parity job 装依赖后只跑本文件。
行情用确定性合成数据生成，不需要任何真实 CSV。

实测边界（2026-11-11 / pandas 3.1.1 / numpy 2.5.4 / 520 个交易日 / 10 只标的）：
  * 默认参数：nav / trades / metrics / 信号权重 **逐位一致**；
  * K=2 / 126 日 / 过滤 / inv_vol：nav、trades、权重逐位一致；
  * K=3 / 189 日 / inv_vol：nav 差 1.4e-14、权重差 1.1e-16，来自浮点求和顺序，
    用 1e-12 容差钉住，差异若扩大到 1e-3 就红；
  * 非数值展示字段：signals.selected 的**顺序**不同（标的集合相同），
    signals.est_vol 偶有 1 ULP 差异。这两项不参与「数值一致性」断言。
"""
import sys
import unittest
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:  # CI 主 job / 干净环境：跳过，不算失败
    raise unittest.SkipTest(f"引擎对照需要 pandas 与 numpy: {exc}")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import engine  # noqa: E402
import xsec  # noqa: E402

# 默认参数逐位一致；第二组同样逐位一致；第三组只允许 ULP 级漂移。
CASES = {
    "default": ({}, True),
    "short_momentum_filtered_invvol": (
        {"k": 2, "mom_days": 126, "use_filter": True, "weight": "inv_vol"}, True),
    "medium_momentum_invvol": (
        {"k": 3, "mom_days": 189, "weight": "inv_vol"}, False),
}


def make_data(n=520):
    """10 只标的、无缺失、无零价的确定性日线；不读磁盘。"""
    dates = pd.bdate_range("2020-01-02", periods=n)
    out = {}
    for j, ticker in enumerate(engine.BASE10):
        x = np.arange(n, dtype=float)
        close = 100.0 + 8.0 * np.sin(x / (23.0 + 3 * j)) + 0.02 * x + 2.0 * j
        op = close * (0.997 + 0.001 * np.sin(x / (11.0 + j)))
        out[ticker] = pd.DataFrame({"date": dates, "open": op, "close": close})
    return out


class EngineParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = make_data()
        cls.results = {}
        for name, (kwargs, _) in CASES.items():
            cls.results[name] = (xsec.run(data, **kwargs), engine.run(data, **kwargs))
        cls.data = data

    @staticmethod
    def _assert_columns_equal(testcase, a, b, columns, exact, label):
        for col in columns:
            x = a[col].to_numpy(dtype=float)
            y = b[col].to_numpy(dtype=float)
            testcase.assertEqual(len(x), len(y), f"{label}.{col} 行数不同")
            if exact:
                testcase.assertTrue(np.array_equal(x, y),
                                   f"{label}.{col} 未逐位一致，最大差 {np.abs(x - y).max():.3e}")
            else:
                testcase.assertTrue(np.allclose(x, y, rtol=0, atol=1e-12),
                                   f"{label}.{col} 差异超出 1e-12：{np.abs(x - y).max():.3e}")

    def test_locked_constants_match(self):
        """常量漂移是最常见的口径分叉，逐项钉死。"""
        for name in ("COST_RATE", "TARGET_VOL", "VOL_DAYS", "VOL60_DAYS",
                     "TRADING_DAYS", "SKIP_DAYS"):
            self.assertEqual(getattr(xsec, name), getattr(engine, name), f"{name} 不一致")
        self.assertEqual(xsec.UNIVERSE, engine.BASE10, "标的池不一致")

    def test_nav_columns_for_all_cases(self):
        for name, (_, exact) in CASES.items():
            a, b = self.results[name]
            self.assertTrue(a.nav.index.equals(b.nav.index), f"{name}: 净值日期轴不同")
            self._assert_columns_equal(self, a.nav, b.nav,
                                       ["nav", "cash", "exposure", "n_holdings"],
                                       exact, f"{name}.nav")

    def test_trades_for_all_cases(self):
        for name, (_, exact) in CASES.items():
            a, b = self.results[name]
            self.assertEqual(len(a.trades), len(b.trades), f"{name}: 成交笔数不同")
            numeric = [c for c in a.trades.columns
                       if c in b.trades.columns and pd.api.types.is_numeric_dtype(a.trades[c])]
            self._assert_columns_equal(self, a.trades, b.trades, numeric,
                                       exact, f"{name}.trades")
            for col in ("signal_date", "exec_date"):
                self.assertEqual(a.trades[col].astype(str).tolist(),
                                 b.trades[col].astype(str).tolist(),
                                 f"{name}.trades.{col} 不同")

    def test_default_metrics_bitwise_identical(self):
        a, b = self.results["default"]
        for key in sorted(set(a.metrics) & set(b.metrics)):
            self.assertEqual(a.metrics[key], b.metrics[key], f"metrics.{key} 不一致")
        # engine 额外算了 avg_holdings，xsec 没有；这不是分叉，但要防止有人误删共有指标
        self.assertGreater(len(set(a.metrics) & set(b.metrics)), 10)

    def test_signal_weights_for_all_cases(self):
        for name, (_, exact) in CASES.items():
            a, b = self.results[name]
            wcols = [c for c in a.signals.columns if c.startswith("w_")]
            self.assertTrue(wcols, f"{name}: 没有权重列")
            for col in ("scale", "sum_weight", "n_selected"):
                self.assertIn(col, b.signals.columns, f"{name}: engine 缺 {col}")
            self._assert_columns_equal(self, a.signals, b.signals,
                                       wcols + ["scale", "sum_weight", "n_selected"],
                                       exact, f"{name}.signals")

    def test_selected_is_same_set_but_may_differ_in_order(self):
        """展示顺序不同是已知差异：xsec 按动量降序，engine 按标的列顺序。"""
        for name in CASES:
            a, b = self.results[name]
            for i in range(len(a.signals)):
                sa = a.signals["selected"].iloc[i]
                sb = b.signals["selected"].iloc[i]
                self.assertEqual(set(filter(None, str(sa).split(","))),
                                 set(filter(None, str(sb).split(","))),
                                 f"{name} 第 {i} 个信号的选中标的不同")

    def test_ulp_case_stays_within_tolerance(self):
        """把已知浮点漂移写成显式上界，防止哪天变成真正的口径分叉。"""
        a, b = self.results["medium_momentum_invvol"]
        nav_diff = float(np.abs(a.nav["nav"].to_numpy(float) - b.nav["nav"].to_numpy(float)).max())
        # 只钉上界，不要求「必须有差异」：换平台/依赖后若变成逐位一致，
        # 把 CASES 里这一组的 exact 改成 True 即可，不该让 CI 因此变红。
        self.assertLess(nav_diff, 1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
