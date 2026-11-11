"""源码级回归：防止评审里的两个坑被改回来。

单测跑不到依赖 pandas 的引擎，但这两处问题都是**写法**层面的，用代码级断言更稳。
注意要先把注释与字符串剥掉：修复说明里写着「原来用 `pd.offsets.BDay(1)`」，
直接对全文 grep 会被自己的注释命中（第一版测试就踩了这个坑）。
"""
import io
import re
import sys
import tokenize
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def code_only(name: str) -> str:
    """去掉注释与字符串字面量后的源码，用于断言「代码里还有没有这个写法」。"""
    src = (ROOT / name).read_text(encoding="utf-8")
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    except tokenize.TokenError:
        return src
    return " ".join(out)


class NoCloseFallbackTest(unittest.TestCase):
    def test_engines_do_not_fall_back_to_close(self):
        for name in ("xsec.py", "engine.py"):
            self.assertNotRegex(
                code_only(name),
                r"op \. where \( op \. notna",
                f"{name} 又用当日收盘价顶替缺失的开盘价（前视偏差）",
            )

    def test_engines_use_strict_guard(self):
        for name in ("xsec.py", "engine.py"):
            code = code_only(name)
            self.assertIn("check_open_prices", code, f"{name} 没接严格开盘价校验")
            self.assertIn("import guards", code)


class NoBDayMonthEndTest(unittest.TestCase):
    def test_run_signal_does_not_use_bday(self):
        for name in ("run_signal.py", "signal_timing.py"):
            self.assertNotIn("BDay", code_only(name), f"{name} 又用 BDay 判月末（不认交易所休市日）")

    def test_timing_uses_exchange_calendar(self):
        code = code_only("signal_timing.py")
        self.assertIn("has_later_trading_day_in_month", code)
        # run_signal 只引用抽出的模块，不再自己实现一份
        self.assertIn("from signal_timing import latest_signal", code_only("run_signal.py"))


class LiveParamsWiringTest(unittest.TestCase):
    def test_signal_declares_param_provenance(self):
        # 这里用原文而不是 code_only：payload 的键本身是字符串字面量，会被 tokenize 剥掉
        src = (ROOT / "run_signal.py").read_text(encoding="utf-8")
        for token in ('"--params"', "params_sha256", "oos_reference", "live_params.load_params"):
            self.assertIn(token, src, f"run_signal.py 缺少参数出处声明: {token}")

    def test_both_param_sets_have_real_reports(self):
        # 评审 P1 的收口点：两组参数必须各有真实存在的报告，且不能共用一份。
        import live_params as lp

        reports = {k: v["report"] for k, v in lp.OOS_REFERENCE.items()}
        self.assertEqual(set(reports), {"walk_forward", "default_params"})
        for key, rel in reports.items():
            self.assertIsNotNone(rel, f"{key} 缺样本外报告路径")
            self.assertTrue((ROOT / rel).is_file(), f"{key} 指向不存在的报告: {rel}")
        self.assertNotEqual(reports["walk_forward"], reports["default_params"],
                            "两组参数不能共用一份样本外报告")

    def test_monthly_run_exports_selected_params(self):
        # --params 通道必须有真实来源，不能只是文档里的口头约定。
        src = (ROOT / "run_monthly.py").read_text(encoding="utf-8")
        for token in ('"live_params.json"', "params_from_window", "pc.MOM_MONTHS"):
            self.assertIn(token, src, f"run_monthly.py 缺少当期选参导出: {token}")


class StaleScriptRefsTest(unittest.TestCase):
    """README 里提到的脚本必须真实存在（评审提到的「搬到旧目录的残留」）。"""

    def test_readme_scripts_exist(self):
        refs = set(re.findall(r"`([A-Za-z0-9_./-]+\.py)`", (ROOT / "README.md").read_text(encoding="utf-8")))
        missing = [r for r in refs if not (ROOT / r).exists() and not (ROOT / "experiments" / r).exists()]
        self.assertEqual(missing, [], f"README 引用了不存在的脚本: {missing}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
