"""实盘参数模块测试：默认值、校验、指纹稳定性。"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import live_params as lp  # noqa: E402


class DefaultTest(unittest.TestCase):
    def test_default_is_valid(self):
        p = lp.load_params()
        self.assertEqual(p["k"], 3)
        self.assertEqual(p["mom_days"], 252)
        self.assertEqual(p["weight"], "equal")
        self.assertFalse(p["use_filter"])

    def test_sha_is_stable_and_order_independent(self):
        a = dict(lp.DEFAULT_PARAMS)
        b = dict(reversed(list(lp.DEFAULT_PARAMS.items())))
        self.assertEqual(lp.params_sha256(a), lp.params_sha256(b))
        self.assertEqual(len(lp.params_sha256(a)), 64)

    def test_sha_changes_with_params(self):
        a = lp.params_sha256(lp.DEFAULT_PARAMS)
        changed = dict(lp.DEFAULT_PARAMS, k=4)
        self.assertNotEqual(a, lp.params_sha256(changed))

    def test_describe_is_human_readable(self):
        text = lp.describe(lp.DEFAULT_PARAMS)
        self.assertIn("K=3", text)
        self.assertIn("10%", text)


class ValidationTest(unittest.TestCase):
    def test_unknown_key_rejected(self):
        # 拼错的键会被静默忽略，所以这里必须拒绝
        with self.assertRaises(lp.ParamsError):
            lp.validate_params(dict(lp.DEFAULT_PARAMS, momdays=252))

    def test_missing_key_rejected(self):
        p = dict(lp.DEFAULT_PARAMS)
        p.pop("target_vol")
        with self.assertRaises(lp.ParamsError):
            lp.validate_params(p)

    def test_k_out_of_range(self):
        with self.assertRaises(lp.ParamsError):
            lp.validate_params(dict(lp.DEFAULT_PARAMS, k=99))

    def test_bool_is_not_int(self):
        with self.assertRaises(lp.ParamsError):
            lp.validate_params(dict(lp.DEFAULT_PARAMS, k=True))

    def test_unsupported_weight(self):
        with self.assertRaises(lp.ParamsError):
            lp.validate_params(dict(lp.DEFAULT_PARAMS, weight="invvol"))

    def test_target_vol_string_rejected(self):
        with self.assertRaises(lp.ParamsError):
            lp.validate_params(dict(lp.DEFAULT_PARAMS, target_vol="10%"))


class FileTest(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "params.json"
            f.write_text(json.dumps(dict(lp.DEFAULT_PARAMS, k=4)), encoding="utf-8")
            p = lp.load_params(f)
            self.assertEqual(p["k"], 4)

    def test_missing_file(self):
        with self.assertRaises(lp.ParamsError):
            lp.load_params("/definitely/not/here.json")

    def test_bad_json(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "bad.json"
            f.write_text("{not json", encoding="utf-8")
            with self.assertRaises(lp.ParamsError):
                lp.load_params(f)


if __name__ == "__main__":
    unittest.main(verbosity=2)
