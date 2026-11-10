"""实盘信号参数：唯一来源 + 可校验指纹。

**为什么要单独一个模块**：评审指出的最大问题是「对外公布的夏普（每年重新选参）
和每月实际下单（写死 K=3 / 12-1 / 等权）不是同一套参数」。
把参数集中到这里之后：
· 出信号时会把这套参数与它的 sha256 写进 metrics.json，谁引用都能核对；
· `--params` 可以指向选参产出的 JSON，让实盘真的用上选出来的参数；
· 不传参时用 DEFAULT_PARAMS，并在报告里标注「该组参数没有对应的样本外报告」。

本模块**不 import pandas**，因此可在干净环境里直接用 unittest 跑。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

__all__ = ["DEFAULT_PARAMS", "OOS_REFERENCE", "ParamsError", "params_sha256",
           "validate_params", "load_params", "describe"]

# 实盘默认参数：与历史「1.02 夏普」那套**不是**同一组（后者来自每年重新选参）。
DEFAULT_PARAMS = {
    "k": 3,
    "mom_days": 252,
    "use_filter": False,
    "weight": "equal",
    "target_vol": 0.10,
}

# 声明：哪组参数对应哪份样本外成绩。宁可不给数字，也不让两组口径混着被引用。
OOS_REFERENCE = {
    "walk_forward": {
        "params": "每年用前 8 年重新选参，后 1 年检验，间隔 1 个月",
        "report": "out_oos/metrics.json",
        "note": "README 里的 1.02 夏普来自这组流程，不是 DEFAULT_PARAMS",
    },
    "default_params": {
        "params": "DEFAULT_PARAMS（K=3 / 12-1 / 等权 / 无过滤 / 目标波动 10%）",
        "report": None,
        "note": "仓库内**没有**这组参数的样本外报告；引用前必须先跑 oos_multi.py --fixed",
    },
}

_LIMITS = {
    "k": (1, 10),
    "mom_days": (21, 504),
    "target_vol": (0.01, 1.0),
}
_WEIGHTS = {"equal"}


class ParamsError(ValueError):
    """参数文件不合法。宁可拒绝运行，也不带着读错的参数出信号。"""


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def params_sha256(params: dict) -> str:
    return hashlib.sha256(_canonical(params).encode("utf-8")).hexdigest()


def describe(params: dict) -> str:
    return (f"K={params['k']} 动量={params['mom_days']}d "
            f"权重={params['weight']} 过滤={'开' if params['use_filter'] else '关'} "
            f"目标波动={params['target_vol']:.0%}")


def _validate(params: dict) -> dict:
    missing = set(DEFAULT_PARAMS) - set(params)
    if missing:
        raise ParamsError(f"参数缺失: {sorted(missing)}")
    unknown = set(params) - set(DEFAULT_PARAMS)
    if unknown:
        raise ParamsError(f"未知参数: {sorted(unknown)}（拼错会被静默忽略，所以这里直接拒绝）")

    out = dict(DEFAULT_PARAMS)
    out.update(params)

    if not isinstance(out["k"], int) or isinstance(out["k"], bool):
        raise ParamsError(f"k 必须是整数，收到 {out['k']!r}")
    lo, hi = _LIMITS["k"]
    if not lo <= out["k"] <= hi:
        raise ParamsError(f"k={out['k']} 超出允许范围 {lo}–{hi}")

    if not isinstance(out["mom_days"], int) or isinstance(out["mom_days"], bool):
        raise ParamsError(f"mom_days 必须是整数，收到 {out['mom_days']!r}")
    lo, hi = _LIMITS["mom_days"]
    if not lo <= out["mom_days"] <= hi:
        raise ParamsError(f"mom_days={out['mom_days']} 超出允许范围 {lo}–{hi}")

    if out["weight"] not in _WEIGHTS:
        raise ParamsError(f"weight 仅支持 {sorted(_WEIGHTS)}，收到 {out['weight']!r}")

    if not isinstance(out["use_filter"], bool):
        raise ParamsError(f"use_filter 必须是布尔值，收到 {out['use_filter']!r}")

    if not isinstance(out["target_vol"], (int, float)) or isinstance(out["target_vol"], bool):
        raise ParamsError(f"target_vol 必须是数字，收到 {out['target_vol']!r}")
    lo, hi = _LIMITS["target_vol"]
    if not lo <= float(out["target_vol"]) <= hi:
        raise ParamsError(f"target_vol={out['target_vol']} 超出允许范围 {lo}–{hi}")

    out["target_vol"] = float(out["target_vol"])
    return out


def validate_params(params: dict) -> dict:
    """对外暴露的校验入口：给调用方在内存里构造参数时用。"""
    if not isinstance(params, dict):
        raise ParamsError(f"参数必须是对象，收到 {type(params).__name__}")
    return _validate(params)


def load_params(path: str | Path | None = None) -> dict:
    """读取参数文件；不传则返回默认参数。文件必须含全部字段。"""
    if path is None:
        return _validate(DEFAULT_PARAMS)
    p = Path(path)
    if not p.exists():
        raise ParamsError(f"参数文件不存在: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ParamsError(f"参数文件不是合法 JSON: {p} ({e})") from e
    if not isinstance(raw, dict):
        raise ParamsError(f"参数文件顶层必须是对象: {p}")
    return _validate(raw)
