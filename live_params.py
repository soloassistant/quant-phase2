"""实盘信号参数：唯一来源 + 可校验指纹。

**为什么要单独一个模块**：评审指出的最大问题是「对外公布的夏普（每年重新选参）
和每月实际下单（写死 K=3 / 12-1 / 等权）不是同一套参数」。
把参数集中到这里之后：
· 出信号时会把这套参数与它的 sha256 写进 metrics.json，谁引用都能核对；
· `--params` 可以指向 run_monthly.py 导出的当期选参 JSON，让实盘用上选出来的参数；
· 不传参时用 DEFAULT_PARAMS，报告里引用它自己那份样本外成绩，不再借用 1.02 夏普；
· **只有带来源标记的导出文件才能引用 walk-forward 成绩**。手写一份 K=4 的 JSON
  同样能通过字段校验，但它没有对应的样本外报告，必须标成 custom 并禁止背书。

本模块**不 import pandas**，因此可在干净环境里直接用 unittest 跑（CI 不装第三方依赖）。
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

__all__ = ["DEFAULT_PARAMS", "OOS_REFERENCE", "EXPORT_SOURCE", "ParamsError",
           "params_sha256", "validate_params", "load_params", "load_params_file",
           "make_export", "params_from_window", "describe"]

# run_monthly.py 导出的选参结果专用标记。手写文件不得自称是这个来源。
EXPORT_SOURCE = "run_monthly.walk_forward"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

# 实盘默认参数：与历史「1.02 夏普」那套**不是**同一组（后者来自每年重新选参）。
DEFAULT_PARAMS = {
    "k": 3,
    "mom_days": 252,
    "use_filter": False,
    "weight": "equal",
    "target_vol": 0.10,
}

# 声明：哪组参数对应哪份样本外成绩。两个路径都指向仓内真实文件，
# 由 tests/test_source_regressions.py 断言存在，避免再出现「指向不存在的报告」。
OOS_REFERENCE = {
    "walk_forward": {
        "params": "每年用前 8 年重新选参，后 1 年检验，间隔 1 个月",
        "report": "experiments/out/baseline.json",
        "note": "README 里的 1.02 夏普来自这组流程，不是 DEFAULT_PARAMS；"
                "仅当 --params 指向 run_monthly.py 导出的当期选参 JSON 时可引用",
    },
    "default_params": {
        "params": "DEFAULT_PARAMS（K=3 / 12-1 / 等权 / 无过滤 / 目标波动 10%）",
        "report": "experiments/out/fixed_default_oos.json",
        "note": "独立样本外成绩（Sharpe 0.969）；复算 python experiments/fixed_default_oos.py，"
                "不得与 walk-forward 的 1.02 混用",
    },
    "custom": {
        "params": "自行编写的参数文件（非 run_monthly.py 导出）",
        "report": None,
        "note": "没有任何对应的样本外报告；禁止引用本仓的 1.02 或 0.969 成绩",
    },
}

_LIMITS = {
    "k": (1, 10),
    "mom_days": (21, 504),
    "target_vol": (0.01, 1.0),
}
# walk-forward 候选网格里就有 inv_vol；不放行的话，导出的当期选参 JSON 会被自己的校验拒掉。
_WEIGHTS = {"equal", "inv_vol"}


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


def make_export(params: dict, run_id: str, result_sha256: str) -> dict:
    """生成 run_monthly.py 落盘的选参文件：参数 + 来源 + 可核对的指纹。

    `params_sha256` 让下游能发现「参数被改过但来源标记还在」；
    `result_sha256` 把这份选参钉到产生它的那次回测上。
    """
    p = validate_params(params)
    return {"params": p, "source": EXPORT_SOURCE, "run_id": str(run_id),
            "params_sha256": params_sha256(p), "result_sha256": str(result_sha256)}


def load_params_file(path: str | Path | None = None) -> tuple[dict, dict]:
    """读取参数文件，返回 (参数, 出处)。出处决定能不能引用样本外成绩。

    - 不传文件        -> ("default", verified=False)
    - 纯参数字典      -> ("custom",  verified=False)  手写文件，无报告
    - 导出格式        -> (EXPORT_SOURCE, verified=True)，且指纹/来源全部校验通过
    """
    if path is None:
        return _validate(DEFAULT_PARAMS), {"source": "default", "verified": False}
    p = Path(path)
    if not p.exists():
        raise ParamsError(f"参数文件不存在: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ParamsError(f"参数文件不是合法 JSON: {p} ({e})") from e
    if not isinstance(raw, dict):
        raise ParamsError(f"参数文件顶层必须是对象: {p}")

    if "params" in raw or "source" in raw:
        if raw.get("source") != EXPORT_SOURCE:
            raise ParamsError(f"参数文件声明的来源 {raw.get('source')!r} 不是 {EXPORT_SOURCE!r}；"
                              f"只有 run_monthly.py 导出的选参文件才能引用 walk-forward 成绩")
        if not isinstance(raw.get("params"), dict):
            raise ParamsError(f"导出文件缺少 params 对象: {p}")
        params = _validate(raw["params"])
        if raw.get("params_sha256") != params_sha256(params):
            raise ParamsError(f"参数指纹不符，文件可能被改过: {p}")
        run_id, result_sha = raw.get("run_id"), raw.get("result_sha256")
        if not isinstance(run_id, str) or not run_id:
            raise ParamsError(f"导出文件缺少 run_id: {p}")
        if not isinstance(result_sha, str) or not _HEX64.match(result_sha):
            raise ParamsError(f"导出文件缺少合法的 result_sha256: {p}")
        return params, {"source": EXPORT_SOURCE, "verified": True,
                        "run_id": run_id, "result_sha256": result_sha}

    return _validate(raw), {"source": "custom", "verified": False}


def load_params(path: str | Path | None = None) -> dict:
    """只取参数，不关心出处；保留给不需要引用样本外成绩的调用方。"""
    return load_params_file(path)[0]


def params_from_window(window: dict, mom_days: dict) -> dict:
    """把 walk_forward 最后一个验证窗的选中参数转成可落盘的字段。

    窗口里 filter/weight 是中文展示值、动量按月计；这里统一转成引擎字段名。
    映射表由调用方（run_monthly.py）传入 pipeline_core.MOM_MONTHS，
    避免在这个不依赖 pandas 的模块里再抄一份候选网格。
    遇到未知展示值直接失败 —— 静默猜一个参数就是评审 P1 那类事故的源头。
    """
    weights = {"等权": "equal", "波动倒数": "inv_vol"}
    filters = {"无": False, "有": True}
    if window["weight"] not in weights or window["filter"] not in filters:
        raise ParamsError(f"窗口展示值无法映射: filter={window['filter']!r} "
                          f"weight={window['weight']!r}")
    months = int(window["mom_months"])
    if months not in mom_days:
        raise ParamsError(f"窗口动量月数 {months} 不在候选网格 {sorted(mom_days)} 中")
    return validate_params({
        "k": int(window["k"]),
        "mom_days": int(mom_days[months]),
        "use_filter": filters[window["filter"]],
        "weight": weights[window["weight"]],
        "target_vol": DEFAULT_PARAMS["target_vol"],
    })
