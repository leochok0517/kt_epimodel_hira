"""직렬화 안전 저장 유틸. numpy/jax 타입 → Python 기본 타입 강제 변환.
반복된 저장 crash (int64, mlflow 키 등) 영구 차단."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def to_native(obj):
    """numpy/jax 타입을 재귀적으로 Python 기본 타입으로 변환."""
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if hasattr(obj, "item") and getattr(obj, "ndim", None) == 0:
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            return str(obj)
    return obj


def safe_json_dump(data, path):
    """무조건 native 변환 후 저장. default=str로 최후 방어."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(to_native(data), f, indent=2, ensure_ascii=False, default=str)


def write_flag(path, msg: str = "done") -> None:
    """flag는 가장 단순하게 (의존성 0)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(msg))
