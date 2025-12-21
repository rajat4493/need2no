from __future__ import annotations

from typing import Callable, Dict, List

from n2n.models import DetectionResult, TextSpan

PrimitiveFunc = Callable[[List[TextSpan]], List[DetectionResult]]

_REGISTRY: Dict[str, PrimitiveFunc] = {}


def register_primitive(name: str) -> Callable[[PrimitiveFunc], PrimitiveFunc]:
    def decorator(func: PrimitiveFunc) -> PrimitiveFunc:
        _REGISTRY[name] = func
        return func

    return decorator


def get_primitive(name: str) -> PrimitiveFunc:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown primitive: {name}") from exc


def list_primitives() -> Dict[str, PrimitiveFunc]:
    return dict(_REGISTRY)


__all__ = ["PrimitiveFunc", "register_primitive", "get_primitive", "list_primitives"]
