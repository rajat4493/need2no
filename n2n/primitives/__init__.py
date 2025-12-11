from typing import Callable, Dict

PrimitiveFunc = Callable[..., object]
PRIMITIVES: Dict[str, PrimitiveFunc] = {}


def register_primitive(name: str) -> Callable[[PrimitiveFunc], PrimitiveFunc]:
    """
    Decorator used by primitive modules to self-register detection helpers.
    """

    def decorator(func: PrimitiveFunc) -> PrimitiveFunc:
        PRIMITIVES[name] = func
        return func

    return decorator


__all__ = ["PRIMITIVES", "register_primitive"]
