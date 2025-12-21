from .registry import get_primitive, list_primitives, register_primitive

# Import built-in primitives so they register on module import
from . import card_pan, uk_bank  # noqa: F401

__all__ = ["get_primitive", "list_primitives", "register_primitive"]
