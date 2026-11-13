"""Named registries for the classes a config can refer to.

A config names rules and neighbourhoods by string; the parser looks them up
here, so only registered classes can be built from a config. Register a new
rule with `@register_rule` on its class; putting the module in
`cellular_automata/rules/` is enough for it to be imported.
"""

import importlib
import pkgutil
from types import ModuleType
from typing import TypeVar

T = TypeVar("T")


class Registry:
    """Maps names to objects of one kind, e.g. rule classes."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, object] = {}

    def register(self, obj: T, name: str | None = None) -> T:
        """Register `obj` under `name` (default: its `__name__`) and return it."""
        name = name or obj.__name__
        if name in self._items:
            raise ValueError(f"{self.kind} '{name}' is already registered")
        self._items[name] = obj
        return obj

    def get(self, name: str) -> object:
        if name not in self._items:
            raise ValueError(f"Unknown {self.kind} '{name}'. Known: {self.names()}")
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)


RULES = Registry("rule")
NEIGHBOURHOODS = Registry("neighbourhood")

register_rule = RULES.register
register_neighbourhood = NEIGHBOURHOODS.register


def import_submodules(package: ModuleType) -> None:
    """Import every module in `package`, so the classes in them register themselves."""
    for module_info in pkgutil.iter_modules(package.__path__):
        importlib.import_module(f"{package.__name__}.{module_info.name}")
