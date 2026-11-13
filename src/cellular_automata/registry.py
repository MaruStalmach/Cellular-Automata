"""Named registries for the classes a config can refer to.

A config names rules and neighbourhoods by string; the parser looks them up
here, so only registered classes can be built from a config. Register a new
rule with `@register_rule` on its class; putting the module in
`cellular_automata/rules/` is enough for it to be imported.
"""

import importlib
import pkgutil
from types import ModuleType
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from cellular_automata.Rule import Rule

T = TypeVar("T")


def _same_definition(a: object, b: object) -> bool:
    """True if `a` and `b` come from the same source definition, e.g. after a reload."""
    return (
        getattr(a, "__module__", None) == getattr(b, "__module__", None)
        and getattr(a, "__qualname__", None) == getattr(b, "__qualname__", None)
    )


class Registry(Generic[T]):
    """Maps names to objects of one kind, e.g. rule classes."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, obj: T, name: str | None = None) -> T:
        """Register `obj` under `name` (default: its `__name__`) and return it.

        Re-registering the same definition (a reloaded module) replaces the
        old entry; a different object under a taken name is an error.
        """
        name = name or obj.__name__
        existing = self._items.get(name)
        if existing is not None and not _same_definition(existing, obj):
            raise ValueError(f"{self.kind} '{name}' is already registered")
        self._items[name] = obj
        return obj

    def get(self, name: str) -> T:
        if name not in self._items:
            raise ValueError(f"Unknown {self.kind} '{name}'. Known: {self.names()}")
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)


RULES: Registry["type[Rule]"] = Registry("rule")
NEIGHBOURHOODS: Registry[type] = Registry("neighbourhood")

register_rule = RULES.register
register_neighbourhood = NEIGHBOURHOODS.register


def import_submodules(package: ModuleType) -> None:
    """Import every module in `package`, so the classes in them register themselves."""
    for module_info in pkgutil.iter_modules(package.__path__):
        importlib.import_module(f"{package.__name__}.{module_info.name}")
