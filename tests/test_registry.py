"""Rules and neighbourhoods are looked up in explicit registries (ARCH-1)."""

import importlib
import inspect
import json
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import cellular_automata.rules as rules_package
from cellular_automata.config.Parser import parse_json
from cellular_automata.registry import (
    NEIGHBOURHOODS,
    RULES,
    Registry,
    import_submodules,
    register_rule,
)
from cellular_automata.Rule import Rule


def test_unknown_name_lists_known_names():
    registry = Registry("widget")
    registry.register(dict, name="Known")

    with pytest.raises(ValueError, match=r"Unknown widget 'Nope'\. Known: \['Known'\]"):
        registry.get("Nope")


def test_registering_a_name_twice_is_rejected():
    registry = Registry("widget")
    registry.register(dict, name="Thing")

    with pytest.raises(ValueError, match="already registered"):
        registry.register(list, name="Thing")


def test_register_defaults_to_the_class_name_and_returns_the_class():
    registry = Registry("widget")

    class Gadget:
        pass

    assert registry.register(Gadget) is Gadget
    assert registry.get("Gadget") is Gadget


def _rule_classes_in_package():
    found = {}
    for module_info in pkgutil.iter_modules(rules_package.__path__):
        module = importlib.import_module(f"{rules_package.__name__}.{module_info.name}")
        for name, obj in vars(module).items():
            if inspect.isclass(obj) and issubclass(obj, Rule) and obj.__module__ == module.__name__:
                found[name] = obj
    return found


def test_every_built_in_rule_is_registered():
    defined = _rule_classes_in_package()

    assert defined
    assert {name: RULES.get(name) for name in defined} == defined


def test_import_submodules_imports_every_module_of_a_package(tmp_path: Path, monkeypatch):
    package = tmp_path / "fresh_rules"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "first.py").write_text("LOADED = True\n")
    (package / "second.py").write_text("LOADED = True\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    import_submodules(importlib.import_module("fresh_rules"))

    assert {"fresh_rules.first", "fresh_rules.second"} <= set(sys.modules)


def test_importing_the_rules_package_imports_every_rule_module():
    code = (
        "import sys, pkgutil, cellular_automata.rules as r; "
        "print(all(f'{r.__name__}.{m.name}' in sys.modules "
        "for m in pkgutil.iter_modules(r.__path__)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "True"


def test_built_in_neighbourhoods_are_registered():
    assert NEIGHBOURHOODS.names() == ["MooreNeighbourhood", "VonNeumannNeighbourhood"]


def _config(tmp_path: Path, rule_name: str) -> Path:
    config = {
        "sim": {"size": [3, 3, 3], "periodicity": "", "max_steps": 1, "render": False,
                "seed": 1},
        "state_keys": [{"name": "alive"}],
        "rules": [{"name": rule_name, "args": {"time_step": 1}}],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


@pytest.mark.parametrize("name", ["NoSuchRule", "Path", "parse_json", "State", "Rule"])
def test_config_cannot_name_anything_but_a_registered_rule(tmp_path: Path, name):
    with pytest.raises(ValueError, match=f"Unknown rule '{name}'"):
        parse_json(_config(tmp_path, name))


def test_a_newly_registered_rule_is_usable_from_a_config(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(RULES, "_items", dict(RULES._items))

    @register_rule
    class KeepState(Rule):
        def __init__(self, geometry, time_step):
            super().__init__(geometry, time_step)
            self.required_keys.append("alive")

        def apply_state(self, state):
            return state

    sim, _ = parse_json(_config(tmp_path, "KeepState"))

    assert isinstance(sim.rules[0], KeepState)
