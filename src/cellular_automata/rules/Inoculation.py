from cellular_automata.Geometry import Geometry
from cellular_automata.registry import register_rule
from cellular_automata.Rule import Rule, StepContext
from cellular_automata.State import State


@register_rule
class Inoculation(Rule):
    """Fill a box of `target_key` with 1 every time the rule runs (a constant source).

    args:
    - target_key - key to fill
    - location - lowest corner of the box, one index per axis
    - size - box size, one length per axis
    - layer - layer to fill when the key has several layers (default 0; ignored
      for single-layer keys)
    """

    def __init__(
        self,
        geometry: Geometry,
        time_step: float,
        target_key: str,
        location: list[int],
        size: list[int],
        layer: int = 0,
    ):
        super().__init__(geometry, time_step)
        if len(location) != geometry.ndim or len(size) != geometry.ndim:
            raise ValueError(
                f"Inoculation: location {location} and size {size} need one value per "
                f"axis ({geometry.ndim})"
            )
        self.target_key = target_key
        self.required_keys.append(target_key)
        self.box = tuple(
            slice(start, start + length) for start, length in zip(location, size, strict=True)
        )
        if layer < 0:
            raise ValueError(f"Inoculation: layer must be >= 0, got {layer}")
        self.layer = layer

    def apply_state(self, state: State, ctx: StepContext | None = None):
        grid = state[self.target_key].copy()
        selection = self.box
        if grid.ndim == self.geometry.ndim + 1:
            if self.layer >= grid.shape[-1]:
                raise ValueError(
                    f"Inoculation: layer {self.layer} but '{self.target_key}' has "
                    f"{grid.shape[-1]} layers"
                )
            selection += (self.layer,)
        grid[selection] = 1
        state[self.target_key] = grid
        return state
