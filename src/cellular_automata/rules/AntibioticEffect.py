import numpy as np

from cellular_automata.Geometry import Geometry
from cellular_automata.registry import register_rule
from cellular_automata.Rule import Rule
from cellular_automata.State import State

# the time step base_prob was calibrated for (the bioreactor Diffusion dt)
DEFAULT_REFERENCE_TIME_STEP = 0.01


@register_rule
class AntibioticEffect(Rule):
    """Kill bacteria with a probability that biofilm and the wall protect against.

    While the dose is active (steps `inoculation_start` to `inoculation_end`), a
    cell of species i dies with probability

        base_prob * (time_step / reference_time_step) * (1 + effectiveness[i])
        / (1 + protecting_neighbours / (neighbourhood size + 1))

    where protecting_neighbours counts biofilm neighbours plus, in the bottom
    z-layer, the wall cells the neighbourhood reaches past the edge.

    args:
    - bacteria_keys, effectiveness - one effectiveness per key, each >= -1
      (-1: the antibiotic has no effect on that species)
    - biofilm_key - key whose cells protect their neighbours
    - base_prob - death probability per `reference_time_step` with no protection
    - reference_time_step - the time step base_prob is calibrated for
    - inoculation_start, inoculation_end - dosing window in steps (end
      exclusive; None means until the end of the run)
    """

    def __init__(
        self,
        geometry: Geometry,
        time_step: float,
        bacteria_keys: list[str],
        biofilm_key: str,
        effectiveness: list[float],
        base_prob: float = 0.01,
        inoculation_start: int = 0,
        inoculation_end: int | None = None,
        reference_time_step: float = DEFAULT_REFERENCE_TIME_STEP,
    ):
        super().__init__(geometry, time_step)
        if len(bacteria_keys) != len(effectiveness):
            raise ValueError(
                f"AntibioticEffect: {len(bacteria_keys)} bacteria_keys but "
                f"{len(effectiveness)} effectiveness values; give one per key"
            )
        if any(value < -1 for value in effectiveness):
            raise ValueError(
                f"AntibioticEffect: effectiveness values must be >= -1, got {effectiveness}"
            )
        self.bacteria_keys = bacteria_keys
        self.biofilm_key = biofilm_key
        self.effectiveness = effectiveness
        self.base_prob = base_prob
        self.scale = time_step / reference_time_step
        self.start = inoculation_start
        self.end = inoculation_end
        # the wall sits below index 0 of the last axis, z in 3D (ARCH-2 will make it
        # configurable)
        self.wall_neighbours = sum(1 for offset in geometry._offsets if offset[-1] < 0)
        self.neighbourhood_size = len(geometry._offsets) + 1

    def death_probability(self, biofilm: np.ndarray) -> np.ndarray:
        """Per-cell death probability for effectiveness 0, given the biofilm grid."""
        protecting = self.neighborhood_count(biofilm).astype(np.float64)
        protecting[..., 0] += self.wall_neighbours
        return self.base_prob * self.scale / (1 + protecting / self.neighbourhood_size)

    def apply_state(self, state: State):
        step_no = state.sim.step_no
        end = state.sim.max_steps if self.end is None else self.end
        if step_no < self.start or step_no >= end:
            return state
        death_prob = self.death_probability(state[self.biofilm_key])
        for key, effectiveness in zip(self.bacteria_keys, self.effectiveness, strict=True):
            bacteria = state[key].copy()
            bacteria_mask = self._cell_mask(bacteria)
            rolls = self.rng.random(bacteria_mask.shape)
            died = bacteria_mask & (rolls < death_prob * (1 + effectiveness))
            bacteria[died] = 0
            state[key] = bacteria
        return state
