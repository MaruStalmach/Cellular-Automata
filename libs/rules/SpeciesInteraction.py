from .. import Rule, State, Geometry
import numpy as np
from json import load, JSONDecodeError
from typing import Dict
from itertools import permutations


InteractionConfig = Dict[str, Dict[str, float]]


class SpeciesInteraction(Rule):
    def __init__(
        self,
        geometry: Geometry,
        interactions: InteractionConfig,
        spawn_claim_scale: float | None = None,
    ):
        super().__init__(geometry=geometry)
        self._interactions = interactions
        self.spawn_claim_scale = spawn_claim_scale

        self.required_keys.extend(interactions.keys())

    @classmethod
    def load_json_config(cls, filepath: str):
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                interaction_rules = load(file)
            return interaction_rules
        except FileNotFoundError as e:
            raise FileNotFoundError("config file not found") from e
        except JSONDecodeError as e:
            raise ValueError(e) from e

    def get_interaction(self, species_a: str, species_b: str) -> float:
        """gets interaction coeff, returns 0.0 if there's no interaction"""
        return self._interactions.get(species_a, {}).get(species_b, 0.0)

    def _get_neighbour_counts(self, grid: np.ndarray) -> np.ndarray:
        neighbour_counts = np.zeros_like(grid, dtype=np.int32)

        for offset in getattr(self.geometry, "_offsets", []):
            shifted = self._shift_for_offset(grid, tuple(int(v) for v in offset))
            neighbour_counts += shifted

        return neighbour_counts

    def apply_state(self, state: State) -> State:

        bacteria_types = list(self._interactions.keys())

        presence = {}
        occupied = np.zeros(state.shape, dtype=bool)
        neighbour_counts = {}
        spawns = {}
        deaths = {}

        for sp in bacteria_types:
            presence[sp] = (
                state[sp].reshape(state.shape).astype(bool)
            )  # checks presence of each species on the grid
            occupied |= presence[sp]  # mark all occupied grid spots for all species
            neighbour_counts[sp] = self._get_neighbour_counts(
                presence[sp].astype(np.uint8)
            )

            spawns[sp] = np.zeros(
                state.shape, dtype=np.float32
            )  # how much a species sp wnats to grow on a gridspot
            deaths[sp] = np.zeros(state.shape, dtype=bool)  # defines cells to die

        empty = ~occupied

        for bact_a, bact_b in permutations(bacteria_types, 2):
            coeff = self.get_interaction(bact_a, bact_b)

            if coeff == 0.0:  # no influence on each other between species
                continue

            if coeff < 0:  # species are destructive
                kill_prob = np.clip(abs(coeff) * neighbour_counts[bact_a], 0.0, 1.0)
                killed = presence[bact_b] & (np.random.random(state.shape) < kill_prob)
                deaths[bact_b] |= killed

            elif coeff > 0:  # species are in symbiosis
                spawns[bact_b] += (
                    empty
                    * coeff
                    * neighbour_counts[bact_a]
                    * (neighbour_counts[bact_b] >= 1)
                )

        # resolving deaths from previous stes
        for sp in bacteria_types:
            state[sp] = (
                (presence[sp] & ~deaths[sp])
                .reshape(state[sp].shape)
                .astype(state[sp].dtype)
            )

        claim_grid_spot = np.stack([spawns[sp] for sp in bacteria_types], axis=-1)
        max_claim = claim_grid_spot.max(axis=-1)

        winner_idx = np.argmax(
            claim_grid_spot + np.random.random(claim_grid_spot.shape) * 1e-6, axis=-1
        )
        any_claim = max_claim > 0

        if self.spawn_claim_scale is None:
            spawn_here = (
                empty & any_claim
            )  # if scale set to none, prob of spawn is = 1 for any claim
        else:
            spawn_probability = np.clip(max_claim / self.spawn_claim_scale, 0.0, 1.0)
            spawn_here = (
                empty & any_claim & (np.random.random(state.shape) < spawn_probability)
            )

        for i, sp in enumerate(bacteria_types):
            wins = spawn_here & (winner_idx == i)
            if wins.any():
                state[sp] = np.where(wins.reshape(state[sp].shape), 1, state[sp])

        return state
