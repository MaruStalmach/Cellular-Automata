import numpy as np
from scipy.ndimage import label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from cellular_automata.Geometry import Geometry
from cellular_automata.registry import register_rule
from cellular_automata.Rule import Rule, StepContext, require_3d
from cellular_automata.State import State

DEFAULT_WALLS = ("y-",)


def parse_walls(geometry: Geometry, walls) -> list[tuple[int, int]]:
    """Turn wall names such as "y-" (the low-y face) into (axis index, side) pairs.

    The side is -1 for the low face (index 0) and +1 for the high face.
    """
    if isinstance(walls, str):
        raise ValueError(f"BiofilmDetachment: walls must be a list, e.g. [{walls!r}]")
    walls = list(DEFAULT_WALLS if walls is None else walls)
    if not walls:
        raise ValueError("BiofilmDetachment: walls must name at least one wall")
    parsed: list[tuple[int, int]] = []
    for wall in walls:
        if not (
            isinstance(wall, str)
            and len(wall) == 2
            and wall[0] in geometry.axes
            and wall[1] in "+-"
        ):
            raise ValueError(
                f"BiofilmDetachment: wall {wall!r} must be an axis of {geometry.axes!r} "
                "followed by '-' (low face) or '+' (high face), e.g. 'y-'"
            )
        axis = geometry.axes.index(wall[0])
        if axis in geometry.periodic_dims:
            raise ValueError(
                f"BiofilmDetachment: wall {wall!r} is on the periodic axis {wall[0]!r}, "
                "which has no faces"
            )
        side = -1 if wall[1] == "-" else 1
        if (axis, side) in parsed:
            raise ValueError(f"BiofilmDetachment: wall {wall!r} is listed twice")
        parsed.append((axis, side))
    return parsed


@register_rule
class BiofilmDetachment(Rule):
    """Biofilm comes off more easily the farther it is from the gut wall.

    Every call does three things, in this order:

    1. **Detachment.** An exposed biofilm cell detaches with probability
       ``detachment_rate * scaling * d**2``. ``d`` is the cell's distance to its
       nearest wall, in grid cells, divided by the largest such distance on the
       grid, so it runs from 0 at a wall to 1 at the cells farthest from every
       wall. ``detachment_rate`` and ``scaling`` are only ever used as this
       product: it is the chance per call that an exposed cell at ``d = 1``
       detaches. Two kinds of cell never detach:

       - cells touching a wall (``d = 0``)
       - sandwiched cells, with biofilm both below and above them. "Below" is one
         cell towards the cell's nearest wall, "above" one cell away from it.

    2. **Attachment.** A floating bacterium with biofilm below it attaches with
       ``attachment_probability``. It reads the biofilm as it was at the start of
       the call, so a cell that detached in this call can't re-attach until the next.

    3. **Release.** Biofilm only exists in bulk, attached to a wall: every
       biofilm cell no longer linked to a wall through a chain of neighbouring
       biofilm cells becomes floating. "Neighbouring" follows the geometry's
       neighbourhood (Moore or von Neumann) and wraps around periodic axes.

    Detached and released cells move from the "biofilm" key to "floating_bacteria".

    args:
    - detachment_rate, scaling - their product is the detachment probability at d = 1
    - species_keys - species that can attach (no attachment without them)
    - attachment_probability - chance per call that a floating bacterium on biofilm attaches
    - walls - the grid faces that are gut wall, each an axis and a side: "y-" is
      the low-y face (the floor when rendered), "y+" the high-y face. Default
      ["y-"]. With several walls each cell uses its nearest one, and the one
      listed first on a tie.
    """

    def __init__(
        self,
        geometry,
        time_step: float,
        detachment_rate: float,
        scaling: float,
        species_keys: list | None = None,
        attachment_probability: float = 1.0,
        walls: list[str] | None = None,
    ):
        super().__init__(geometry, time_step)
        require_3d(type(self).__name__, geometry)

        self.detachment_probability = detachment_rate * scaling
        self.target_keys = ["biofilm", "floating_bacteria"]
        self.required_keys.extend(self.target_keys)

        self.species_keys = species_keys or []
        self.attachment_probability = attachment_probability
        self.required_keys.extend(self.species_keys)

        if np.abs(geometry._offsets).max() > 1:
            raise ValueError(
                "BiofilmDetachment: only radius-1 neighbourhoods (Moore, von Neumann) are supported"
            )

        self.walls = parse_walls(geometry, walls)
        distance, self.nearest_wall = self._wall_distances()
        self.touches_wall = distance == 0
        scaled = distance.astype(np.float32) / max(1, int(distance.max()))
        self.probabilities = np.clip(
            np.float32(self.detachment_probability) * scaled**2, 0.0, 1.0
        ).astype(np.float32)

    def _wall_distances(self) -> tuple[np.ndarray, np.ndarray]:
        """Distance in cells from every cell to its nearest wall, and that wall's index (int8).

        On a tie the wall listed first wins.
        """
        size = self.geometry.size
        distance = np.full(size, np.iinfo(np.int32).max, dtype=np.int32)
        nearest = np.zeros(size, dtype=np.int8)
        for wall_index, (axis, side) in enumerate(self.walls):
            index = np.arange(size[axis], dtype=np.int32)
            along = index if side < 0 else size[axis] - 1 - index
            shape = [1] * len(size)
            shape[axis] = -1
            closer = along.reshape(shape) < distance  # strictly closer, so ties stay put
            distance = np.where(closer, along.reshape(shape), distance)
            nearest[closer] = wall_index
        return distance, nearest

    def apply_state(self, state: State, ctx: StepContext | None = None) -> State:
        biofilm_grid = state["biofilm"]
        floating_bact_grid = state["floating_bacteria"]
        biofilm_at_start = biofilm_grid.astype(bool)
        below_is_biofilm = self.towards_wall(biofilm_at_start)

        detached = self._detach(biofilm_at_start, below_is_biofilm)
        biofilm_grid[detached] = False
        floating_bact_grid[detached] = True

        if self.species_keys and self.attachment_probability > 0.0:
            attached = self._attach(state, biofilm_at_start, below_is_biofilm, detached)
            biofilm_grid[attached] = True
            floating_bact_grid[attached] = False

        biofilm_now = biofilm_grid.astype(bool)
        released = biofilm_now & ~self._linked_to_wall(biofilm_now)
        biofilm_grid[released] = False
        floating_bact_grid[released] = True

        state["biofilm"] = biofilm_grid
        state["floating_bacteria"] = floating_bact_grid

        return state

    def towards_wall(self, grid: np.ndarray, away: bool = False) -> np.ndarray:
        """For every cell, the value of `grid` one cell towards its nearest wall.

        With `away`, one cell away from that wall. Past the grid edge reads as 0.
        """
        result = np.zeros_like(grid)
        for wall_index, (axis, side) in enumerate(self.walls):
            offset = [0] * grid.ndim
            offset[axis] = -side if away else side  # towards a low wall is index - 1
            neighbour = self._shift_for_offset(grid, tuple(offset))
            mine = self.nearest_wall == wall_index
            result[mine] = neighbour[mine]
        return result

    def _detach(self, biofilm: np.ndarray, below_is_biofilm: np.ndarray) -> np.ndarray:
        sandwiched = below_is_biofilm & self.towards_wall(biofilm, away=True)
        coords = np.nonzero(biofilm & ~sandwiched)
        detach_mask = self.rng.random(len(coords[0])) < self.probabilities[coords]
        detached = np.zeros_like(biofilm)
        detached[tuple(c[detach_mask] for c in coords)] = True
        return detached

    def _attach(
        self,
        state: State,
        biofilm_at_start: np.ndarray,
        below_is_biofilm: np.ndarray,
        detached: np.ndarray,
    ) -> np.ndarray:
        # attachment sees the biofilm as it was at the start of the step, so a cell that
        # detached in this call can't re-attach until the next one
        occupied = np.zeros(self.geometry.size, dtype=bool)
        for sp in self.species_keys:
            occupied |= self._cell_mask(state[sp])

        candidates = occupied & ~biofilm_at_start & ~detached & below_is_biofilm
        if self.attachment_probability < 1.0:
            cand_coords = np.nonzero(candidates)
            rejected = self.rng.random(len(cand_coords[0])) >= self.attachment_probability
            candidates[tuple(c[rejected] for c in cand_coords)] = False
        return candidates

    def _linked_to_wall(self, biofilm: np.ndarray) -> np.ndarray:
        """Biofilm cells linked to a wall through a chain of neighbouring biofilm cells.

        Only the bounding box of the biofilm along non-periodic axes is searched:
        no link can cross the empty cells outside it, and biofilm is usually a
        thin mat on a large grid.
        """
        linked = np.zeros_like(biofilm)
        if not biofilm.any():
            return linked
        box: list[slice] = []
        for axis in range(biofilm.ndim):
            if axis in self.geometry.periodic_dims:
                box.append(slice(None))
                continue
            others = tuple(a for a in range(biofilm.ndim) if a != axis)
            present = np.flatnonzero(biofilm.any(axis=others))
            box.append(slice(int(present[0]), int(present[-1]) + 1))
        window = tuple(box)
        cropped = biofilm[window]
        component = self._components_by_label(cropped)
        anchored = np.unique(component[cropped & self.touches_wall[window]])
        linked[window] = cropped & np.isin(component, anchored)
        return linked

    def _components_by_label(self, biofilm: np.ndarray) -> np.ndarray:
        """Connected-component id of every biofilm cell (-1 elsewhere).

        `biofilm` spans the whole grid along every periodic axis. Periodic axes are
        padded with one wrapped layer on each side, and the
        labels of each padded copy and its original cell are then merged.
        """
        ndim = biofilm.ndim
        periodic = [axis in self.geometry.periodic_dims for axis in range(ndim)]
        structure = np.zeros((3,) * ndim, dtype=bool)
        structure[(1,) * ndim] = True
        for offset in self.geometry._offsets:
            structure[tuple(np.asarray(offset) + 1)] = True

        padded = np.pad(biofilm, [(1, 1) if p else (0, 0) for p in periodic], mode="wrap")
        labels, n_labels = label(padded, structure=structure)

        copies, originals = [], []
        for axis in range(ndim):
            if not periodic[axis]:
                continue
            length = biofilm.shape[axis]
            # padded index 0 copies original index length-1 (padded index length),
            # padded index length+1 copies original index 0 (padded index 1)
            for copy_index, original_index in ((0, length), (length + 1, 1)):
                copy_slice: list[slice | int] = [slice(None)] * ndim
                original_slice: list[slice | int] = [slice(None)] * ndim
                copy_slice[axis] = copy_index
                original_slice[axis] = original_index
                copies.append(labels[tuple(copy_slice)].ravel())
                originals.append(labels[tuple(original_slice)].ravel())
        merged = np.arange(n_labels + 1)
        if copies:
            source, target = np.concatenate(copies), np.concatenate(originals)
            graph = coo_matrix(
                (np.ones(len(source), dtype=np.int8), (source, target)),
                shape=(n_labels + 1, n_labels + 1),
            )
            _, merged = connected_components(graph, directed=False)

        core = tuple(slice(1, -1) if p else slice(None) for p in periodic)
        component = merged[labels[core]]
        component[~biofilm] = -1
        return component
