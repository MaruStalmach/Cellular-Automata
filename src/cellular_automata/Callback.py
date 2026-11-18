import numpy as np

from cellular_automata.registry import register_callback
from cellular_automata.Sim import CellularAutomaton


@register_callback
class Callback:
    """A class to generalize update_callback functions used for rendering"""

    def __init__(
        self,
        key_to_display: str,
        sim_obj_reference: CellularAutomaton,
        step_stride: int = 1,
    ):
        if not isinstance(sim_obj_reference, CellularAutomaton):
            raise TypeError(
                f"Callback needs a CellularAutomaton, got {type(sim_obj_reference).__name__}"
            )
        self.key = key_to_display
        self.sim = sim_obj_reference
        self.stride = step_stride

    def __call__(self, step: bool = True):

        if step:
            for _ in range(self.stride):
                self.sim.step()

        try:
            return self.sim.state[self.key]
        except KeyError:
            print(f"RENDERER ERROR: NO SUCH KEY TO RENDER AS {self.key}")
            return 0


@register_callback
class ReduceCallback(Callback):
    def __init__(
        self,
        key_to_display,
        sim_obj_reference,
        step_stride=1,
        reduce_op: np.ufunc = np.maximum,
    ):
        super().__init__(key_to_display, sim_obj_reference, step_stride)
        if not isinstance(reduce_op, np.ufunc):
            raise TypeError(f"reduce_op must be a numpy ufunc, got {reduce_op!r}")
        self.op = reduce_op
        # this is intended only for layered state arrays
        if self.sim.state[self.key].ndim != self.sim.geometry.ndim + 1:
            raise ValueError(
                f"ReduceCallback needs a key with several layers; '{self.key}' has one"
            )

    def __call__(self, step=True):

        if step:
            for _ in range(self.stride):
                self.sim.step()

        try:
            return self.op.reduce(self.sim.state[self.key], axis=-1)
        except KeyError:
            return 0


@register_callback
class MultiKeyCallback(Callback):
    def __init__(self, keys_to_display, sim_obj_reference, step_stride=1):
        super().__init__(keys_to_display, sim_obj_reference, step_stride)
        self.callbacks = []
        for k in self.key:
            if self.sim.state[k].ndim == self.sim.geometry.ndim + 1:
                self.callbacks.append(ReduceCallback(k, self.sim, step_stride))
            else:
                self.callbacks.append(Callback(k, self.sim, step_stride))

    def __call__(self, step=True):
        arr_list = []

        if step:
            for _ in range(self.stride):
                self.sim.step()

        for cb in self.callbacks:
            try:
                arr_list.append(cb(step=False))
            except Exception:
                print("ERROR IN MULTICALLBACK")
                arr_list.append(0)
        # stack on axis 0 so it gives the arrays in order when flattened
        return np.stack(arr_list, axis=0)
