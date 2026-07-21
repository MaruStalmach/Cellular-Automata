import numpy as np
from typing import Protocol

class Neighbourhood(Protocol):
    def generate_offsets(self, ndim:int, radius:int=1) -> np.ndarray:
        pass

