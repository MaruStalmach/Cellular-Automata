from typing import Protocol

import numpy as np


class Neighbourhood(Protocol):
    def generate_offsets(self, ndim: int, radius: int = 1) -> np.ndarray:
        pass
