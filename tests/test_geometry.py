import numpy as np
from scipy.sparse import csr_matrix
from libs.Geometry import Geometry

def test_applying_offset_explained() -> None:
    geometry = Geometry((2, 2), axes='xy', periodicity='y')
    adj_matrix = geometry.generate_neighbourhood_matrix()

    # coords = np.indices(geometry.size).reshape(geometry.ndim, -1).T
   
    # print(geometry._offsets)
    # print(adj_matrix.toarray())

