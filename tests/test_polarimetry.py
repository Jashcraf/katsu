import numpy as np
from katsu.polarimetry import *

def test_broadcast_outer():

    v1 = np.random.random(4)
    v2 = np.random.random(4)

    v1_img = np.broadcast_to(v1,[32,32,*v1.shape])
    v2_img = np.broadcast_to(v2,[32,32,*v2.shape])

    res_broadcast = broadcast_outer(v1_img, v2_img)
    res_naive = np.zeros_like(res_broadcast)

    for i in range(32):
        for j in range(32):
            res_naive[i, j] = np.outer(v1_img[i, j], v2_img[i, j])

    np.testing.assert_allclose(res_broadcast, res_naive)

