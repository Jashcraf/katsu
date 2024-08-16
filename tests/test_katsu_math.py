
import jax.numpy as jnp
import pytest
from katsu.katsu_math import (
    BackendShim,
    np,
    set_backend_to_numpy,
    set_backend_to_cupy,
    set_backend_to_jax,
    broadcast_outer,
    broadcast_kron
)

try:
    import cupy
    cupy_installed = True

except Exception:
    cupy_installed = False

try:
    import jax.numpy
    jax_installed = True

except Exception:
    jax_installed = False

def test_BackendShim():

    try:
        from katsu.katsu_math import np
        success = True

    except Exception:
        success = False

    assert success


def test_set_backend_to_numpy():
    set_backend_to_numpy()

#TODO install cupy and test for specifics
@pytest.mark.skipif(cupy_installed is False, reason='cupy not found')
def test_set_backend_to_cupy():
    set_backend_to_cupy()
    assert np.__name__ == "cupy"
    set_backend_to_numpy()


@pytest.mark.skipif(jax_installed is False, reason='jax not found')
def test_set_backend_to_jax(): 
    set_backend_to_jax()
    assert jnp.__name__ == np.__name__
    set_backend_to_numpy()
    

def test_broadcast_outer():

    v1 = np.random.random(4)
    v2 = np.random.random(4)

    v1_img = np.broadcast_to(v1, [32, 32, *v1.shape])
    v2_img = np.broadcast_to(v2, [32, 32, *v2.shape])

    res_broadcast = broadcast_outer(v1_img, v2_img)
    res_naive = np.zeros_like(res_broadcast)

    for i in range(32):
        for j in range(32):
            res_naive[i, j] = np.outer(v1_img[i, j], v2_img[i, j])

    np.testing.assert_allclose(res_broadcast, res_naive)


def test_broadcast_kron():

    v1 = np.random.random([4, 4])
    v2 = np.random.random([4, 4])
    
    v1_img = np.broadcast_to(v1, [32, 32,*v1.shape])
    v2_img = np.broadcast_to(v2, [32, 32,*v2.shape])

    res_broadcast = broadcast_kron(v1_img, v2_img)
    res_naive = np.zeros_like(res_broadcast)

    for i in range(32):
        for j in range(32):
            res_naive[i, j] = np.kron(v1_img[i, j], v2_img[i, j])

    np.testing.assert_allclose(res_broadcast, res_naive)