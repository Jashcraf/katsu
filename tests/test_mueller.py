import pytest
from katsu.katsu_math import np, broadcast_outer, set_backend_to_numpy, set_backend_to_jax
from katsu.mueller import (
    _empty_mueller,
    mueller_rotation,
    linear_polarizer,
    linear_retarder,
    linear_diattenuator,
    depolarizer,
    decompose_diattenuator,
    decompose_retarder,
    decompose_depolarizer,
    mueller_to_jones,
    depolarization_index,
    retardance_from_mueller,
    retardance_parameters_from_mueller,
    diattenuation_from_mueller,
    diattenuation_parameters_from_mueller

)

def test_empty_mueller():

    # just check the shape
    M = _empty_mueller(None)

    np.testing.assert_allclose(M.shape,[4,4])

def test_empty_mueller_broadcast():

    # just check the shape
    M = _empty_mueller([32, 32])

    np.testing.assert_allclose(M.shape,[32, 32, 4, 4])

def test_mueller_rotation():

    test = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    
    rotation = mueller_rotation(0)

    np.testing.assert_allclose(rotation, test)

def test_mueller_rotation_broadcast():
    test = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    
    test = np.broadcast_to(test, [32, 32, 4, 4])
    rotation = mueller_rotation(0, shape=[32, 32])

    np.testing.assert_allclose(rotation, test)
    

def test_linear_polarizer():

    # make a horizontal polarizer
    hpol = np.array([[1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]]) / 2
    
    hpol_test = linear_polarizer(0)

    np.testing.assert_allclose(hpol_test, hpol)

def test_linear_polarizer_broadcast():

    # make a horizontal polarizer
    hpol = np.array([[1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]]) / 2
    
    hpol = np.broadcast_to(hpol, [32, 32, 4, 4])
    
    hpol_test = linear_polarizer(0, shape=[32, 32])

    np.testing.assert_allclose(hpol_test, hpol)

def test_linear_retarder():

    # make a horizontal QWP
    qwp = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, -1, 0]])
    
    qwp_test = linear_retarder(0, np.pi/2)

    np.testing.assert_allclose(qwp_test, qwp, atol=1e-12)

def test_linear_retarder_broadcast():

    # make a horizontal QWP
    qwp = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, -1, 0]])
    
    qwp = np.broadcast_to(qwp, [32, 32, 4, 4])
    
    qwp_test = linear_retarder(0, np.pi/2, shape=[32, 32])

    np.testing.assert_allclose(qwp_test, qwp, atol=1e-12)

def test_linear_diattenuator():

    # make a horizontal polarizer
    hpol = np.array([[1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]]) / 2
    
    ppol = np.array([[1, 0, 1, 0],
                     [0, 0, 0, 0],
                     [1, 0, 1, 0],
                     [0, 0, 0, 0]]) / 2
    
    hpol_test = linear_diattenuator(0, 0)
    ppol_test = linear_diattenuator(np.pi/4, 0)

    np.testing.assert_allclose((hpol_test, ppol_test), (hpol, ppol), atol=1e-12)

def test_depolarizer():
    a, b, c = 0.9, 0.4, 0.1
    test = np.array([[1, 0, 0, 0],
                     [0, a, 0, 0],
                     [0, 0, b, 0],
                     [0, 0, 0, c]])
    
    depol = depolarizer(0, a, b, c)

    np.testing.assert_allclose(depol, test)

def test_decompose_diattenuator():

    # make a horizontal polarizer
    hpol = linear_diattenuator(0, 0)
    qwp = linear_retarder(np.pi/4,np.pi/2)

    cpol = qwp @ hpol

    Md = decompose_diattenuator(cpol)

    np.testing.assert_allclose(Md, hpol)

def test_decompose_retarder():

    # make a horizontal polarizer
    hpol = linear_diattenuator(0, 0.1)
    qwp = linear_retarder(np.pi/4,np.pi/2)

    cpol = qwp @ hpol

    Mr = decompose_retarder(cpol)

    np.testing.assert_allclose(Mr, qwp, atol=1e-12)

def test_decompose_depolarizer():

    # make a horizontal polarizer
    hpol = linear_diattenuator(0, 0.1)
    qwp = linear_retarder(np.pi/4,np.pi/2)

    depol = np.array([[1., 0., 0., 0.],
                      [0., 0.9, 0., 0.],
                      [0., 0., 0.8, 0.],
                      [0., 0., 0., 0.7]])
    
    Mtot = depol @ qwp @ hpol

    Md = decompose_depolarizer(Mtot)

    np.testing.assert_allclose(Md, depol, atol=1e-12)

def test_mueller_to_jones():
    
    jones = np.array([[1, 0],[0, 0]])  # h polarizer
    mueller = linear_polarizer(0)

    jones_test = mueller_to_jones(mueller)

    np.testing.assert_allclose(jones, jones_test)

def test_depolarization_index():

    M = linear_polarizer(0)
    DI = depolarization_index(M)

    np.testing.assert_allclose(DI, 1.)

def test_retardance_from_mueller():

    retardance = np.pi / 2
    M = linear_retarder(0, retardance)
    R = retardance_from_mueller(M)

    np.testing.assert_allclose(R, retardance)

def test_retardance_parameters_from_mueller():

    retardance = np.pi / 2
    M = linear_retarder(0, retardance)
    rh, rp, rc = retardance_parameters_from_mueller(M)

    np.testing.assert_allclose((rh, rp, rc), (retardance, 0., 0.))


def test_diattenuation_from_mueller():

    M = linear_polarizer(0)
    D = diattenuation_from_mueller(M)

    np.testing.assert_allclose(D, 1.)

def test_diattenuation_parameters_from_mueller():

    M = linear_polarizer(0)
    dh, dp, dc = diattenuation_parameters_from_mueller(M)

    np.testing.assert_allclose((dh, dp, dc), (1., 0., 0.))

# testing each method using jax backend
# first test will set to jax backend and last test will set back to numpy

try:
    import jax.numpy as jnp
    jax_installed = True
    import jax
    jax.config.update("jax_enable_x64", True)

except Exception:
    jax_installed = False

@pytest.mark.skipif(jax_installed is False, reason='jax not found')    
def test_empty_mueller_jax():
    # setting backend to jax
    set_backend_to_jax()

    # just check the shape
    M = _empty_mueller(None)

    assert np.allclose(np.array(M.shape),np.array([4,4]))

@pytest.mark.skipif(jax_installed is False, reason='jax not found') 
def test_empty_mueller_broadcast_jax():

    # just check the shape
    M = _empty_mueller([32, 32])

    assert np.allclose(np.array(M.shape),np.array([32, 32, 4, 4]))

@pytest.mark.skipif(jax_installed is False, reason='jax not found') 
def test_mueller_rotation_jax():

    test = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    
    rotation = mueller_rotation(0)

    assert np.allclose(rotation, test)

@pytest.mark.skipif(jax_installed is False, reason='jax not found') 
def test_mueller_rotation_broadcast_jax():
    test = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    
    test = np.broadcast_to(test, [32, 32, 4, 4])
    rotation = mueller_rotation(0, shape=[32, 32])

    assert np.allclose(rotation, test)
    
@pytest.mark.skipif(jax_installed is False, reason='jax not found') 
def test_linear_polarizer_jax():

    # make a horizontal polarizer
    hpol = np.array([[1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]]) / 2
    
    hpol_test = linear_polarizer(0)

    assert np.allclose(hpol_test, hpol)

@pytest.mark.skipif(jax_installed is False, reason='jax not found') 
def test_linear_polarizer_broadcast_jax():

    # make a horizontal polarizer
    hpol = np.array([[1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]]) / 2
    
    hpol = np.broadcast_to(hpol, [32, 32, 4, 4])
    
    hpol_test = linear_polarizer(0, shape=[32, 32])

    assert np.allclose(hpol_test, hpol)

@pytest.mark.skipif(jax_installed is False, reason='jax not found') 
def test_linear_retarder_jax():

    # make a horizontal QWP
    qwp = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, -1, 0]])
    
    qwp_test = linear_retarder(0, np.pi/2)

    assert np.allclose(qwp_test, qwp, atol=1e-12)

@pytest.mark.skipif(jax_installed is False, reason='jax not found') 
def test_linear_retarder_broadcast_jax():

    # make a horizontal QWP
    qwp = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, -1, 0]])
    
    qwp = np.broadcast_to(qwp, [32, 32, 4, 4])
    
    qwp_test = linear_retarder(0, np.pi/2, shape=[32, 32])

    assert np.allclose(qwp_test, qwp, atol=1e-12)

@pytest.mark.skipif(jax_installed is False, reason='jax not found') 
def test_linear_diattenuator_jax():

    # make a horizontal polarizer
    hpol = np.array([[1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]]) / 2
    
    ppol = np.array([[1, 0, 1, 0],
                     [0, 0, 0, 0],
                     [1, 0, 1, 0],
                     [0, 0, 0, 0]]) / 2
    
    hpol_test = linear_diattenuator(0, 0)
    ppol_test = linear_diattenuator(np.pi/4, 0)

    assert np.allclose(np.array([hpol_test, ppol_test]), np.array([hpol, ppol]), atol=1e-12)

@pytest.mark.skipif(jax_installed is False, reason='jax not found') 
def test_depolarizer_jax():
    a, b, c = 0.9, 0.4, 0.1
    test = np.array([[1, 0, 0, 0],
                     [0, a, 0, 0],
                     [0, 0, b, 0],
                     [0, 0, 0, c]])
    
    depol = depolarizer(0, a, b, c)

    assert np.allclose(depol, test)

@pytest.mark.skipif(jax_installed is False, reason='jax not found') 
def test_decompose_diattenuator_jax():

    # make a horizontal polarizer
    hpol = linear_diattenuator(0, 0)
    qwp = linear_retarder(np.pi/4,np.pi/2)

    cpol = qwp @ hpol

    Md = decompose_diattenuator(cpol)

    assert np.allclose(Md, hpol)

@pytest.mark.skipif(jax_installed is False, reason='jax not found') 
def test_decompose_retarder_jax():

    # make a horizontal polarizer
    hpol = linear_diattenuator(0, 0.1)
    qwp = linear_retarder(np.pi/4,np.pi/2)

    cpol = qwp @ hpol

    Mr = decompose_retarder(cpol)

    assert np.allclose(Mr, qwp, atol=1e-12)

@pytest.mark.skipif(jax_installed is False, reason='jax not found') 
def test_decompose_depolarizer_jax():

    # make a horizontal polarizer
    hpol = linear_diattenuator(0, 0.1)
    qwp = linear_retarder(np.pi/4,np.pi/2)

    depol = np.array([[1., 0., 0., 0.],
                      [0., 0.9, 0., 0.],
                      [0., 0., 0.8, 0.],
                      [0., 0., 0., 0.7]])
    
    Mtot = depol @ qwp @ hpol

    Md = decompose_depolarizer(Mtot)
    
    assert np.allclose(Md, depol, atol=1e-12)

@pytest.mark.skipif(jax_installed is False, reason='jax not found') 
def test_mueller_to_jones_jax():
    
    jones = np.array([[1, 0],[0, 0]])  # h polarizer
    mueller = linear_polarizer(0)

    jones_test = mueller_to_jones(mueller)

    assert np.allclose(jones, jones_test)

@pytest.mark.skipif(jax_installed is False, reason='jax not found') 
def test_depolarization_index_jax():

    M = linear_polarizer(0)
    DI = depolarization_index(M)

    assert np.allclose(DI, 1.)

@pytest.mark.skipif(jax_installed is False, reason='jax not found') 
def test_retardance_from_mueller_jax():

    retardance = np.pi / 2
    M = linear_retarder(0, retardance)
    R = retardance_from_mueller(M)

    assert np.allclose(R, retardance)

@pytest.mark.skipif(jax_installed is False, reason='jax not found') 
def test_retardance_parameters_from_mueller_jax():

    retardance = np.pi / 2
    M = linear_retarder(0, retardance)
    rh, rp, rc = retardance_parameters_from_mueller(M)

    assert np.allclose(np.array([rh, rp, rc]), np.array([retardance, 0., 0.]))

@pytest.mark.skipif(jax_installed is False, reason='jax not found') 
def test_diattenuation_from_mueller_jax():

    M = linear_polarizer(0)
    D = diattenuation_from_mueller(M)

    assert np.allclose(D, 1.)

@pytest.mark.skipif(jax_installed is False, reason='jax not found') 
def test_diattenuation_parameters_from_mueller_jax():

    M = linear_polarizer(0)
    dh, dp, dc = diattenuation_parameters_from_mueller(M) 

    result = np.allclose(np.array([dh, dp, dc]), np.array([1., 0., 0.]))
    # setting backend back to numpy
    set_backend_to_numpy()

    assert result