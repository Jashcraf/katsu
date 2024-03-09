from katsu.katsu_math import np, broadcast_outer
from katsu.mueller import (
    _empty_mueller,
    linear_polarizer,
    linear_retarder,
    linear_diattenuator,
    decompose_diattenuator,
    decompose_retarder
)

def test_empty_mueller():

    # just check the shape
    M = _empty_mueller(None)

    np.testing.assert_allclose(M.shape,[4,4])

def test_empty_mueller_broadcast():

    # just check the shape
    M = _empty_mueller([32, 32])

    np.testing.assert_allclose(M.shape,[32, 32, 4, 4])

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