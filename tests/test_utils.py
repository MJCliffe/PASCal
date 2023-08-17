from PASCal.utils import round_array, orthomat, cell_vol
import numpy as np
import random
from pytest import fixture
import numpy as np

def test_round_array():
    """Fuzz array rounding function with random arrays of different sizes."""
    c = 1.23
    rounded = 1.2
    tests = 0
    I1 = np.float64(1)
    I2 = np.ones((2, 2)) * np.float64(1.23)
    I3 = np.ones((3, 3, 3)) * np.float64(1.23)

    while tests < 100:
        tests += 1
        c = 100 * random.random()
        dec = random.randint(0, 5)
        rounded = round(c, dec)
        for _I in [I1, I2, I3]:
            assert round_array(_I * c, 1).all() == (rounded * _I).all()

def test_orthomat():
    lattice = np.array([[1, 1, 1, 90, 90, 90]])
    orth = orthomat(lattice)
    M = np.array([[[
        [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],

       [[ 0.00000000e+00,  0.000000000e+00, 0.00000000e+00],
        [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],

       [[0.000000000e+00, 0.000000000e+00, 0.000000000e+00],
        [0.000000000e+00, 0.000000000e+00, 0.000000000e+00],
        [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00]],

       [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],

       [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],

       [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]]])
    assert orth.all() == np.array([M]).all()
    orth = orthomat(np.hstack((lattice, lattice)))
    assert orth.all() == np.array([M, M]).all()

def test_cell_vol():
    lattice = np.array([[1, 1, 1, 90, 90, 90]])
    lattice2 = np.array([[2, 2, 2, 90, 90, 90]])
    assert cell_vol(lattice).all() == np.array([1]).all()
    assert cell_vol(lattice2).all() == np.array([8]).all()
    assert cell_vol(np.hstack((lattice, lattice2))).all() == np.array([1, 8]).all()
