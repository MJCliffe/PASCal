from PASCal.utils import (
    normalise_crys_axes,
    round_array,
    orthomat,
    cell_vols,
)
from PASCal._legacy import Round, Orthomat, CellVol, CompErr, NormCRAX
import numpy as np
import random
import numpy as np


def test_round_array():
    """Fuzz array rounding function with random arrays of different sizes."""
    c = 1.23
    rounded = 1.2
    tests = 0
    I1 = np.float64(1)
    I2 = np.ones((2, 2))
    I3 = np.ones((3, 3, 3))

    while tests < 100:
        tests += 1
        c = 100 * random.random()
        dec = random.randint(0, 5)
        rounded = round(c, dec)
        for _I in [I1, I2, I3]:
            np.testing.assert_array_equal(round_array(_I * c, dec), (rounded * _I))
            np.testing.assert_array_equal(round_array(_I * c, dec), Round(_I * c, dec))


def test_orthomat():
    lattice = np.array(
        [
            [1, 1, 1, 90, 90, 90],
            [2.8704, 2.8704, 14.1918, 90.0, 90.0, 120.0],
        ]
    )
    orth = orthomat(lattice)
    M = np.array(
        [
            [
                [1.00000000e00, 0.00000000e00, 0.00000000e00],
                [0.00000000e00, 1.00000000e00, 0.00000000e00],
                [0.00000000e00, 0.00000000e00, 1.00000000e00],
            ],
            [
                [4.0227861e-01, 0.00000000e00, 0.00000000e00],
                [2.0113930e-01, 3.4838350e-01, 0.00000000e00],
                [0.00000000e00, 0.00000000e00, 7.0463225e-02],
            ],
        ]
    )
    for i in range(M.shape[0]):
        np.testing.assert_array_almost_equal(orth[i], M[i], decimal=5)
    np.testing.assert_array_almost_equal(orth, M, decimal=5)
    np.testing.assert_array_almost_equal(orth, Orthomat(lattice), decimal=5)


def test_cell_vols():
    lattice = np.array([[1, 1, 1, 90, 90, 90]])
    lattice2 = np.array([[2, 2, 2, 90, 90, 90]])
    np.testing.assert_array_equal(cell_vols(lattice), np.array([1]))
    np.testing.assert_array_equal(cell_vols(lattice2), np.array([8]))
    np.testing.assert_array_equal(
        cell_vols(np.vstack((lattice, lattice2))), np.array([1, 8])
    )
    np.testing.assert_array_almost_equal(
        cell_vols(np.vstack((lattice, lattice2))),
        CellVol(np.vstack((lattice, lattice2))),
        decimal=10,
    )

    test_lattices = np.array(
        [
            [6.6200, 11.3500, 6.6230, 90.0000, 78.5000, 90.0000],
            [6.6310, 11.3622, 6.6140, 90.0000, 78.4680, 90.0000],
            [6.6328, 11.3643, 6.6099, 90.0000, 78.4700, 90.0000],
            [6.6361, 11.3759, 6.6007, 90.0000, 78.4330, 90.0000],
            [6.6330, 11.3829, 6.6062, 90.0000, 78.3690, 90.0000],
        ]
    )

    np.testing.assert_array_almost_equal(
        cell_vols(test_lattices),
        np.array(
            [
                487.64223456,
                488.25751412,
                488.18098761,
                488.17749675,
                488.54451206,
            ]
        ),
    )

    np.testing.assert_array_almost_equal(
        CellVol(test_lattices), cell_vols(test_lattices), decimal=10
    )


def test_norm_crax():
    test_cryst_ax = np.array(
        [
            [1.84486359e-01, 6.61218445e-16, 6.54039014e00],
            [-5.67306611e-16, 1.13829000e01, 9.26697299e-16],
            [-6.53486001e00, 6.71767249e-16, 9.68237090e-01],
        ],
    )

    principal_components = np.array([-19.13685275, 30.03399627, 9.49341493])

    norm_crys_axes = np.array(
        [
            [4.86770737e-01, 1.74463733e-15, 1.72569427e01],
            [-1.49684919e-15, 3.00339963e01, 2.44510830e-15],
            [-1.72423514e01, 1.77247055e-15, 2.55471182e00],
        ]
    )

    np.testing.assert_array_almost_equal(
        normalise_crys_axes(test_cryst_ax, principal_components),
        NormCRAX(test_cryst_ax, principal_components),
        decimal=10,
    )

    np.testing.assert_array_almost_equal(
        normalise_crys_axes(test_cryst_ax, principal_components),
        norm_crys_axes,
        decimal=7,
    )
