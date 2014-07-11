"""Simulate a two-dimensional Ising model

We manipulate the single-body H term to stabilize two clusters
of oppositely magnetized spins.
"""

__author__ = 'Matthew Harrigan <matthew.p.harrigan@gmail.com>'

import numpy as np
cimport numpy as np
cimport cython

# Constants
cdef int NDIM = 2
cdef int BOXL = 120
cdef float TWOPI = np.pi * 2

DTYPEI1 = np.int8
DTYPEI = np.int
DTYPED = np.double

ctypedef np.int_t DTYPEI_T
ctypedef np.int8_t DTYPEI1_T
ctypedef np.double_t DTYPED_T


cdef np.ndarray[DTYPEI_T, ndim=2] generate_hmask(int y, int H):
    """Generate single-body field, H.

    This will be positive for the whole space except for two blocks

    :param y: Move towards each other by y (computed elsewhere)

    """
    cdef np.ndarray[DTYPEI_T, ndim=2] hmask = H * np.ones((BOXL, BOXL), dtype=DTYPEI)

    hmask[_block(35 + y, 50)] = -3 * H
    hmask[_block(85 - y, 50)] = -3 * H

    return hmask


def generate_cells():
    """Generate initial configuration.

    This will be positive for the whole space except for two blocks
    centered at (20, 50) and (80, 50) in accordance with hmask
    """
    cdef np.ndarray[DTYPEI1_T, ndim=2] cells = np.ones((BOXL, BOXL), dtype=DTYPEI1)

    cells[_block(35 - 20, 50)] = -1
    cells[_block(85 + 20, 50)] = -1

    return cells


def _block(int x, int y, int size=20):
    """Helper function to give an array slice for a block around a point."""
    cdef int s2 = size // 2
    return slice(x - s2, x + s2), slice(y - s2, y + s2)


def mc_loop(int n_steps, np.ndarray[DTYPEI1_T, ndim=2] cells,
        np.ndarray[DTYPEI_T, ndim=1] ys,
        int stride=1000,
        int J=20, int H=10, int TEMP=30):
    """Perform Monte Carlo simulation.
    
    :param cells: Initial configuration of cells
    :param ys: Array of values to set the movement of the blocks
    
    """

    # Save configurations
    cdef np.ndarray[DTYPEI1_T, ndim=3] cells_t = np.ones((n_steps // stride, BOXL, BOXL), dtype=DTYPEI1)
    cells_t[0, :, :] = cells

    # Magnetization
    cdef np.ndarray[DTYPEI_T, ndim=1] m = np.zeros(n_steps + 1, dtype=DTYPEI)
    m[0] = np.sum(cells)

    # Block Movement
    cdef np.ndarray[DTYPEI_T, ndim=2] hmask
    hmask = generate_hmask(ys[0], H=H)

    # Variable definitions
    cdef int s_pick
    cdef int picki1, picki2
    cdef np.ndarray[DTYPEI1_T, ndim=1] cell_neighbs

    cdef int e_old, e_new
    cdef float prob
    cdef int step
    cdef int ssi

    # Make all the random integers
    cdef np.ndarray[DTYPEI_T, ndim=1] randints = np.random.random_integers(0, BOXL - 1, NDIM * n_steps)
    # And floats
    cdef np.ndarray[double, ndim=1] randfloats = np.random.random_sample(n_steps)

    for step in range(n_steps):

        # Pick a cell
        picki1 = randints[step]
        picki2 = randints[n_steps + step]
        s_pick = cells[picki1, picki2]

        cell_neighbs = np.array([
                cells[(picki1 + 1) % BOXL, picki2],
                cells[picki1, (picki2 + 1) % BOXL],
                cells[(picki1 - 1) % BOXL, picki2],
                cells[picki1, (picki2 - 1) % BOXL]
        ], dtype=DTYPEI1)

        # Calculate the old energy
        e_old = -J * (
                s_pick * cell_neighbs[0] +
                s_pick * cell_neighbs[1] +
                s_pick * cell_neighbs[2] +
                s_pick * cell_neighbs[3]
                ) - hmask[picki1, picki2] * s_pick

        # Flip the spin and see what the new energy would be
        s_pick *= -1
        e_new = -J * (
                s_pick * cell_neighbs[0] +
                s_pick * cell_neighbs[1] +
                s_pick * cell_neighbs[2] +
                s_pick * cell_neighbs[3]
                ) - hmask[picki1, picki2] * s_pick

        # Accept with probability
        prob = np.exp(-(e_new - e_old)/TEMP)
        if prob > randfloats[step]:
            cells[picki1, picki2] *= -1

            # Magnetization changes by two times the new value
            m[step + 1] = m[step] + 2 * s_pick
        else:
            m[step + 1] = m[step]

        # Save configuration
        if step % stride == 0:
            ssi = step // stride
            cells_t[ssi, :, :] = cells
            hmask = generate_hmask(ys[ssi], H=H)

    return cells_t, m, ys


