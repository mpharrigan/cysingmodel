"""Simulate a two-dimensional Ising model

We manipulate the single-body H term to stabilize two clusters
of oppositely magnetized spins.
"""

__author__ = 'Matthew Harrigan <matthew.p.harrigan@gmail.com>'

import numpy as np
cimport numpy as np
cimport cython

from cpython cimport bool

# Constants
cdef int NDIM = 2
cdef int BOXL = 100
cdef int J = 20
cdef int H = 10
cdef float BETA = 1 / 30

DTYPEI1 = np.int8
DTYPEI = np.int

ctypedef np.int_t DTYPEI_T
ctypedef np.int8_t DTYPEI1_T

# Neighbor offsets
cdef np.ndarray NEIGHB = np.asarray([
    [-1, 0],
    [0, -1],
    [1, 0],
    [0, 1]
], dtype=DTYPEI)


cdef np.ndarray[DTYPEI_T, ndim=2] generate_hmask(int y):
    """Generate single-body field, H.

    This will be positive for the whole space except for two blocks
    centered at (20, 50) and (80, 50)

    :param y: Move towards each other by y

    They touch at 40, 60 which is 20 steps.
    """
    cdef np.ndarray[DTYPEI_T, ndim=2] hmask = H * np.ones((BOXL, BOXL), dtype=DTYPEI)

    #TODO: Use like a sine function here or something

    # Third speed
    y //= 3

    # Move back and forth
    y %= 60
    if y > 30:
        y = 60 - y

    hmask[_block(20 + y, 50)] = -3 * H
    hmask[_block(80 - y, 50)] = -3 * H

    return hmask


cdef np.ndarray[DTYPEI1_T, ndim=2] generate_cells():
    """Generate initial configuration.

    This will be positive for the whole space except for two blocks
    centered at (20, 50) and (80, 50) in accordance with hmask
    """
    cdef np.ndarray[DTYPEI1_T, ndim=2] cells = np.ones((BOXL, BOXL), dtype=DTYPEI1)

    cells[_block(20, 50)] = -1
    cells[_block(80, 50)] = -1

    return cells


def _block(int x, int y, int size=20):
    """Helper function to give an array slice for a block around a point."""
    cdef int s2 = size // 2
    return slice(x - s2, x + s2), slice(y - s2, y + s2)


def mc_loop(int n_steps, np.ndarray[DTYPEI1_T, ndim=2] cells, int stride=1000, bool equilib=False):
    """Perform Monte Carlo simulation."""

    # Generate H
    cdef np.ndarray[DTYPEI_T, ndim=2] hmask = generate_hmask(0)

    # Save configurations
    cdef np.ndarray[DTYPEI1_T, ndim=3] cells_t = np.ones((n_steps // stride, BOXL, BOXL), dtype=DTYPEI1)
    cells_t[0, :, :] = cells

    # Magnetization
    cdef np.ndarray[DTYPEI_T, ndim=1] m = np.zeros(n_steps, dtype=DTYPEI)

    cdef int s_pick
    cdef np.ndarray[DTYPEI_T, ndim=1] pick_i
    cdef np.ndarray[DTYPEI_T, ndim=2] neighbs
    cdef np.ndarray[DTYPEI1_T, ndim=1] cell_neighbs

    cdef int e_old, e_new
    cdef float prob
    cdef int step

    for step in range(n_steps):

        # Pick a cell
        pick_i = np.random.random_integers(0, BOXL - 1, NDIM)
        s_pick = cells[pick_i[0], pick_i[1]]

        # Find neighbors
        neighbs = np.remainder(pick_i + NEIGHB, BOXL)
        cell_neighbs = cells[neighbs[:, 0], neighbs[:, 1]]

        # Calculate the old energy
        e_old = -J * np.sum(s_pick * cell_neighbs) - hmask[
            pick_i[0], pick_i[1]] * s_pick

        # Flip the spin and see what the new energy would be
        s_pick *= -1
        e_new = -J * np.sum(s_pick * cell_neighbs) - hmask[
            pick_i[0], pick_i[1]] * s_pick

        # Accept with probability
        prob = np.exp(-BETA * (e_new - e_old))
        if prob > np.random.ranf():
            cells[pick_i[0], pick_i[1]] *= -1

        # Save configuration
        if step % stride == 0:
            cells_t[step // stride, :, :] = cells
            if not equilib:
                hmask = generate_hmask(step // stride)

        # Compute total magnetization
        m[step] = np.sum(cells)

    return cells_t, m


def main():
    print('Running Equilibration')
    cells_eq, _ = mc_loop(100000, generate_cells(), equilib=True)

    print('Running Production')
    cells_t, m = mc_loop(400000, cells_eq[-1, ...])

if __name__ == '__main__':
    main()

