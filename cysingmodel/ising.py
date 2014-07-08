"""Simulate a two-dimensional Ising model

We manipulate the single-body H term to stabilize two clusters
of oppositely magnetized spins.
"""

__author__ = 'Matthew Harrigan <matthew.p.harrigan@gmail.com>'

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import Normalize

# Constants
NDIM = 2
J = 10
H = 10
BETA = 1 / 30

# Neighbor offsets
NEIGHB = np.asarray([
    [-1, 0],
    [0, -1],
    [1, 0],
    [0, 1]
])


def generate_hmask(boxl):
    """Generate single-body field, H.

    This will be positive for the whole space except for two blocks
    centered at (20, 50) and (80, 50)
    """

    # TODO: Just make boxl a constant
    assert boxl == 100
    hmask = H * np.ones((boxl, boxl), dtype=int)

    hmask[_block(20, 50)] = -2 * H
    hmask[_block(80, 50)] = -2 * H

    return hmask


def _block(x, y, size=20):
    """Helper function to give an array slice for a block around a point."""
    s2 = size // 2
    return slice(x - s2, x + s2), slice(y - s2, y + s2)


def mc_loop(n_steps, boxl, init_cells=None):
    """Perform Monte Carlo simulation."""

    # Generate H
    hmask = generate_hmask(boxl)

    # Optionally generate initial condition
    if init_cells is None:
        cells = np.ones((boxl, boxl), dtype='i1')
        cells[_block(20, 50)] = -1
        cells[_block(80, 50)] = -1
    else:
        cells = init_cells

    # Save configurations
    # TODO: Only save every N ~ 100 frames
    cells_t = np.ones((n_steps, boxl, boxl), dtype='i1')
    cells_t[0, :, :] = cells

    for step in range(n_steps):

        # Pick a cell
        pick_i = np.random.random_integers(0, boxl - 1, NDIM)
        s_pick = cells[pick_i[0], pick_i[1]]

        # Find neighbors
        neighbs = np.remainder(pick_i + NEIGHB, boxl)
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
        # TODO: Only save some
        cells_t[step, :, :] = cells

    return cells_t


def plot(cells_t):
    """Save images over time of the 2D cell array."""
    boxl = cells_t.shape[1]
    xx, yy = np.meshgrid(np.arange(boxl), np.arange(boxl))

    for i, cells in enumerate(cells_t[::1000, ...]):
        plt.scatter(xx, yy, c=cells, norm=Normalize(-1, 1), s=10, linewidths=0)
        plt.title(str(i))
        figfn = '/home/harrigan/implement/wetmsm/ising/mov/ising-{:05d}.png'
        plt.savefig(figfn.format(i))
        plt.clf()


def plot_m(cells_t):
    """Plot the total magnetization, M, over time."""
    # TODO: Compute this in the main loop at every step?
    m = np.sum(cells_t, axis=(1, 2))
    plt.plot(m)
    plt.xlabel('Time')
    plt.ylabel('Magnetization')
    plt.show()


print('Running Equilibration')
CELLST = mc_loop(100000, 100)
print('Running Production')
CELLST = mc_loop(100000, 100, init_cells=CELLST[-1, ...])
print('Making Movie')
plot(CELLST)
print('Plotting Magnetization')
plot_m(CELLST)
