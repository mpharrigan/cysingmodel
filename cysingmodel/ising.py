"""Simulate a two-dimensional Ising model

We manipulate the single-body H term to stabilize two clusters
of oppositely magnetized spins.
"""

__author__ = 'Matthew Harrigan <matthew.p.harrigan@gmail.com>'

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import Normalize

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import cysingmodel as cy

BOXL = 120


def plot(cells_t):
    """Save images over time of the 2D cell array.

    Note: This is very slow
    """
    xx, yy = np.meshgrid(np.arange(BOXL), np.arange(BOXL))
    for i, cells in enumerate(cells_t):
        plt.scatter(xx, yy, c=cells, norm=Normalize(-1, 1), s=10, linewidths=0)
        plt.title(str(i))
        figfn = '/home/harrigan/implement/wetmsm/ising/mov/ising-{:05d}.png'
        plt.savefig(figfn.format(i))
        plt.clf()


def plot_movie(cells_t):
    """Use pyqtgraph to make a movie quickly."""
    pg.image(cells_t)
    QtGui.QApplication.instance().exec_()


def plot_m(m, ys):
    """Plot the total magnetization, M, over time."""

    m -= np.mean(np.asarray(m, dtype=float))
    ys -= np.mean(np.asarray(ys, dtype=float))

    m = -m / (1.0 * np.max(m))
    ys = ys / (1.0 * np.max(ys))

    plt.plot(m, label='m')
    plt.plot(ys, label='y')
    plt.xlabel('Time')
    plt.ylabel('Magnetization')
    plt.legend(loc='best')
    plt.show()


def main():
    J = 20
    H = 20
    TEMP = 30
    STRIDE = 2000

    print('Running Equilibration')
    cells_eq, _, _ = cy.mc_loop(100000, cy.generate_cells(), equilib=True, J=J,
                                H=H, TEMP=TEMP, stride=STRIDE)

    print('Running Production')
    cells_t, m, ys = cy.mc_loop(1000000, cells_eq[-1, ...], J=J, H=H, TEMP=TEMP,
                                stride=STRIDE)

    plot_m(m[::STRIDE], ys)

    plot_movie(cells_t)

if __name__ == "__main__":
    app = QtGui.QApplication([])
    main()
