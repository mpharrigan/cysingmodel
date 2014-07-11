"""Simulate a two-dimensional Ising model

We manipulate the single-body H term to stabilize two clusters
of oppositely magnetized spins.
"""

__author__ = 'Matthew Harrigan <matthew.harrigan@outlook.com>'

from os.path import join as pjoin

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import Normalize
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

import cysingmodel as cy


BOXL = 120


class IsingModel:
    """A 2D Ising model with a dynamic single-body term, H

    :param j: Coupling strength, J
    :param h: Single-body strength, H
    :param temp: Temperature for monte-carlo criterion
    :param stride: How many MC steps to do between saving results

    """

    def __init__(self, j, h, temp, stride=2000):
        self.j = j
        self.h = h
        self.temp = temp
        self.stride = stride

        self.cells_t = np.ndarray((0, 0))
        self.m = np.ndarray((0,))
        self.y = np.ndarray((0,))

    def run(self, n_eq=100000, n_prod=1000000):
        """Run equilibration then production run.

        :param n_eq: Number of steps for equilibration
        :param n_prod: Number of steps for production run
        """

        # Run equilibration (or not)
        if n_eq > 0:
            print('Running Equilibration')
            cell_eq, _, _ = cy.mc_loop(n_eq, cy.generate_cells(), equilib=True,
                                       J=self.j, H=self.h, TEMP=self.temp,
                                       stride=self.stride)
            prod_init = cell_eq[-1, ...]
        else:
            prod_init = cy.generate_cells()

        # Run production
        print('Running Production')
        self.cells_t, self.m, self.y = cy.mc_loop(n_prod, prod_init, J=self.j,
                                                  H=self.h, TEMP=self.temp,
                                                  stride=self.stride)

    def plot_movie(self):
        """Use pyqtgraph to make a movie quickly."""
        pg.image(self.cells_t)
        QtGui.QApplication.instance().exec_()

    def plot_vars(self):
        """Plot variables over time

        Specifically M, total magnetization and Y, the relative position
        of the two blocks.

        Scale and offset the two variables so they can be seen on the
        same Y-axis
        """

        m = self.m[::self.stride] - np.mean(self.m[::self.stride])
        y = self.y - np.mean(self.y)

        m = -m / (1.0 * np.max(m))
        y = y / (1.0 * np.max(y))

        plt.plot(m, label='M')
        plt.plot(y, label='Y')
        plt.xlabel('Time')
        plt.ylabel('(Arbitrary units)')
        plt.legend(loc='best')
        plt.show()

    def save_movie(self, dirname):
        """Save images over time of the 2D cell array.

        :param dirname: Directory to save frames

        Note: This is slow
        """
        xx, yy = np.meshgrid(np.arange(BOXL), np.arange(BOXL))
        for i, cells in enumerate(self.cells_t):
            plt.scatter(xx, yy, c=cells, norm=Normalize(-1, 1), s=10,
                        linewidths=0)
            plt.title(str(i))
            figfn = pjoin(dirname, 'ising-{:05d}.png')
            plt.savefig(figfn.format(i))
            plt.clf()


if __name__ == "__main__":
    app = QtGui.QApplication([])
