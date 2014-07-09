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
BOXL = 100

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


def plot_m(m):
    """Plot the total magnetization, M, over time."""
    plt.plot(m)
    plt.xlabel('Time')
    plt.ylabel('Magnetization')
    plt.show()


def main():

    J = 17
    H = 20
    TEMP = 30

    print('Running Equilibration')
    cells_eq, _ = cy.mc_loop(100000, cy.generate_cells(), equilib=True, J=J, H=H, TEMP=TEMP)

    print('Running Production')
    cells_t, m = cy.mc_loop(400000, cells_eq[-1, ...], J=J, H=H, TEMP=TEMP)

    print('Plotting Magnetization')
    #plot_m(m)

    plot_movie(cells_t)



app = QtGui.QApplication([])
main()
