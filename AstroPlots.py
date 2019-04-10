import numpy as np
import matplotlib.pyplot as plt
import sys


def plot_cmd(color, mag, xlim=[-1,4], ylim=[20,30], xlabel='color', \
    ylabel='mag', cbar_max=None, cbar_min=None, cbar_scale='linear', \
    cmap=plt.cm.viridis, plt_axes=False, save_as='None', rasterized=False):
    """
    Plot color magnitude diagram as a 2D density plot.

    Required parameters:

    color - array containing the colors of stars
    mag   - array containing the magnitudes of stars

    Optional Keywords:

    xlim       - define the color range to be considered
    ylim       - define the magnitude range to be considered
    xlabel     - label for the x-axis of the plot
    ylabel     - label for the y-axis of the plot
    cbar_max   - density value that corresponds to top of color bar; default will
                 use the maximum density
    cbar_min   - density value the corresponds to the bottom end of the color bar;
                 default will use the minimum density
    cbar_scale - Available options ['linear', 'log', 'arcsinh']
    cmap       - change the color map used to generate cmd
    plt_axes   - Axes on which to plot; default is False, and a new figure will be created
    save_as    - file name to save output plot; default is to display, but not
                 save the cmd
    rasterized - Whether or not to rasterize the points; default is False

    """

    Z, xedges, yedges = np.histogram2d(color,mag,bins=(500,1000), \
        range=[xlim, ylim])
    Z[Z == 0] = np.nan
    Y, X = np.meshgrid(yedges, xedges)

    if plt_axes == False:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
    else:
        ax = plt_axes

    if cbar_scale == 'linear':
        ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=cbar_min, vmax=cbar_max)
    elif cbar_scale == 'log':
        Z_new = np.log10(Z)
        ax.pcolormesh(X, Y, Z_new, cmap=cmap, vmin=cbar_min, vmax=cbar_max)
    elif cbar_scale == 'arcsinh':
        Z_new = np.arcsinh(Z)
        ax.pcolormesh(X, Y, Z_new, cmap=cmap, vmin=cbar_min, vmax=cbar_max)
    elif cbar_scale == 'sqrt':
        Z_new = np.sqrt(Z)
        ax.pcolormesh(X, Y, Z_new, cmap=cmap, vmin=cbar_min, vmax=cbar_max)
    else:
        print 'Invalid color scale. Choose from: linear, log, arcsinh, sqrt'
        sys.exit()
    # Flip y axis for CMD
    ax.set_ylim(ylim[1],ylim[0])
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    #if save_as == 'None':
    #    plt.show()
    #else:
    if save_as != 'None':
        print 'Saving CMD to file...'
        plt.savefig(save_as, format='pdf')

def plot_2D_density(x, y, xlim=[-10,10], ylim=[-10,10], xlabel='X', \
    ylabel='Y', cbar_max=None, cbar_min=None, cbar_scale='linear', \
    cmap=plt.cm.viridis, plt_axes=False, save_as='None'):
    """
    Plot generic data as a 2D density plot.

    Required parameters:

    x - array containing the x values
    y   - array containing the y values

    Optional Keywords:

    xlim       - define the color range to be considered
    ylim       - define the magnitude range to be considered
    xlabel     - label for the x-axis of the plot
    ylabel     - label for the y-axis of the plot
    cbar_max   - density value that corresponds to top of color bar; default will
                 use the maximum density
    cbar_min   - density value the corresponds to the bottom end of the color bar;
                 default will use the minimum density
    cbar_scale - Available options ['linear', 'log', 'arcsinh']
    cmap       - change the color map used to generate cmd
    save_as    - file name to save output plot; default is to display, but not
                 save the cmd

    """

    Z, xedges, yedges = np.histogram2d(x,y,bins=(200,200), \
        range=[xlim, ylim])
    Z[Z == 0] = np.nan
    Y, X = np.meshgrid(yedges, xedges)

    if plt_axes == False:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    else: ax = plt_axes

    if cbar_scale == 'linear':
        ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=cbar_min, vmax=cbar_max)
    elif cbar_scale == 'log':
        Z_new = np.log10(Z)
        ax.pcolormesh(X, Y, Z_new, cmap=cmap, vmin=cbar_min, vmax=cbar_max)
    elif cbar_scale == 'arcsinh':
        Z_new = np.arcsinh(Z)
        ax.pcolormesh(X, Y, Z_new, cmap=cmap, vmin=cbar_min, vmax=cbar_max)
    elif cbar_scale == 'sqrt':
        Z_new = np.sqrt(Z)
        ax.pcolormesh(X, Y, Z_new, cmap=cmap, vmin=cbar_min, vmax=cbar_max)
    else:
        print 'Invalid color scale. Choose from: linear, log, arcsinh, sqrt'
        sys.exit()

    if (plt_axes == False) & (save_as == 'None'):
        plt.show()

    if save_as != 'None':
        print 'Saving CMD to file...'
        plt.savefig(save_as, format='pdf')


def residual_plot(w=3.5, h=3.5):

    fig = plt.figure(figsize=(w,h))
    grid = plt.Gridspec(3,1. hspace=0.0)

    ax1 = fig.add_subplot(grid[:-1,:])
    ax2 = fig.add_subplot(grid[-1:,:])
    ax1.tick_params(axis='x', direction='in')
    ax1.set_xticklabels([])

    return ax1, ax2
