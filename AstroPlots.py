import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde



def plot_cmd(color, mag, xlim=[-1,4], ylim=[20,30], xlabel='color', \
    ylabel='mag', cbar_max=None, cbar_min=None, cbar_scale='linear', \
    cmap=plt.cm.viridis, save_as='None'):
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
    save_as    - file name to save output plot; default is to display, but not
                 save the cmd

    """

    Z, xedges, yedges = np.histogram2d(color,mag,bins=(500,1000), \
        range=[xlim, ylim])
    Z[Z == 0] = np.nan
    Y, X = np.meshgrid(yedges, xedges)

    
    fig = plt.figure(figsize=(7, 7))
    Z2 = np.log10(Z)
    Z3 = np.arcsinh(Z)

    ax = fig.add_subplot(111)
    if cbar_scale == 'linear':
        ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=cbar_min, vmax=cbar_max)
    elif cbar_scale == 'log':
        ax.pcolormesh(X, Y, Z2, cmap=cmap, vmin=cbar_min, vmax=cbar_max)
    elif cbar_scale == 'arcsinh':
        ax.pcolormesh(X, Y, Z3, cmap=pcmap, vmin=cbar_min, vmax=cbar_max)
    ax.set_ylim(ylim[1],ylim[0])
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if save_as == 'None':
        plt.show()
    else:
        print 'Saving CMD to file...'
        plt.savefig(save_as, format='pdf')
