import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.gridspec as gridspec
from astropy.visualization import LogStretch, ImageNormalize, PercentileInterval
from astropy.io import fits
from matplotlib.patches import Circle

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
        ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=cbar_min, vmax=cbar_max,
            rasterized=rasterized)
    elif cbar_scale == 'log':
        Z_new = np.log10(Z)
        ax.pcolormesh(X, Y, Z_new, cmap=cmap, vmin=cbar_min, vmax=cbar_max,
            rasterized=rasterized)
    elif cbar_scale == 'arcsinh':
        Z_new = np.arcsinh(Z)
        ax.pcolormesh(X, Y, Z_new, cmap=cmap, vmin=cbar_min, vmax=cbar_max,
            rasterized=rasterized)
    elif cbar_scale == 'sqrt':
        Z_new = np.sqrt(Z)
        ax.pcolormesh(X, Y, Z_new, cmap=cmap, vmin=cbar_min, vmax=cbar_max,
            rasterized=rasterized)
    else:
        print('Invalid color scale. Choose from: linear, log, arcsinh, sqrt')
        sys.exit()
    # Flip y axis for CMD
    #ax.set_ylim(ylim[1],ylim[0])
    #ax.set_xlim(xlim)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    #if save_as == 'None':
    #    plt.show()
    #else:
    if save_as != 'None':
        print('Saving CMD to file...')
        plt.savefig(save_as, format='pdf')

def plot_2D_density(x, y, xlim=[-10,10], ylim=[-10,10], xlabel='X', \
    ylabel='Y', cbar_max=None, cbar_min=None, cbar_scale='linear', \
    cmap=plt.cm.viridis, plt_axes=False, save_as='None', rasterized=False):
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
        ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=cbar_min, vmax=cbar_max,
            rasterized=rasterized)
    elif cbar_scale == 'log':
        Z_new = np.log10(Z)
        ax.pcolormesh(X, Y, Z_new, cmap=cmap, vmin=cbar_min, vmax=cbar_max,
            rasterized=rasterized)
    elif cbar_scale == 'arcsinh':
        Z_new = np.arcsinh(Z)
        ax.pcolormesh(X, Y, Z_new, cmap=cmap, vmin=cbar_min, vmax=cbar_max,
            rasterized=rasterized)
    elif cbar_scale == 'sqrt':
        Z_new = np.sqrt(Z)
        ax.pcolormesh(X, Y, Z_new, cmap=cmap, vmin=cbar_min, vmax=cbar_max,
            rasterized=rasterized)
    else:
        print('Invalid color scale. Choose from: linear, log, arcsinh, sqrt')
        sys.exit()

    if (plt_axes == False) & (save_as == 'None'):
        plt.show()

    if save_as != 'None':
        print('Saving CMD to file...')
        plt.savefig(save_as, format='pdf')


def residual_plot(w=3.5, h=3.5):

    fig = plt.figure(figsize=(w,h), tight_layout=True)
    grid = gridspec.GridSpec(3,1, hspace=0.0)

    ax1 = fig.add_subplot(grid[:-1,:])
    ax2 = fig.add_subplot(grid[-1:,:], sharex=ax1)
    ax1.tick_params(axis='x', direction='in')
    #ax1.set_xticklabels([])
    plt.setp(ax1.get_xticklabels(), visible=False)
    return fig, ax1, ax2


# A helper function to make the plots with error ellipses
def plot_error_ellipses(ax, X, S, color="k"):
    for n in range(len(X)):
        vals, vecs = np.linalg.eig(S[n])
        theta = np.degrees(np.arctan2(*vecs[::-1, 0]))
        w, h = 2 * np.sqrt(vals)
        ell = Ellipse(xy=X[n], width=w, height=h,
                      angle=theta, color=color, lw=0.5)
        ell.set_facecolor("none")
        ax.add_artist(ell)
    ax.plot(X[:, 0], X[:, 1], ".", color=color, ms=4)

# Plot corner plot of 3D data
def plot_3D_data(x, y, z, xerr=None, yerr=None, zerr=None, plt_axes=False,
    error_ellipse=True, invert_x=False, invert_y=False, invert_z=False,
    xlabel='x', ylabel='y', zlabel='z', color='k', plt_fig=False):

    """
    Plot the 2D planes of 3D data in a corner-type plot.

    Required parameters:

    x - array containing the x values
    y - array containing the y values
    z - array containing the z values

    Optional Keywords:


    """

    if plt_axes == False:
        fig, axes = plt.subplots(2,2, figsize=(5,5))
    else:
        axes = plt_axes
        fig = plt_fig
    # if len(plt_axes) > 1:
    #     axes = plt_axes
    #     fig = plt_fig
    # else:
    #     fig, axes = plt.subplots(2,2, figsize=(5,5))

    N = len(x)
    X = np.empty((N, 3))
    X[:,0] = x
    X[:,1] = y
    X[:,2] = z

    # if (xerr == None) & (yerr == None) & (zerr == None):
    #     axes[0,0].scatter(x, y, marker='.', alpha=0.7,
    #         color=color)
    #     axes[1,0].scatter(x, z, marker='.', alpha=0.7,
    #         color=color)
    #     axes[1,1].scatter(y, z, marker='.', alpha=0.7,
    #         color=color)
    if len(xerr) > 0:

        S = np.zeros((N, 3, 3))
        for n in range(N):
            S[n,0,0] = xerr[n]**2
            S[n,1,1] = yerr[n]**2
            S[n,2,2] = zerr[n]**2

        if error_ellipse == True:
            for xi, yi in product(range(3), range(3)):

                if yi <= xi:
                    continue
                ax = axes[yi-1, xi]
                plot_error_ellipses(ax, X[:, [xi, yi]],
                                    S[:,
                                        [[xi, xi], [yi, yi]],
                                        [[xi, yi], [xi, yi]]])
        else:
            axes[0,0].errorbar(x, y, xerr=xerr, yerr=yerr, fmt='.', alpha=0.7,
                color=color, elinewidth=1.0)
            axes[1,0].errorbar(x, z, xerr=xerr, yerr=zerr, fmt='.', alpha=0.7,
                color=color, elinewidth=1.0)
            axes[1,1].errorbar(y, z, xerr=yerr, yerr=zerr, fmt='.', alpha=0.7,
                color=color, elinewidth=1.0)

    # determine plot limits
    if invert_x == True:
        axes[0,0].invert_xaxis()
        axes[1,0].invert_xaxis()
    if invert_y == True:
        axes[0,0].invert_yaxis()
        axes[1,1].invert_xaxis()
    if invert_z == True:
        axes[1,0].invert_yaxis()
        axes[1,1].invert_yaxis()

    # Make the plots look nicer...
    ax = axes[0, 1]
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    # Plot labels
    axes[0,0].set_ylabel(ylabel)
    axes[0,0].set_xticklabels([])
    axes[1,0].set_xlabel(xlabel)
    axes[1,0].set_ylabel(zlabel)
    axes[1,1].set_xlabel(ylabel)
    axes[1,1].set_yticklabels([])
    fig.subplots_adjust(wspace=0, hspace=0)

    return fig, axes


def plot_region(x, y, image, xall=[], yall=[], ext=0,
    axes=None, xoff=0, yoff=0, aperture=None, img_limits=[0,500]):

    image_data = fits.getdata(image, ext=ext)

    if axes is None:
        fig, ax = plt.subplots(1,1, figsize=(8,5))
    else:
        ax = axes

    norm1 = ImageNormalize(image_data, vmin=img_limits[0], vmax=img_limits[1],
        stretch=LogStretch())

    ax.imshow(image_data, cmap='gray', norm=norm1)

    #ax.set_aspect('equal')
    if aperture != None:
        ap = Circle((x-1, y-1), aperture, facecolor=None, edgecolor='red', fill=0)
        ax.add_patch(ap)
    if (len(xall) > 0) & (len(yall) > 0):
        x_all = xall - (xoff+1)
        y_all = yall - (yoff+1)
        ax.scatter(x_all, y_all, marker='x', color='green')
    ax.set_xlim(x-20, x+20)
    ax.set_ylim(y-20, y+20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    if axes == None:
        return fig, ax
    #return fig, ax
