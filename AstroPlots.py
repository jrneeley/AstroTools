import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde



def plot_cmd(color, mag, xlim=[-1,4], ylim=[20,30]):


    Z, xedges, yedges = np.histogram2d(color,mag,bins=(500,1000), \
        range=[xlim, ylim])

    Y, X = np.meshgrid(yedges, xedges)

    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(111)
    ax.pcolormesh(X, Y, Z)

    #Z2 = np.arcsinh(Z)
    #cntr = plt.imshow(Z, cmap=plt.cm.jet, vmax=50, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    #cntr.cmap.set_under('white')
    #plt.ylim(30,20)
    plt.show()

def plot_cmd_density(color, mag):


    # Calculate the point density
    xy = np.vstack([color,mag])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    color, mag, z = color[idx], mag[idx], z[idx]

    fig, ax = plt.subplots()
    ax.scatter(color, mag, c=z, s=1, edgecolor='')
    ax.set_ylim(30,20)
    plt.colorbar()
    plt.show()
