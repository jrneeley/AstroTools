import plot_cmd as p
from astropy.io import fits
import numpy as np


c1 = fits.open('M33-A_RRL-ACSarc_F0_noapcor.fits')
data = c1[1].data

f606w = data['mag_sw'][data['filter'] == 'F606W']
f814w = data['mag_sw'][data['filter'] == 'F814W']

color = f606w - f814w
mag = f606w

mag = mag[~np.isnan(color)]
color = color[~np.isnan(color)]

p.plot_cmd(color, mag)

#p.plot_cmd_density(color, mag)
