import numpy as np
from astropy.io import fits
from acstools import acszpt


def get_acs_zp(image, detector='WFC'):

    # get information from image header
    with fits.open(image) as hdul:
        exptime = hdul[0].header['EXPTIME']
        date = hdul[0].header['DATE-OBS']
        filt1 = hdul[0].header['FILTER1']
        filt2 = hdul[0].header['FILTER2']
        if filt1[0] == 'F':
            filt = filt1
        else:
            filt = filt2

    # get ACS zero point
    detector=detector
    q = acszpt.Query(date=date, detector=detector, filt=filt)
    zpt_table = q.fetch()
    zp = zpt_table['VEGAmag'][0].value

    zp_total = -25.0 + zp + 2.5*np.log10(exptime)

    return zp_total

# Do not use, in progress!!! 
def get_acs_apcor(image, filter='None', detector='WFC'):

    # get information from image header
    with fits.open(image) as hdul:
        exptime = hdul[0].header['EXPTIME']
        date = hdul[0].header['DATE-OBS']
        filt1 = hdul[0].header['FILTER1']
        filt2 = hdul[0].header['FILTER2']
        if filt1[0] == 'F':
            filt = filt1
        else:
            filt = filt2

    # get ACS zero point
    detector=detector
    q = acszpt.Query(date=date, detector=detector, filt=filt)
    zpt_table = q.fetch()
    zp = zpt_table['VEGAmag'][0].value


    # get aperture correction (0.5" -> 5.5")
    dt = np.dtype([('filter', 'U6'), ('10', float)])
    ee = np.loadtxt('acs-wfc-ee.txt', dtype=dt, usecols=(0,10), skiprows=1)

    row = ee['filter'] == filt
    apcorr_inf = 2.5*np.log10(ee['10'][row])

    # get PSF to aperture correction (PSF -> ap(0.5"))
    apcorr_0p5 = 0.0


    zp_total = -25.0 + zp + apcorr_inf + apcorr_0p5 + 2.5*np.log10(exptime)

    return zp_total
