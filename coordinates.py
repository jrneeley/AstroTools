import numpy as np


# Convert coordinates in sexagesimal to decimal units
def hms2deg(ra_h, ra_m, ra_s, dec_d, dec_m, dec_s):

    if type(dec_d) != np.ndarray:
        if dec_d < 0:
            dec = dec_d - dec_m/60. - dec_s/3600.
        if dec_d >= 0 :
            dec = dec_d + dec_m/60. + dec_s/3600.
        ra = 15*(ra_h+ra_m/60.+ra_s/3600.)
    else:
        dec = np.zeros(len(dec_d))
        for ind, d in enumerate(dec_d):
            if dec_d[ind] < 0:
            #    dec.append(dec_d[ind] - dec_m[ind]/60. - dec_s[ind]/3600.)
                dec[ind] = dec_d[ind] - dec_m[ind]/60. - dec_s[ind]/3600.
            if dec_d[ind] >= 0:
            #    dec.append(dec_d[ind] + dec_m[ind]/60. + dec_s[ind]/3600.)
                dec[ind] = dec_d[ind] + dec_m[ind]/60. + dec_s[ind]/3600.
#    if len(ra_h) > 1:
        ra = np.zeros(len(ra_h))
        for ind, r in enumerate(ra_h):
            #ra.append(15.*(ra_h[ind]+ra_m[ind]/60.+ra_s[ind]/3600.))
            ra[ind] = 15.*(ra_h[ind]+ra_m[ind]/60.+ra_s[ind]/3600.)

    return ra, dec

def radec_string2deg(ra, dec):

    if type(ra) != np.ndarray:

        ra_sep = ra.split(':')
        dec_sep = dec.split(':')
        ra_new, dec_new = hms2deg(float(ra_sep[0]), float(ra_sep[1]),
            float(ra_sep[2]), float(dec_sep[0]), float(dec_sep[1]), float(dec_sep[2]))
    else:
        num_stars = len(ra)

        ra_new = np.zeros(num_stars)
        dec_new = np.zeros(num_stars)
        for ind, star in enumerate(ra):

            ra_sep = ra[ind].split(':')
            dec_sep = dec[ind].split(':')
            #print ra_sep, dec_sep
            ra_deg, dec_deg = hms2deg(float(ra_sep[0]), float(ra_sep[1]),
                float(ra_sep[2]), float(dec_sep[0]), float(dec_sep[1]), float(dec_sep[2]))
            ra_new[ind] = ra_deg
            dec_new[ind] = dec_deg

    return ra_new, dec_new

# Finds radial distance between coordinates in arcsec
# second RA/DEC should be a scalar, first can be scalar or array
def radial_dist(ra1, dec1, ra2, dec2):

    ra1 = np.radians(ra1)
    dec1 = np.radians(dec1)
    ra2 = np.radians(ra2)
    dec2 = np.radians(dec2)


    x1 = np.cos(dec1)*np.cos(ra1)
    y1 = np.cos(dec1)*np.sin(ra1)
    z1 = np.sin(dec1)
    x2 = np.cos(dec2)*np.cos(ra2)
    y2 = np.cos(dec2)*np.sin(ra2)
    z2 = np.sin(dec2)

    dist = np.sqrt( (x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    dist = np.degrees(dist)*3600.

    return dist
