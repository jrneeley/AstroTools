import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# Give central wavelength (in microns) and an Rv value, and the function
# returns the extinction ratio for that band
def cardelli(wave, Rv=3.1, extend=False, verbose=0, output='Ax/Av'):
    ''' Reddening curve of Cardelli, Clayton, and Mathis (1989 ApJ 345, 245).
    Valid from 0.125 to 3.5 microns, but you can choose to extend the relation
    beyond these limits.

    Returns Ax/Av by default, but you can change to Ax/E(B-V). '''

    if np.isscalar(wave): wave = np.array([wave])

    x = 1./wave
    N = len(x)
    ratio = np.zeros(N)

    for ii in range(N):

        if (x[ii] >= 0.3) & (x[ii] <= 1.1): use = 'ir'
        if (x[ii] > 1.1) & (x[ii] <= 3.3): use = 'opt'
        if (x[ii] > 3.3) & (x[ii] <= 8.0): use = 'uv'
        if extend == False:
            if (x[ii] < 0.3) | (x[ii] > 8.0):
                use = 'fail'
        else:
            if x[ii] < 0.3: use = 'ir'
            if x[ii] > 8.0: use = 'uv'

        # infrared 0.909 - 3.33 microns
        if use =='ir':

            a = 0.574*x[ii]**(1.61)
            b = -0.527*x[ii]**(1.61)

            ratio[ii] = a + b/Rv

        # optical 0.303 - 0.900 microns
        elif use == 'opt':

            y = x[ii] - 1.82
            a = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 \
                + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
            b = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 \
                - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7

            ratio[ii] = a + b/Rv

        # UV 0.125 - 0.303 microns
        elif use == 'uv':

            a = 1.752 - 0.316*x[ii] - 0.104/((x[ii]-4.67)**2+0.341)
            b = -3.090 + 1.825*x[ii] + 1.206/((x[ii]-4.62)**2+0.263)

            if (x[ii] <= 8.0) & (x[ii] >= 5.9):
                Fa = -0.04473*(x[ii]-5.9)**2 - 0.009779*(x[ii]-5.9)**3
                Fb = 0.2130*(x[ii]-5.9)**2 + 0.1207*(x[ii]-5.9)**3
                a += Fa
                b += Fb

            ratio[ii] = a + b/Rv

        elif use == 'fail':

            if verbose == 1:
                print('Outside allowed wavelength range of Cardelli Law.')
            ratio[ii] = np.nan

        if verbose == 1:
            print('Ax/Av = {:.3f} for x = {:.3f}'.format(ratio[ii], wave[ii]))

    if output == 'Ax/Av': return ratio
    if output == 'Ax/ebv': return ratio*Rv

# Give central wavelength (in microns) and an Rv value, and the function
# returns the extinction ratio for that band
def cardelli_odonnell(wave, Rv=3.1, extend=False, verbose=0, output='Ax/Av'):
    ''' Reddening curve of Cardelli, Clayton, and Mathis (1989 ApJ 345, 245), but
    updated in the 0.3 to 0.9 micron range with O'Donnell (1994 ApJ 422 158).
    Valid from 0.125 to 3.5 microns, but you can choose to extend the relation
    beyond these limits.

    Returns Ax/Av by default, but you can change to Ax/E(B-V). '''

    if np.isscalar(wave): wave = np.array([wave])

    x = 1./wave
    N = len(x)
    ratio = np.zeros(N)

    for ii in range(N):

        if (x[ii] >= 0.3) & (x[ii] <= 1.1): use = 'ir'
        if (x[ii] > 1.1) & (x[ii] <= 3.3): use = 'opt'
        if (x[ii] > 3.3) & (x[ii] <= 8.0): use = 'uv'
        if extend == False:
            if (x[ii] < 0.3) | (x[ii] > 8.0):
                use = 'fail'
        else:
            if x[ii] < 0.3: use = 'ir'
            if x[ii] > 8.0: use = 'uv'


        # infrared 0.909 - 3.33 microns
        if use =='ir':

            a = 0.574*x[ii]**(1.61)
            b = -0.527*x[ii]**(1.61)

            ratio[ii] = a + b/Rv

        # optical 0.303 - 0.900 microns
        elif use == 'opt':

            y = x[ii] - 1.82
            a = 1 + 0.104*y - 0.609*y**2 + 0.701*y**3 + 1.137*y**4 \
                - 1.718*y**5 - 0.827*y**6 + 1.647*y**7 - 0.505*y**8
            b = 1.952*y + 2.908*y**2 - 3.989*y**3 - 7.985*y**4 \
                + 11.102*y**5 + 5.491*y**6 - 10.805*y**7 + 3.347*y**8

            ratio[ii] = a + b/Rv

        # UV 0.125 - 0.303 microns
        elif use == 'uv':

            a = 1.752 - 0.316*x[ii] - 0.104/((x[ii]-4.67)**2+0.341)
            b = -3.090 + 1.825*x[ii] + 1.206/((x[ii]-4.62)**2+0.263)

            if (x[ii] <= 8.0) & (x[ii] >= 5.9):
                Fa = -0.04473*(x[ii]-5.9)**2 - 0.009779*(x[ii]-5.9)**3
                Fb = 0.2130*(x[ii]-5.9)**2 + 0.1207*(x[ii]-5.9)**3
                a += Fa
                b += Fb

            ratio[ii] = a + b/Rv

        elif use == 'fail':

            if verbose == 1: print('Outside allowed wavelength range of Cardelli Law.')
            ratio[ii] = np.nan

        if verbose == 1: print('Ax/Av = {:.3f} for x = {:.3f}'.format(ratio[ii], wave[ii]))

    if output == 'Ax/Av': return ratio
    if output == 'Ax/ebv': return ratio*Rv


def get_ext_ratio(band, Rv=3.1, red_law='Cardelli', verbose=0, output='Ax/Av'):
    ''' Allowed reddening laws are 'Cardelli', 'ODonnell', 'Fitzpatrick' '''

    if np.isscalar(band): band = np.array([band])
    N = len(band)
    R = np.zeros(N)
    wave = np.zeros(N)

    named_opt_bands = np.array([
        'U', 'B', 'V', 'R', 'I',
        'J', 'H', 'Ks',
        'sloan u', 'sloan g', 'sloan r', 'sloan i', 'sloan z',
        'PS g', 'PS r', 'PS i', 'PS z', 'PS y',
        'ACS F435W', 'ACS F475W', 'ACS F555W', 'ACS F606W', 'ACS F775W', 'ACS F814W', 'X',
        'Gaia G', 'Gaia BP', 'Gaia RP'
        ])
    named_mir_bands = np.array([
        '[3.6]', '[4.5]', '[5.8]', '[8.0]',
        'W1', 'W2'
        ])

    opt_waves = np.array([
        0.366, 0.436, 0.545, 0.641, 0.798,               # Johnson - Cousins
        1.235, 1.662, 2.159,                             # 2MASS
        0.360, 0.464, 0.612, 0.744, 0.890,               # sloan
        0.481, 0.617, 0.752, 0.866, 0.962,               # PanSTARRS
        0.432, 0.474, 0.536, 0.592, 0.631, 0.769, 0.806, # HST ACS/WFC
        0.673, 0.532, 0.797                              # Gaia DR2
        ])
    mir_waves = np.array([
        3.545, 4.442, 5.675, 7.760,                      # IRAC
        3.35, 4.60                                       # WISE
        ])

    for ii in range(N):

        if np.isin(band[ii], named_mir_bands):

            wave[ii] = mir_waves[np.where(named_mir_bands == band[ii])]
            if (red_law == 'Cardelli') | (red_law == 'ODonnell'):
                ak = cardelli(2.159, Rv=Rv, output='Ax/Av')
                R[ii] = indebetouw(wave[ii], Ak_Av=ak)

            elif red_law == 'Fitzpatrick':
                ak = fitzpatrick(2.159, R=Rv, output='Ax/Av')
                R[ii] = indebetouw(wave[ii], Ak_Av=ak)

            else:
                print('ERROR! Unsupported reddening law.')
                R[ii] = np.nan

        elif np.isin(band[ii], named_opt_bands):

            wave[ii] = opt_waves[np.where(named_opt_bands == band[ii])]

            if red_law == 'Cardelli':
                R[ii] = cardelli(wave[ii], Rv=Rv, output='Ax/Av')
            elif red_law == 'ODonnell':
                R[ii] = cardelli_odonnell(wave[ii], Rv=Rv, output='Ax/Av')
            elif red_law == 'Fitzpatrick':
                R[ii] = fitzpatrick(wave[ii], R=Rv, output='Ax/Av')
            else:
                print('ERROR! Unsupported reddening law.')
                R[ii] = np.nan
        else:
            print('ERROR! Band not currently supported.')
            R[ii] = np.nan

        if output == 'Ax/ebv': R[ii] *= Rv
        if verbose == 1:
            if output == 'Ax/Av':
                print('A({})/Av = {:.3f};  lamb = {}'.format(band[ii], R[ii], wave[ii]))
            if output == 'Ax/ebv':
                print('A({})/E(B-V) = {:.3f};  lamb = {}'.format(band[ii], R[ii], wave[ii]))

    return R, wave




def indebetouw(wave, Ak_Av=0.117):
    # Recommended for 1.25 - 7.75 micron

    log_ratio = 0.61 - 2.22*np.log10(wave) + 1.21*np.log10(wave)**2
    Awave_Av = 10**(log_ratio)*Ak_Av

    return Awave_Av


def fitzpatrick(wave, R=3.1, fixed=0, verbose=0, output='Ax/ebv'):

    if np.isscalar(wave): wave = np.array([wave])

    x = 1./wave
    N = len(x)
    R_new = np.zeros(N)


    c2 = -0.824 + 4.717/R
    c1 = 2.030 - 3.007*c2
    x0 = 4.596
    gamma = 0.99
    c3 = 3.23
    c4 = 0.41

    for ii in range(N):

        if wave[ii] <= 0.27:

            D = x[ii]**2/((x[ii]**2-x0**2)**2 + x[ii]**2*gamma**2)
            k = c1 + c2*x[ii] + c3*D
            if x[ii] >= 5.9:
                F = 0.5392*(x[ii]-5.9)**2 + 0.05644*(x[ii]-5.9)**3
                k += c4*F

            R_new[ii] = k + R

        else:
            if fixed == 1:

                x_anchor = np.array([0.000, 0.377, 0.820, 1.667, 1.828, 2.141, 2.433, 3.704, 3.846])
                y_anchor = np.array([0.000, 0.265, 0.829, 2.688, 3.055, 3.806, 4.315, 6.265, 6.591])

                tck = interpolate.splrep(x_anchor, y_anchor, s=0)
                R_new[ii] = interpolate.splev(x[ii], tck, der=0)

            if fixed == 0:

                # Use spline interpolation
                # Figure out where these spline anchor oringate from. Got them from
                # IDL procedure FM_UNRED
                x_anchor = np.array([0.0, 0.3774, 0.8197, 1.6667, 1.8282, 2.1413, 2.4331, 3.7037, 3.8462])
                y_anchor = np.zeros(len(x_anchor))
                y_anchor[0] = 0.0
                y_anchor[1] = 0.26469*R/3.1
                y_anchor[2] = 0.82925*R/3.1
                y_anchor[3] = -4.22809e-1 + 1.00270*R + 2.13572e-4*R**2
                y_anchor[4] = -5.13540e-2 + 1.00216*R - 7.35778e-5*R**2
                y_anchor[5] = 7.00127e-1 + 1.00184*R - 3.32598e-5*R**2
                y_anchor[6] = 1.19459 + 1.01707*R - 5.46959e-3*R**2 + 7.97809e-4*R**3 - 4.45636e-5*R**4
                D = x_anchor[7:]**2/((x_anchor[7:]**2-x0**2)**2 + x_anchor[7:]**2*gamma**2)
                y_anchor[7] = c1 + c2*x_anchor[7] + c3*D[0] + R
                y_anchor[8] = c1 + c2*x_anchor[8] + c3*D[1] + R

                tck = interpolate.splrep(x_anchor, y_anchor, s=0)
                R_new[ii] = interpolate.splev(x[ii], tck, der=0)

        if verbose == 1: print('Ax/ebv = {:.3f} for x = {:.3f}'.format(R_new[ii], wave[ii]))
    if output == 'Ax/ebv': return R_new
    if output == 'Ax/Av': return R_new/R


def wang(wave, Rv=3.1):

    if np.isscalar(wave): wave = np.array([wave])

    x = 1./wave
    N = len(x)
    R = np.zeros(N)

    for ii in range(N):

        if (wave[ii] > 0.3) & (wave[ii] < 1.0):
            y = x[ii] - 1.82
            R[ii] = 1.0 + 0.7527*y - 0.008444*y**2 + 0.2523*y**3 \
                + 0.4619*y**4 - 0.04605*y**5 - 0.5847*y**6 - 0.3443*y**7
        elif (wave[ii] >= 1.0) & (wave[ii] < 3.33):
            R[ii] = 0.3722*wave[ii]**(-2.070)

        else:
            R[ii] = np.nan

    return R
