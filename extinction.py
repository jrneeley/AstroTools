import numpy as np
import matplotlib.pyplot as plt

# Give central wavelength (in microns) and an Rv value, and the function
# returns the extinction ratio for that band
def cardelli(wave, Rv=3.1, extend=False, verbose=1):

    x = 1./wave

    if (x >= 0.3) & (x <= 1.1): use = 'ir'
    if (x > 1.1) & (x <= 3.3): use = 'opt'
    if (x > 3.3) & (x <= 8.0): use = 'uv'
    if extend == False:
        if (x< 0.3) | (x > 8.0):
            use = 'fail'
    else:
        if x < 0.3: use = 'ir'
        if x > 8.0: use = 'uv'


    # infrared 0.909 - 3.33 microns
    if use =='ir':

        a = 0.574*x**(1.61)
        b = -0.527*x**(1.61)

        ratio = a + b/Rv

    # optical 0.303 - 0.900 microns
    elif use == 'opt':

        y = x - 1.82
        a = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 \
            + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
        b = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 \
            - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7

        ratio = a + b/Rv
    # UV 0.125 - 0.303 microns
    elif use == 'uv':

        if (x <= 8.0) & (x >= 5.9):
            Fa = -0.0447*(x-5.9)**2 - 0.009779*(x-5.9)**3
            Fb = 0.2130*(x-5.9)**2 + 0.1207*(x-5.9)**3
        else:
            Fa = 0.0
            Fb = 0.0

        a = 1.752 - 0.316*x - 0.104/((x-4.67)**2+0.341) + Fa
        b = -3.090 + 1.825*x + 1.206/((x-4.62)**2+0.263) + Fb

        ratio = a + b/Rv

    elif use == 'fail':

        if verbose == 1: print 'Outside allowed wavelength range of Cardelli Law.'
        ratio = np.nan

    if verbose == 1: print 'Extinction ratio Ax/Av = {:.3f}'.format(ratio)
    return ratio

def get_ext_ratio(band, Rv=3.1):

    cardelli_bands = np.array(['U', 'B', 'V', 'R', 'I', 'J', 'H', 'Ks',
        'u', 'g', 'r', 'i', 'z',
        'F435W', 'F475W', 'F555W', 'F606W', 'F775W', 'F814W',
        'G', 'BP', 'RP'])
    indebetouw_bands = np.array(['[3.6]', '[4.5]', '[5.8]', '[8.0]', 'W1', 'W2'])

    if np.isin(band, cardelli_bands):

        # wave is effective wavelength
        # Johnson-Cousins (from Bessel 2005)
        if band == 'U': wave = 0.366 # 0.3663
        if band == 'B': wave = 0.436 # 0.4361
        if band == 'V': wave = 0.545 # 0.5448
        if band == 'R': wave = 0.641 # 0.6407
        if band == 'I': wave = 0.798 # 0.7980
        # 2MASS
        if band == 'J': wave = 1.235
        if band == 'H': wave = 1.662
        if band == 'Ks': wave = 2.159
        # HST ACS/WFC
        if band == 'F435W': wave = 0.432 # 0.4317
        if band == 'F475W': wave = 0.474 # 0.4744
        if band == 'F555W': wave = 0.536 # 0.5360
        if band == 'F606W': wave = 0.592 # 0.5918
        if band == 'F625W': wave = 0.631 # 0.6311
        if band == 'F775W': wave = 0.769 # 0.7693
        if band == 'F814W': wave = 0.806 # 0.8060
        # SDSS (from Bessel 2005)
        if band == 'u': wave = 0.360 # 0.3596
        if band == 'g': wave = 0.464 # 0.4639
        if band == 'r': wave = 0.612 # 0.6122
        if band == 'i': wave = 0.744 # 0.7439
        if band == 'z': wave = 0.890 # 0.8896
        # Gaia
        if band == 'G': wave = 0.673
        if band == 'BP': wave = 0.532
        if band == 'RP': wave = 0.797

        ratio = cardelli(wave, Rv=Rv, verbose=0)
        return ratio

    elif np.isin(band, indebetouw_bands):

        # Spitzer/IRAC
        if band == '[3.6]': wave = 3.545
        if band == '[4.5]': wave = 4.442
        if band == '[5.8]': wave = 5.675
        if band == '[8.0]': wave = 7.760

        # WISE
        if band == 'W1': wave = 3.35
        if band == 'W2': wave = 4.60
        #if band == 'W3': wave = 11.56  # not yet supported
        #if band == 'W4': wave = 22.1  # not yet supported

        akav = cardelli(2.159, Rv=Rv, verbose=0)
        ratio = indebetouw(wave, Ak_Av=akav)
        return ratio

    else:
        print 'ERROR: Band not currently supported.'


def plot_cardelli(Rv=[3.1]):

    wave = np.linspace(0.1, 8.0, 1000)
    wave2 = np.linspace(1.25, 7.75, 100)
    ratio = np.zeros(len(wave))
    ratio_ext = np.zeros(len(wave))
    ratio_ir = np.zeros(len(wave2))

    for ii, w in enumerate(wave):
        ratio[ii] = cardelli(w, Rv=3.1, verbose=0)
        ratio_ext[ii] = cardelli(w, Rv=3.1, verbose=0, extend=True)


    ak_av = cardelli(2.159, verbose=0)
    for ii, w in enumerate(wave2):
        ratio_ir[ii] = indebetouw(w, Ak_Av=ak_av)

    indeb_w = 1.0/np.array([1.235, 1.662, 2.159, 3.6, 4.5, 5.8, 8.0])
    indeb_r = np.array([2.50, 1.55, 1.0, 0.56, 0.43, 0.43, 0.43])*ak_av
    indeb_e = np.array([0.15, 0.08, 0.0, 0.06, 0.08, 0.10, 0.10])*ak_av

    plt.plot(1/wave, ratio_ext, linestyle='--')
    plt.plot(1/wave, ratio)
    plt.plot(1/wave2, ratio_ir, linestyle=':', color='k')
    plt.errorbar(indeb_w, indeb_r, yerr=indeb_e, color='k', fmt='o')
    plt.xlabel('$1/\lambda}$ [$\mu m^{-1}$]')
    plt.ylabel('$A_{\lambda}/A_V$')
    plt.xlim(0,1.0)
    plt.ylim(0,0.6)
    plt.show()


def indebetouw(wave, Ak_Av=0.117):
    # Recommended for 1.25 - 7.75 micron

    log_ratio = 0.61 - 2.22*np.log10(wave) + 1.21*np.log10(wave)**2
    Awave_Av = 10**(log_ratio)*Ak_Av

    return Awave_Av
