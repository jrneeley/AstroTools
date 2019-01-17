import numpy as np

def cardelli(wave, Rv):

    x = 1./wave

    if (x <= 1.1) & (x >= 0.3):

        a = 0.574*x**(1.61)
        b = -0.527*x**(1.61)

        ratio = a + b/Rv
    elif (x > 1.1) & (x <= 3.3):

        y = x - 1.82
        a = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 \
            + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
        b = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 \
            - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7

        ratio = a + b/Rv

    else:
        print 'Outside allowed wavelength range of Cardelli Law.'
        ratio = np.nan

    print 'Extinction ratio Ax/Av = {:.3f}'.format(ratio)
    return ratio
