import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits



def plot_rrlyrae_template(amplitude=1.0, mode='ab', band='V', plt_axes=False,
    y_offset=0.0, x_offset=0.0, color='k'):

    bands = np.array(['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', '[3.6]', '[4.5]'])

    # Read in template files
    if mode == 'ab':
        c = fits.open('master_template_rrab.fits')
        temp = c[1].data
        cols = c[1].columns
    if mode == 'c':
        c = fits.open('master_template_rrc.fits')
        temp = c[1].data

    index = bands == band

    if plt_axes == False:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
    else: ax = plt_axes

    p = temp['phase'][index,:][0] + x_offset
    m = temp['mag'][index,:][0]*amplitude + y_offset
    ax.plot(p, m, color=color)

    if plt_axes == False:
        ax.set_xlabel('phase')
        ax.set_ylabel('mag')
        ax.set_ylim([0.75*amplitude,-0.75*amplitude])
        plt.show()


def simulate_observing_cadence(times, period=0.5, mode='ab', amplitude=1.0,
    T0=0, band='V'):
# Need to modify this to show multiple T0 values, if it is not known
    phase = np.mod((times-T0)/period, 1)

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)

    plot_rrlyrae_template(mode=mode, band=band, amplitude=amplitude, plt_axes=ax)

    mag = get_template_mags(phase, amplitude=amplitude, mode=mode, band=band)

    ax.plot(phase, mag, 'o', color='xkcd:cadet blue')
    ax.set_xlabel('phase')
    ax.set_ylabel('mag')
    ax.set_ylim(0.75*amplitude,-0.75*amplitude)
    plt.show()

def get_template_mags(phases, amplitude=1.0, mode='ab', band='V'):

    # Read in template files
    if mode == 'ab':
        c = fits.open('master_template_rrab.fits')
        temp = c[1].data
        cols = c[1].columns
    if mode == 'c':
        c = fits.open('master_template_rrc.fits')
        temp = c[1].data
    bands = np.array(['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', '[3.6]', '[4.5]'])

    # Find magnitude of closes phase for appropriate filter
    index = bands == band
    mags = np.interp(phases, temp['phase'][index,:][0], temp['mag'][index,:][0])
    mags *= amplitude

    return mags

def simulate_exposure_time(exptime, period, amplitude, mean=0.0, mode='ab', band='V'):

    # Read in template files
    if mode == 'ab':
        c = fits.open('master_template_rrab.fits')
        temp = c[1].data
        cols = c[1].columns
    if mode == 'c':
        c = fits.open('master_template_rrc.fits')
        temp = c[1].data
    bands = np.array(['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', '[3.6]', '[4.5]'])

    # Find magnitude of closes phase for appropriate filter
    index = bands == band
    phase = temp['phase'][index,:][0]
    mag = temp['mag'][index, :][0]*amplitude + mean

    # convert phase to time and copy into multiple cycles
    t = phase*period
    phase_copy = np.concatenate((phase-1.0, phase, phase+1.0))
    t_copy = phase_copy*period
    mag_copy = np.tile(mag, 3)
    # convert magnitude to flux
    flux = 9.9*np.power(10, -mag/2.5)
    flux_copy = 9.9*np.power(10, -mag_copy/2.5)

    new_mag = np.zeros(len(mag))
    measured_flux = np.zeros(len(mag))

    for ii, x in enumerate(t):
        exptime_days = exptime/86400.
        min_t = x - exptime_days/2.0
        max_t = x + exptime_days/2.0
        window_t = t_copy[(t_copy >= min_t) & (t_copy <= max_t)]
        window_flux = flux_copy[(t_copy >= min_t) & (t_copy <= max_t)]
        measured_flux[ii] = np.mean(window_flux)
        new_mag[ii] = -2.5*np.log10(measured_flux[ii]/9.9)

    old_mean_mag = -2.5*np.log10(np.mean(flux)/9.9)
    new_mean_mag = -2.5*np.log10(np.mean(measured_flux)/9.9)
    delta_amp = (np.max(mag) - np.min(mag)) - (np.max(new_mag) - np.min(new_mag))
    delta_mag = old_mean_mag - new_mean_mag
    #print amplitude, np.max(mag)-np.min(mag), np.max(new_mag) - np.min(new_mag)
    return phase, new_mag, delta_amp, delta_mag
