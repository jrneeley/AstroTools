import numpy as np
from astropy.stats import sigma_clip

# Calculate the weighted mean, as defined by Peter Stetson
def stetson_robust_mean(mags, errs):

    # As a first guess, calculate the weighted mean the usual way
    weights = 1/errs**2
    initial_guess = np.sum(mags*weights)/np.sum(weights)
    n = len(mags)

    # Iteratively reweight points based on their difference from the weighted
    # mean, and recalculate the weighed mean. Stop when this converges.
    diff = 99
    old_mean = initial_guess
    for i in range(1000):

        delta = np.sqrt(n/(n-1))*(mags-old_mean)/errs
        weight_factor = 1/(1+(np.abs(delta)/2)**2)
        weights = weights*weight_factor

        new_mean = np.sum(mags*weights)/np.sum(weights)

        diff = np.abs(old_mean - new_mean)
        # break the loop if the weighted mean has converged
        if diff < 0.00001:
            break
        old_mean = new_mean

    return new_mean

# Simple function for the standard weighted mean.
def weighted_mean(measurements, errors):

    # Select out indices where magnitude and error are both finite (not nan)
    finite = (~np.isnan(measurements)) & (~np.isnan(errors))
    weights = 1./errors[finite]**2
    sum_weights = np.sum(weights)

    mean = np.sum(measurements[finite]*weights)/sum_weights

    return mean

def weighted_intensity_mean(mags, errs):

    finite = (~np.isnan(mags)) & (~np.isnan(errs))
    flux = 10**(-mags/2.5)
    eflux = flux*errs
    weights = 1./eflux[finite]**2
    sum_weights = np.sum(weights)

    mean_flux = np.sum(flux[finite]*weights)/sum_weights
    mean_mag = -2.5*np.log10(mean_flux)
    return mean_mag

def weighted_stddev(mags, errs):
    ## CHECK

    finite = (~np.isnan(mags)) & (~np.isnan(errs))
    weights = 1./errs[finite]**2
    mean = weighted_intensity_mean(mags, errs)
    sum_weights = np.sum(weights)
    top = np.sum(weights*(mags[finite]-mean)**2)
    num = float(len(mags[finite]))
    bottom = (num-1)*sum_weights/num
    stddev = np.sqrt(top/bottom)

    return stddev

# Helper function to do a binned sigma clip
def binned_sigma_clip(xdata, ydata, bins=10, sigma=3, iters=5):

    # check for and remove nan values

    clipped = np.zeros(len(ydata), dtype=int)
    ind = np.arange(len(ydata))

    # check for and remove nan values
    good = (~np.isnan(ydata))

    std, edges, num = stats.binned_statistic(xdata[good],
        ydata[good], 'std', bins=bins)

    for i in range(bins):

        in_bin = (xdata[good] >= edges[i]) & (xdata[good] < edges[i+1])
        filtered_data = sigma_clip(ydata[good][in_bin], sigma=sigma, maxiters=iters)
        s = ind[good][in_bin]
        clipped[s] = filtered_data.mask*1

    return clipped, edges, std
