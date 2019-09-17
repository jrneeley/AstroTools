import numpy as np
import matplotlib.pyplot as plt

MAX_TIME = 0.02 # Requires observations to be within 0.02 days (or 30 min)
                # of each other to form a pair


def compute_variability_index(filters, mjds, mags, errs,
    statistic='WelchStetsonI'):

    # separate filters
    filter_list = np.unique(filters)
    n_filts = len(filter_list)

    if statistic == 'WelchStetsonI':
    # This is an appropriate index when you have more or less consecutive
    # observations in two filters

        if n_filts == 1:
            # Derive weighted mean from full sample
            mean_mag = np.sum(mags/errs**2)/np.sum(1/errs**2)

            # Get observation pairs
            # sort by observation time
            order = np.argsort(mjds)
            mags = mags[order]
            errs = errs[order]
            mjds = mjds[order]

            group1_mags = np.array([])
            group1_errs = np.array([])
            group2_mags = np.array([])
            group2_errs = np.array([])
            skip = False
            for i in range(len(mags)-1):
                if skip == True:
                    skip = False
                    continue
                if mjds[i+1] - mjds[i] < MAX_TIME:
                    group1_mags = np.append(group1_mags, mags[i])
                    group2_mags = np.append(group2_mags, mags[i+1])
                    group1_errs = np.append(group1_errs, errs[i])
                    group2_errs = np.append(group2_errs, errs[i+1])
                    skip = True


            n = len(group1_mags)
            delta1 = np.sum((group1_mags-mean_mag)/group1_errs)
            delta2 = np.sum((group2_mags-mean_mag)/group2_errs)

            stat = np.sqrt(1/(n*(n-1)))*delta1*delta2

        if n_filts == 2:

            filt1 = filters == filter_list[0]
            filt2 = filters == filter_list[1]
            group1_mags = mags[filt1]
            group2_mags = mags[filt2]
            group1_errs = errs[filt1]
            group2_errs = errs[filt2]
            group1_mjds = mjds[filt1]
            group2_mjds = mjds[filt2]

            order1 = np.argsort(group1_mjds)
            order2 = np.argsort(group2_mjds)

            group1_mags = group1_mags[order1]
            group2_mags = group2_mags[order2]
            group1_errs = group1_errs[order1]
            group2_errs = group2_errs[order2]
            group1_mjds = group1_mjds[order1]
            group2_mjds = group2_mjds[order2]


            group1_mean_mag = np.sum(group1_mags/group1_errs**2)/np.sum(1/group1_errs**2)
            group2_mean_mag = np.sum(group2_mags/group2_errs**2)/np.sum(1/group2_errs**2)

            n1 = len(group1_mags)
            n2 = len(group2_mags)
            n = np.array([n1, n2])
            pick = np.argmax(n)

            i2 = 0
            for i in range(np.max(n)):
                if pick == 0:
                    if np.abs(group1_mjds[i] - group2_mjds[i2]) > MAX_TIME:
                        group1_mags[i] = np.nan
                        group1_errs[i] = np.nan
                        group1_mjds[i] = np.nan
                    else:
                        i2 += 1
                if pick == 1:
                    if np.abs(group1_mjds[i2] - group2_mjds[i]) > MAX_TIME:
                        group2_mags[i] = np.nan
                        group2_errs[i] = np.nan
                        group2_mjds[i] = np.nan
                    else:
                        i2 += 1
            group1_mags = group1_mags[~np.isnan(group1_mags)]
            group2_mags = group2_mags[~np.isnan(group2_mags)]
            group1_errs = group1_errs[~np.isnan(group1_errs)]
            group2_errs = group2_errs[~np.isnan(group2_errs)]


            nfinal = len(group1_mags)
            delta1 = np.sum((group1_mags-group1_mean_mag)/group1_errs)
            delta2 = np.sum((group2_mags-group2_mean_mag)/group2_errs)
            stat = np.sqrt(1/(nfinal*(nfinal-1)))*delta1*delta2

        return stat

    if statistic == 'StetsonJ':

        weighted_mean = stetson_robust_mean(mags, errs)
        n = len(mags)
        select = filters == filter_list[0]
        weighted_mean1 = stetson_robust_mean(mags[select], errs[select])
        n1 = len(mags[select])
        if n_filts > 1:
            select = filters == filter_list[1]
            weighted_mean2 = stetson_robust_mean(mags[select], errs[select])
            n2 = len(mags[select])

        order = np.argsort(mjds)
        mags = mags[order]
        errs = errs[order]
        mjds = mjds[order]
        fils = filters[order]

        P = 0
        n_pairs = 0
        skip_next = False
        for i in range(len(mags)-1):
            if skip_next == True:
                skip_next = False
                continue
            if mjds[i+1] - mjds[i] < MAX_TIME:
                # Check if they are observations in the same or different filter
                if fils[i+1] == fils[i]:
                    delta1 = np.sqrt(n/(n-1))*(mags[i] - weighted_mean)/errs[i]
                    delta2 = np.sqrt(n/(n-1))*(mags[i+1] - weighted_mean)/errs[i+1]
                    P += np.sign(delta1*delta2)*np.sqrt(np.abs(delta1*delta2))
                    skip_next = True
                else:
                    if fils[i] == filter_list[0]:
                        delta1 = np.sqrt(n1/(n1-1))*(mags[i] - weighted_mean1)/errs[i]
                        delta2 = np.sqrt(n2/(n2-1))*(mags[i+1] - weighted_mean2)/errs[i+1]
                        P += np.sign(delta1*delta2)*np.sqrt(np.abs(delta1*delta2))
                        skip_next = True
                    else:
                        delta1 = np.sqrt(n2/(n2-1))*(mags[i] - weighted_mean2)/errs[i]
                        delta2 = np.sqrt(n1/(n1-1))*(mags[i+1] - weighted_mean1)/errs[i+1]
                        P += np.sign(delta1*delta2)*np.sqrt(np.abs(delta1*delta2))
                        skip_next = True
            else:
                if fils[i] == filter_list[0]:
                    delta1 = np.sqrt(n1/(n1-1))*(mags[i] - weighted_mean1)/errs[i]
                    P += np.sign(delta1*delta1-1)*np.sqrt(np.abs(delta1*delta1-1))
                    skip_next = False
                else:
                    delta1 = np.sqrt(n2/(n2-1))*(mags[i] - weighted_mean2)/errs[i]
                    P += np.sign(delta1*delta1-1)*np.sqrt(np.abs(delta1*delta1-1))
                    skip_next = False
            n_pairs += 1

        stat = P

        return stat

    if statistic == 'StetsonK':
        # NOT FINISHED
        weighted_mean = stetson_robust_mean(mags, errs)
        n = len(mags)

        delta = np.sqrt(n/(n-1))*(mags-weighted_mean)

    if statistic == 'reduced chisq':

        if n_filts == 1:
            n = len(mags)
            mean_mag = np.mean(mags)
            stat = 1/n*np.sum((mags-mean_mag)**2/errs**2)

        if n_filts == 2:
            f1 = filters == filter_list[0]
            f2 = filters == filter_list[1]
            n_tot = len(mags[f1]) + len(mags[f2])
            mean_mag1 = np.mean(mags[f1])
            mean_mag2 = np.mean(mags[f2])
            sum1 = np.sum((mags[f1]-mean_mag1)**2/errs[f1]**2)
            sum2 = np.sum((mags[f2]-mean_mag2)**2/errs[f2]**2)
            stat = 1/n_tot*(sum1+sum2)

        return stat



def stetson_robust_mean(mags, errs):

    # calculate the simple weighted mean
    weights = 1/errs**2
    initial_guess = np.sum(mags*weights)/np.sum(weights)
    n = len(mags)

    diff = 99
    old_mean = initial_guess
    for i in range(1000):

        delta = np.sqrt(n/(n-1))*(mags-old_mean)/errs
        weight_factor = 1/(1+(np.abs(delta)/2)**2)
        weights = weights*weight_factor

        new_mean = np.sum(mags*weights)/np.sum(weights)

        diff = np.abs(old_mean - new_mean)
        if diff < 0.00001:
            break
        old_mean = new_mean

    return new_mean

def find_variables(index='StetsonJ'):

    # compute variability index for each star
    for i in range(N):
        compute_variability_index(filters, mjds, mags, errs)
