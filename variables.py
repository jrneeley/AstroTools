import numpy as np
import matplotlib.pyplot as plt


def compute_variability_index(filters, mjds, mags, errs,
    statistic='WelchStetsonI', max_time=0.02):

    # separate filters
    filter_list = np.unique(filters)
    n_filts = len(filter_list)

    if statistic == 'WelchStetsonI':
    # This is an appropriate index when you have more or less consecutive
    # observations in two filters

        if n_filts == 1:

            # Derive weighted mean from full sample
            mean_mag = weighted_mean(mags, errs)

            # Get observation pairs
            # sort by observation time
            order = np.argsort(mjds)
            m = mags[order]
            e = errs[order]
            j = mjds[order]
            #print(mjds)
            sum = 0
            npairs = 0.0
            skip = False
            for i in range(len(m)-1):
                if skip == True:
                    skip = False
                    continue
                if j[i+1] - j[i] < max_time:
                    delta1 = (m[i]-mean_mag)/e[i]
                    delta2 = (m[i+1]-mean_mag)/e[i]
                    sum += delta1*delta2
                    npairs += 1
                    skip = True

            npairs = float(npairs)
            stat = np.sqrt(1/(npairs*(npairs-1)))*sum

        if n_filts == 2:

            f1 = filters == filter_list[0]
            f2 = filters == filter_list[1]
            mags1 = mags[f1]
            mags2 = mags[f2]
            errs1 = errs[f1]
            errs2 = errs[f2]
            mjds1 = mjds[f1]
            mjds2 = mjds[f2]

            order1 = np.argsort(mjds1)
            order2 = np.argsort(mjds2)

            mags1 = mags1[order1]
            mags2 = mags2[order2]
            errs1 = errs1[order1]
            errs2 = errs2[order2]
            mjds1 = mjds1[order1]
            mjds2 = mjds2[order2]


            mean_mag1 = weighted_mean(mags1, errs1)
            mean_mag2 = weighted_mean(mags2, errs2)

            n1 = len(mags1)
            n2 = len(mags2)
            n = np.array([n1, n2])
            pick = np.argmax(n)

            sum = 0
            i2 = 0
            npairs = 0.0
            for i in range(np.max(n)):
                if pick == 0:
                    if i2 == n2:
                        break
                    if np.abs(mjds1[i] - mjds2[i2]) <= max_time:
                        delta1 = (mags1[i]-mean_mag1)/errs1[i]
                        delta2 = (mags2[i2]-mean_mag2)/errs2[i2]
                        sum += delta1*delta2
                        npairs += 1
                        i2 += 1

                if pick == 1:
                    if i2 == n1:
                        break
                    if np.abs(mjds1[i2] - mjds2[i]) <= max_time:
                        delta1 = (mags1[i2]-mean_mag1)/errs1[i2]
                        delta2 = (mags2[i]-mean_mag2)/errs2[i]
                        sum += delta1*delta2
                        npairs += 1
                        i2 += 1

            stat = np.sqrt(1./(npairs*(npairs-1)))*sum

        return stat

    if statistic == 'StetsonJ':

        order = np.argsort(mjds)
        m = mags[order]
        e = errs[order]
        j = mjds[order]
        fils = filters[order]

        if n_filts == 1:
            wmean = stetson_robust_mean(mags, errs)
            n = float(len(mags))

            P = 0
            n_pairs = 0
            skip_next = False
            for i in range(len(mags)-1):
                if skip_next == True:
                    skip_next = False
                    continue
                if j[i+1] - j[i] <= max_time:
                    delta1 = np.sqrt(n/(n-1))*(m[i] - wmean)/e[i]
                    delta2 = np.sqrt(n/(n-1))*(m[i+1] - wmean)/e[i+1]
                    P += np.sign(delta1*delta2)*np.sqrt(np.abs(delta1*delta2))
                    skip_next = True
                    n_pairs += 1
                else:
                    delta1 = np.sqrt(n/(n-1))*(m[i] - wmean)/e[i]
                    P += np.sign(delta1*delta1-1)*np.sqrt(np.abs(delta1*delta1-1))
                    skip_next = False
                    n_pairs += 1
            stat = P

        if n_filts == 2:

            f1 = filters == filter_list[0]
            wmean1 = stetson_robust_mean(mags[f1], errs[f1])
            n1 = float(len(mags[f1]))
            f2 = filters == filter_list[1]
            wmean2 = stetson_robust_mean(mags[f2], errs[f2])
            n2 = float(len(mags[f2]))

            P = 0
            n_pairs = 0
            skip_next = False
            for i in range(len(mags)-1):

                if skip_next == True:
                    skip_next = False
                    continue
                #print(i, j[i], fils[i])
                #print(i+1, j[i+1], fils[i+1])
                if j[i+1] - j[i] <= max_time:
                    #print(fils[i], fils[i+1])
                    # Check if they are observations in the same or different filter
                    # Same filter, two obs
                    if fils[i+1] == fils[i]:
                        if fils[i] == filter_list[0]:
                            delta1 = np.sqrt(n1/(n1-1))*(m[i] - wmean1)/e[i]
                            delta2 = np.sqrt(n1/(n1-1))*(m[i+1] - wmean1)/e[i+1]
                        else:
                            delta1 = np.sqrt(n2/(n2-1))*(m[i] - wmean2)/e[i]
                            delta2 = np.sqrt(n2/(n2-1))*(m[i+1] - wmean2)/e[i+1]
                        #print(np.sign(delta1*delta2))
                        P += np.sign(delta1*delta2)*np.sqrt(np.abs(delta1*delta2))
                        skip_next = True
                    # Different filters, two obs
                    else:
                        if fils[i] == filter_list[0]:
                            delta1 = np.sqrt(n1/(n1-1))*(m[i] - wmean1)/e[i]
                            delta2 = np.sqrt(n2/(n2-1))*(m[i+1] - wmean2)/e[i+1]
                        else:
                            delta1 = np.sqrt(n2/(n2-1))*(m[i] - wmean2)/e[i]
                            delta2 = np.sqrt(n1/(n1-1))*(m[i+1] - wmean1)/e[i+1]
                        #print(np.sign(delta1*delta2))
                        P += np.sign(delta1*delta2)*np.sqrt(np.abs(delta1*delta2))
                        skip_next = True
                # Single obs
                else:
                    if fils[i] == filter_list[0]:
                        delta1 = np.sqrt(n1/(n1-1))*(m[i] - wmean1)/e[i]
                    else:
                        delta1 = np.sqrt(n2/(n2-1))*(m[i] - wmean2)/e[i]
                    P += np.sign(delta1*delta1-1)*np.sqrt(np.abs(delta1*delta1-1))
                    skip_next = False
                n_pairs += 1

            stat = P

        return stat

    if statistic == 'StetsonK':
        # NOT FINISHED
        wmean = stetson_robust_mean(mags, errs)
        n = float(len(mags))

        delta = np.sqrt(n/(n-1))*(mags-wmean)

    if statistic == 'reduced chisq':

        if n_filts == 1:
            n = float(len(mags))
            mean_mag = weighted_mean(mags, errs)
            stat = 1/n*np.sum((mags-mean_mag)**2/errs**2)

        if n_filts > 1:

            n_tot = 0
            sum = 0
            for i in range(n_filts):
                f = filters == filter_list[i]
                n_tot += float(len(mags[f]))
                mean_mag = weighted_mean(mags[f], errs[f])
                sum += np.sum((mags[f]-mean_mag)**2/errs[f]**2)

            stat = 1./(n_tot-1)*(sum)

        return stat

    if statistic == 'weighted std':

        if n_filts == 1:
            n = float(len(mags))
            w = 1./errs**2
            mean_mag = weighted_mean(mags, errs)
            stat = np.sum(w)*np.sum(w*(mags-mean_mag)**2)/(np.sum(w)**2-np.sum(w**2))
            stat = np.sqrt(stat)

        # should it be added in quadrature?
        #if n_filts > 1:

        return stat

    if statistic == 'MAD':

        if n_filts == 1:
            n = float(len(mags))
            order = np.argsort(mjds)
            m = mags[order]
            e = errs[order]

            med = np.median(m)
            stat = np.median(np.abs(m - med))

        return stat

    if statistic == 'IQR':

        if n_filts == 1:
            n = float(len(mags))
            order = np.argsort(mags)

            m = mags[order]
            e = errs[order]

            q2= np.median(m)
            q1 = np.median(m[0:n/2])
            q3 = np.median(m[n/2:-1])

            stat = q3 - q1

    if statistic == 'RoMS':

        if n_filts == 1:
            n = float(len(mags))
            sum = np.sum(np.abs(mags-np.median(mags))/errs)

            stat = sum/(n-1)

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

def weighted_mean(mags, errs):

    w = 1./errs**2
    mean = np.sum(mags*w)/np.sum(w)

    return mean


def find_variables(index='StetsonJ'):

    # compute variability index for each star
    for i in range(N):
        compute_variability_index(filters, mjds, mags, errs)
