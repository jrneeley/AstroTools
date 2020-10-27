import numpy as np
import matplotlib.pyplot as plt
import sys
from . import config
from . import math_tools
from astropy.io import fits
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
#from astropy.visualization import LogStretch, ImageNormalize, PercentileInterval
from . import AstroPlots as ap
#from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec


np.warnings.filterwarnings('ignore')

def compute_variability_index(filters, mjds, mags, errs,
    statistic='WelchStetsonI', max_time=0.02):
    # Currently doesn't allow nans for all indices

    # separate filters
    filter_list = np.unique(filters)
    n_filts = len(filter_list)
    # How many total observations (in all filters) do we have?
    num_obs_total = len(mags)

    if statistic == 'WelchStetsonI':

    ## NOT FINISHED

    # This is an appropriate index when you have more or less consecutive
    # observations in one or two filters

        if n_filts == 1:

            # Derive weighted mean from full sample
            mean_mag = math_tools.weighted_mean(mags, errs)

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


            mean_mag1 = math_tools.weighted_mean(mags1, errs1)
            mean_mag2 = math_tools.weighted_mean(mags2, errs2)

            n1 = len(mags1)
            n2 = len(mags2)
            n = np.array([n1, n2])
            pick = np.argmax(n)


            tot = 0
            npairs = 0.0
            for i in range(np.max(n)):
                if pick == 0:
                    time_diff = mjds1[i] - mjds2
                    match = np.argwhere(np.abs(time_diff) <= max_time)

                    if len(mjds2[match]) > 0:
                        delta1 = (mags1[i]-mean_mag1)/errs1[i]
                        delta2 = (mags2[match][0]-mean_mag2)/errs2[match][0]
                        tot += delta1*delta2
                        npairs += 1

                if pick == 1:
                    time_diff = mjds1 - mjds2[i]
                    match = np.argwhere(np.abs(time_diff) <= max_time)

                    if len(mjds1[match] > 0):
                        delta1 = (mags1[match][0]-mean_mag1)/errs1[match][0]
                        delta2 = (mags2[i]-mean_mag2)/errs2[i]
                        tot += delta1*delta2
                        npairs += 1

            if npairs > 2:
                stat = np.sqrt(1./(npairs*(npairs-1)))*tot
            else:
                stat = np.nan

    if statistic == 'StetsonJ':

        # set up arrays for the weighted mean, the number of observations in each
        # filter, and the filter number of each observation (integer for indexing)
        weighted_means = np.zeros(n_filts)
        num_obs = np.zeros(n_filts)
        filter_num = np.zeros(num_obs_total)

        for i in range(n_filts):
            # select out only the observations in this filter.
            # NOTE: This creates a boolean
            # array of the same length as filters, where the elements are true
            # if it satisfies the condition. It can be used to index another array -
            # I do this a lot in python, but don't remember if you can do the same
            # in IDL without using the where() function
            f = filters == filter_list[i]
            # compute the weighted mean in each filter
            weighted_means[i] = math_tools.stetson_robust_mean(mags[f], errs[f])
            num_obs[i] = float(len(mags[f]))
            filter_num[f] = i

        order = np.argsort(mjds)
        mags_temp = mags[order]
        errs_temp = errs[order]
        mjds_temp = mjds[order]
        filt_temp = filter_num[order]


        P = 0
        n_pairs = 0
        skip_next = False
        for i in range(num_obs_total-1):
            # If skip_next == True, then this observation has already been counted
            # in a pair, so change it back to False and move on to the next
            # iteration of the loop
            if skip_next == True:
                skip_next = False
                continue

            # Check if the current observation and the next one were taken close
            # together in time. If they are within your maximum time difference,
            # count them as a pair
            if mjds_temp[i+1] - mjds_temp[i] <= max_time:

                # Check which filters the observations in our pair were taken in, so
                # we compare them to the appropriate weighted mean.
                # This allows for the possibility that these two observations are
                # from the same or different filters
                fnum1 = int(filt_temp[i])
                fnum2 = int(filt_temp[i+1])

                temp1 = (mags_temp[i] - weighted_means[fnum1])/errs_temp[i]
                delta1 = np.sqrt(num_obs[fnum1]/(num_obs[fnum1]-1))*temp1

                temp2 = (mags_temp[i+1] - weighted_means[fnum2])/errs_temp[i+1]
                delta2 = np.sqrt(num_obs[fnum2]/(num_obs[fnum2]-1))*temp2
                # Stetson math
                P += np.sign(delta1*delta2)*np.sqrt(np.abs(delta1*delta2))
                # We paired observation i and i+1, so we will need to skip the
                # next iteration
                skip_next = True


            # This observation is not part of a pair, (could be an isolated
            # observation or part of a grouping of an odd nubmer of observations)
            # and is now treated as a single observation.
            else:
                fnum = int(filt_temp[i])
                temp = (mags_temp[i] - weighted_means[fnum])/errs_temp[i]
                delta = np.sqrt(num_obs[fnum]/(num_obs[fnum]-1))*temp

                P += np.sign(delta*delta-1)*np.sqrt(np.abs(delta*delta-1))
                skip_next = False

            n_pairs += 1

            stat = P

    if statistic == 'StetsonK':
        # NOT FINISHED
        wmean = math_tools.stetson_robust_mean(mags, errs)
        n = float(len(mags))

        delta = np.sqrt(n/(n-1))*(mags-wmean)

    if statistic == 'reduced chisq':

        sum = 0

        # Loop through all filters, but a single chi squared using observations
        # in all filters is computed in the end
        for i in range(n_filts):
            f = filters == filter_list[i]
            # Let's use traditional weighted mean this time.
            weighted_mean_mag = math_tools.weighted_mean(mags[f], errs[f])
            # Use += so we can combine information from different filters
            sum += np.sum((mags[f]-weighted_mean_mag)**2/errs[f]**2)

        chi_squared = 1./(float(num_obs_total)-1)*sum

        stat = chi_squared

    if statistic == 'weighted std':

        weighted_mean_mags = np.zeros(num_obs_total)
        weights = 1./errs**2

        for i in range(n_filts):
            f = filters == filter_list[i]
            weighted_mean_mag[f] = math_tools.weighted_mean(mags[f], errs[f])

        stat_num = np.sum(weights)*np.sum(weights*(mags-weighted_mean_mag)**2)
        stat_den = np.sum(weights)**2-np.sum(weights**2)
        stat = np.sqrt(stat_num/stat_dem)

    if statistic == 'MAD':

        ######## Calculate median absolute deviation (MAD) #########

        # set up empty array for median magnitude. This array has same length as
        # mags, but each element will be the median of all magnitudes of the
        # corresponding filter.
        median_mags = np.zeros(num_obs_total)

        for i in range(n_filts):
            f = filters == filter_list[i]
            # get the median magnitude in this filter, and copy it into an array,
            # whose corresponding elements in mags are the same filter.
            median_mags[f] = np.nanmedian(mags[f])

        absolute_deviation = np.abs(mags - median_mags)
        mad = np.nanmedian(absolute_deviation)

        stat = mad

    if statistic == 'IQR':
        # NOT FINISHED

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

        ######## Calculate Robust median statistic  (RoMS) ########

        sum = 0
        for i in range(n_filts):
            # get all observations in this filter
            f = filters == filter_list[i]
            # Use += so we can combine observations from different filters.
            sum += np.sum(np.abs(mags[f] - np.median(mags[f]))/errs[f])

        # normalize by the total number of observations
        RoMS = sum/(float(num_obs_total)-1)

        stat = RoMS

    return stat


# Classification script
def classify_variable(VAR_FILE, PHOT_FILE, star_id, update=False, plot_lmc=False,
    rrd_fu=False, LASTID_FILE='lastid.txt', band1='M1', band2='M2',
    DM=0, cep=True, rrl=True, image=False, xoff=0, yoff=0, lcv_dir='lcvs/fitlc/',
    img_limits=[0,500]):


    # Load list of variables and types to check
    dt = np.dtype([('id', int), ('cat_id', int), ('type', 'U4'),
        ('subtype', 'U4'), ('x', float), ('y', float), ('period', float),
        ('t0', float), ('mag1', float), ('err1', float), ('amp1', float),
        ('mag2', float), ('err2', float), ('amp2', float)])
    var_list = np.loadtxt(VAR_FILE, dtype=dt)
    n_vars = len(var_list['id'])
    var_list2 = np.copy(var_list)
    vari = var_list['type'] != 'NV'

    # Load photometry file
    dt = np.dtype([('id', int), ('x', float), ('y', float), ('mag1', float),
        ('mag2', float), ('sharp', float)])
    allstars = np.loadtxt(PHOT_FILE, dtype=dt, usecols=(0,1,2,3,5,10))
    #allstars = np.loadtxt(PHOT_FILE, dtype=dt, usecols=(0,3,5,8))

    sel = np.abs(allstars['sharp']) < 0.1
    allstars['mag1'][allstars['mag1'] > 90] = np.nan
    allstars['mag2'][allstars['mag2'] > 90] = np.nan

    # Find index of last plotted variable
    f = open(LASTID_FILE,'r')
    last_id = f.read()
    last_id_index = np.argwhere(var_list['id'] == int(last_id[:-1]))[0]
    f.close()
    ### TO DO: fails if last id is not in variable list. Fix this

    # Find index of next variable to plot
    #plot_next_id = [0]
    if star_id == 'First':
        i = 0
        plot_next_id = var_list['id'][i]
    elif star_id == 'Last':
        i = -1
        plot_next_id = var_list['id'][i]
    elif star_id == 'Same':
        i = last_id_index[0]
        plot_next_id = var_list['id'][i]
    elif star_id == 'Prev':
        if last_id_index[0] == 0:
            print('---> Last id already the first variable.')
            return
        else:
            i = last_id_index[0]-1
            plot_next_id = var_list['id'][i]

    elif star_id == 'Next':
        if last_id_index[0] > n_vars-2:
            print('---> Last id already the last variable.')
            return
        else:
            i = last_id_index[0]+1
            plot_next_id = var_list['id'][i]

    else:
        i = np.argwhere(var_list['id'] == int(star_id))[0]
        plot_next_id = var_list['id'][i][0]


    # Print id of star being processed
    print('---> Processing star id =',plot_next_id)
    print('     ({}/{})'.format(i+1, n_vars))
    print('---> Current params: ', var_list[i])


    # Load light curve
    star = '{}'.format(plot_next_id)

    try:
        dt = np.dtype([('filt', 'U5'), ('mjd', float), ('mag', float), ('err', float)])
        lcv = np.loadtxt(lcv_dir+'c'+star+'.lcv', dtype=dt, skiprows=3,
            usecols=(0,1,2,3))
        dt = np.dtype([('filt', int), ('mjd', float), ('mag', float), ('err', float)])
        lcv_clean = np.loadtxt(lcv_dir+'c'+star+'.fitlc', dtype=dt, skiprows=3,
            usecols=(0,1,2,3))
        fit = np.loadtxt(lcv_dir+'c'+star+'.fitlc_fit', dtype=([('phase', float),
            ('mag1', float), ('mag2', float)]), skiprows=1)
    except:
        print('Light curve file doesn\'t exist for this star!')
        return

    # Initialize plot

    fig = plt.figure(constrained_layout=True, figsize=(12,18))
    gs = GridSpec(6,2, figure=fig)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,0])
    ax4 = fig.add_subplot(gs[1,1])
    ax5 = fig.add_subplot(gs[2,0])
    ax6 = fig.add_subplot(gs[2,1])
    ax7 = fig.add_subplot(gs[3,0])
    ax8 = fig.add_subplot(gs[3,1])
    axbig1 = fig.add_subplot(gs[4,:])
    axbig2 = fig.add_subplot(gs[5,:])

    # Plor color-magnitude diagram
    ax1.scatter(allstars['mag1'][sel]-allstars['mag2'][sel], allstars['mag2'][sel],
        s=1, alpha=0.5, color='gray')
    ax1.scatter(var_list['mag1'][vari]-var_list['mag2'][vari],
        var_list['mag2'][vari], s=10, color='xkcd:blue')
    #ax1.set_ylim(29,21)
    ax1.invert_yaxis()
    ax1.set_xlim(-2,4)
    ax1.set_xlabel('{} - {}'.format(band1, band2))
    ax1.set_ylabel(band2)
    color = var_list['mag1']-var_list['mag2']
    mag = var_list['mag2']
    ax1.scatter(color[i], mag[i], color='xkcd:red',s=40)
    ax1.text(0.1, 0.95, 'V'+star,
        transform=ax1.transAxes, color='xkcd:red')

    if image != False:
        ap.plot_region(var_list['cat_id'][i], var_list['x'][i], var_list['y'][i],
            image, fig=fig, axes=ax2, xall=allstars['x'], yall=allstars['y'],
            xoff=xoff, yoff=yoff, aperture=10, img_limits=img_limits)


    # Plot Band1 PL
    # ignore NV, LPV, and EB stars
    vclean = (var_list['type'] != 'BIN') & (var_list['type'] != 'NV') & \
        (var_list['type'] != 'LPV') & (var_list['type'] != 'EB')
    ax3.scatter(np.log10(var_list['period'][vclean]), var_list['mag1'][vclean],
        s=10, color='gray', alpha=0.5)
    ax3.invert_yaxis()
    ax3.set_ylabel(band1)
    ax3.set_xlabel('$\log P$ [days]')
    ax3.scatter(np.log10(var_list['period'][i]), var_list['mag1'][i],
        s=40, color='xkcd:red')
    ax3.text(0.1, 0.95, 'V'+star,
        transform=ax3.transAxes, color='xkcd:red')

    # Plot Band2 PL
    if plot_lmc == True:
        if rrl == True:
            plot_lmc_rrl(axes=[ax4, ax6], offset=DM)
        if cep == True:
            plot_lmc_cep(axes=[ax4, ax6], offset=DM)
        ax4.scatter(np.log10(var_list['period'][vclean]), var_list['mag2'][vclean], s=10,
            color='gray', alpha=0.5)
    else:
        ax4.scatter(np.log10(var_list['period'][vclean]), var_list['mag2'][vclean], s=10,
            color='gray', alpha=0.5)
    ax4.scatter(np.log10(var_list2['period'][i]), var_list2['mag2'][i],
        s=40, color='xkcd:red')
    ax4.set_xlabel('$\log P$ [days]')
    ax4.set_ylabel(band2)
    ax4.set_xlim(ax3.get_xlim())
    ax4.invert_yaxis()


    # Plot mag1 period-amplitude diagram
    ax5.scatter(var_list['period'][vclean], var_list['amp1'][vclean],
        s=10, color='gray', alpha=0.5)
    ax5.set_ylabel('Amplitude ({})'.format(band1))
    ax5.set_xlabel('P [days]')
    ax5.scatter(var_list['period'][i],
        var_list['amp1'][i], s=40, color='xkcd:red')
    ax5.text(0.1, 0.95, 'V'+star,
        transform=ax5.transAxes, color='xkcd:red')
    ax5.set_xlim(0,2)


    if plot_lmc == False:
        # Plot Amplitude ratio
        ax6.scatter(np.log10(var_list['period'][vclean]),
            var_list['amp1'][vclean]/var_list['amp2'][vclean],
            color='gray', alpha=0.5, s=10)
        ax6.scatter(np.log10(var_list['period'][i]),
            var_list['amp1'][i]/var_list['amp2'][i],
            color='xkcd:red', s=40)
        ax6.set_xlabel('$\log P$')
        ax6.set_ylabel('Amp ratio')
    else:
        ax6.scatter(var_list['period'][i],
            var_list['amp2'][i], s=40, color='xkcd:red')
        ax6.set_ylabel('Amplitude ({})'.format(band2))
        ax6.set_xlabel('P [days]')


    # Plot phased light curve
    filt = lcv['filt'] == band1
    phase = np.mod((lcv['mjd'][filt]-var_list['t0'][i])/var_list['period'][i],1)
    phase_all = np.concatenate((phase, phase+1))
    mag_all = np.tile(lcv['mag'][filt],2)
    err_all = np.tile(lcv['err'][filt],2)
    ax7.errorbar(phase_all, mag_all, yerr=err_all, fmt='.', color='xkcd:gray')

    fil = lcv_clean['filt'] == 0
    phase = np.mod((lcv_clean['mjd'][fil]-var_list['t0'][i])/var_list['period'][i],1)
    phase = np.concatenate((phase, phase+1))
    mag = np.tile(lcv_clean['mag'][fil],2)
    err = np.tile(lcv_clean['err'][fil],2)
    phase_fit = np.concatenate((fit['phase'], fit['phase']+1))
    mag_fit = np.tile(fit['mag1'],2)
    ax7.errorbar(phase, mag, yerr=err, fmt='.', color='xkcd:ocean blue')
    ax7.plot(phase_fit, mag_fit, color='xkcd:ocean blue')
    ax7.set_xlabel('Phase')
    ax7.set_ylabel(band1)
    ax7.set_ylim(np.max(lcv_clean['mag'][fil])+0.3, np.min(lcv_clean['mag'][fil])-0.3)

    # plot unphased light curve

    # check for break in x-axis
    mjd_order = np.argsort(lcv['mjd'])
    min_mjd = lcv['mjd'][mjd_order][0] - 0.1
    max_mjd = lcv['mjd'][mjd_order][-1] + 0.1
    mjd_window = max_mjd - min_mjd
    time_diff = np.diff(lcv['mjd'][mjd_order])
    breaks = np.argwhere(time_diff > 5) # any time differences larger than 5 days?

    axbig1.set_xlim(min_mjd, max_mjd)
    axbig2.set_xlim(min_mjd, max_mjd)
#    if len(breaks) == 0:
#        axbig1.set_xlim(min_mjd, max_mjd)
#        axbig2.set_xlim(min_mjd, max_mjd)
#    else:
#        nbreaks = len(breaks)
#        xlim = np.zeros(2*nbreaks+2)
#        xlim[0] = min_mjd
#        xlim[-1] = max_mjd

#        j = 1
#        while j < 2*nbreaks:
#            xlim[j] = lcv['mjd'][mjd_order][breaks[j-1]]+0.2
#            xlim[j+1] = lcv['mjd'][mjd_order][breaks[j-1]+1]-0.2
#            j += 2
#        xlim = np.reshape(xlim, (nbreaks+1,2))
#        xlim = tuple(map(tuple, xlim))
#        print(xlim)
#        bax1 = brokenaxes(xlims=xlim, subplot_spec=gs[4,:], fig=fig)
#        bax2 = brokenaxes(xlims=xlim, subplot_spec=gs[5,:], fig=fig)


    axbig1.errorbar(lcv['mjd'][filt], lcv['mag'][filt], yerr=lcv['err'][filt],
        fmt='.', color='xkcd:gray')
    axbig1.errorbar(lcv_clean['mjd'][fil], lcv_clean['mag'][fil],
        yerr=lcv_clean['err'][fil], fmt='.', color='xkcd:ocean blue')
    ncycles = int(np.ceil(mjd_window/var_list['period'][i]))*2

    tt = fit['phase']*var_list['period'][i] + var_list['t0'][i]
    ttt = []
    for j in np.arange(0,ncycles):
        ttt = np.append(ttt, tt+(j-ncycles/2)*var_list['period'][i])
    mm = np.tile(fit['mag1'], ncycles)
    axbig1.plot(ttt, mm, color='xkcd:ocean blue')
    axbig1.set_xlim(np.min(lcv['mjd'])-0.1, np.max(lcv['mjd'])+0.1)
    axbig1.set_ylim(np.mean(fit['mag1'])+1.0, np.mean(fit['mag1'])-1.0)


    filt = lcv['filt'] == band2
    phase = np.mod((lcv['mjd'][filt]-var_list['t0'][i])/var_list['period'][i],1)
    phase_all = np.concatenate((phase, phase+1))
    mag_all = np.tile(lcv['mag'][filt],2)
    err_all = np.tile(lcv['err'][filt],2)
    ax8.errorbar(phase_all, mag_all, yerr=err_all, fmt='.', color='xkcd:gray')

    fil = lcv_clean['filt'] == 1
    phase = np.mod((lcv_clean['mjd'][fil]-var_list['t0'][i])/var_list['period'][i],1)
    phase = np.concatenate((phase, phase+1))
    mag = np.tile(lcv_clean['mag'][fil],2)
    err = np.tile(lcv_clean['err'][fil],2)
    phase_fit = np.concatenate((fit['phase'], fit['phase']+1))
    mag_fit = np.tile(fit['mag2'],2)
    ax8.errorbar(phase, mag, yerr=err, fmt='.', color='xkcd:rose')
    ax8.plot(phase_fit, mag_fit, color='xkcd:rose')
    ax8.set_xlabel('Phase')
    ax8.set_ylabel(band2)
    ax8.set_ylim(np.max(lcv_clean['mag'][fil])+0.3, np.min(lcv_clean['mag'][fil])-0.3)

    # plot unphased light curve
    axbig2.errorbar(lcv['mjd'][filt], lcv['mag'][filt], yerr=lcv['err'][filt],
        fmt='.', color='xkcd:gray')
    axbig2.errorbar(lcv_clean['mjd'][fil], lcv_clean['mag'][fil],
        yerr=lcv_clean['err'][fil], fmt='.', color='xkcd:rose')

    mm = np.tile(fit['mag2'], ncycles)
    axbig2.plot(ttt, mm, color='xkcd:rose')
    #axbig2.set_xlim(np.min(lcv['mjd'])-0.1, np.max(lcv['mjd'])+0.1)
    axbig2.set_ylim(np.mean(fit['mag2'])+1.0, np.mean(fit['mag2'])-1.0)

    axbig1.set_xlabel('MJD')
    axbig1.set_ylabel('mag')
    #axbig1.invert_yaxis()
    axbig2.set_xlabel('MJD')
    axbig2.set_ylabel('mag')
    #axbig2.invert_yaxis()

    # Show and close plot
    plt.show()
    plt.close()

    # User input to change type, subtype, and parameters in file
    if update == True:

        old_type = var_list2['type'][i]
        old_subtype = var_list2['subtype'][i]

        new_type = input('New type:')
        new_subtype = input('New subtype:')
        print('---> Changing file...')

        if new_type == '':
            new_type = old_type
            new_subtype = old_subtype

        var_list2['type'][i] = new_type
        var_list2['subtype'][i] = new_subtype

        np.savetxt(VAR_FILE, var_list2,
            fmt='%8i %8i %3s %3s %8.3f %8.3f %7.5f %10.4f %6.3f %5.3f %4.2f %6.3f %5.3f %4.2f')


    # Save id of star that was processed
    new_id = str(plot_next_id)
    f = open(LASTID_FILE,'w')
    f.write(new_id+'\n')
    f.close()

    return



def plot_lmc_cep(axes=None, offset=0, period_cutoff=0):

    t2cep_dir = config.ogle_dir+'LMC/t2cep/'
    acep_dir = config.ogle_dir+'LMC/acep/'
    ccep_dir = config.ogle_dir+'LMC/ccep/'

    dt = np.dtype([('i', float), ('v', float), ('p', float), ('amp', float)])
    t2cep = np.loadtxt(t2cep_dir+'t2cep.dat.txt', usecols=(1,2,3,6), dtype=dt)
    afu = np.loadtxt(acep_dir+'acepF.dat.txt', usecols=(1,2,3,6), dtype=dt)
    afo = np.loadtxt(acep_dir+'acep1O.dat.txt', usecols=(1,2,3,6), dtype=dt)
    cfu = np.loadtxt(ccep_dir+'cepF.dat.txt', usecols=(1,2,3,6), dtype=dt)
    cfo = np.loadtxt(ccep_dir+'cep1O.dat.txt', usecols=(1,2,3,6), dtype=dt)

    if period_cutoff != 0:
        t2cep['p'][t2cep['p'] > period_cutoff] = np.nan
        afo['p'][afo['p'] > period_cutoff] = np.nan
        afu['p'][afu['p'] > period_cutoff] = np.nan
        cfo['p'][cfo['p'] > period_cutoff] = np.nan
        cfu['p'][cfu['p'] > period_cutoff] = np.nan
    if axes == None:
        fig1, ax1 = plt.subplots(1,1)
        fig2, ax2 = plt.subplots(1,1)
    else:
        ax1 = axes[0]
        ax2 = axes[1]

    # classical cepheid lines
    x_fo = np.array([-0.6, 0.8])
    x_fu = np.array([0.0, 2.1])
    y_fo = -3.311*(x_fo-1.0) + 12.897 - 18.477 + offset
    y_fu = -2.912*(x_fu-1.0) + 13.741 - 18.477 + offset
    ax1.scatter(x_fo, y_fo, s=1, color='xkcd:sage')
    ax1.scatter(x_fu, y_fu, s=1, color='xkcd:gray')
    ax1.fill_between(x_fo, y_fo-0.16, y_fo+0.16, color='xkcd:sage', alpha=0.4)
    ax1.fill_between(x_fu, y_fu-0.15, y_fu+0.15, color='xkcd:gray', alpha=0.4)
    # anomalous cepheid lines
    x_fo = np.array([-0.4, 0.07])
    x_fu = np.array([-0.2, 0.37])
    y_fo = -3.302*x_fo + 16.656 - 18.477 + offset
    y_fu = -2.962*x_fu + 17.368 - 18.477 + offset
    ax1.scatter(x_fo, y_fo, s=1, color='xkcd:pale purple')
    ax1.scatter(x_fu, y_fu, s=1, color='xkcd:rose')
    ax1.fill_between(x_fo, y_fo-0.16, y_fo+0.16, color='xkcd:pale purple', alpha=0.4)
    ax1.fill_between(x_fu, y_fu-0.23, y_fu+0.23, color='xkcd:rose', alpha=0.4)
    # type 2 cepheid line
    x_fu = np.array([-0.09, 1.8])
    y_fu = -2.033*x_fu + 18.015 - 18.477 + offset
    ax1.scatter(x_fu, y_fu, s=1, color='xkcd:steel blue')
    ax1.fill_between(x_fu, y_fu-0.4, y_fu+0.4, color='xkcd:steel blue', alpha=0.4)

    ax2.scatter(t2cep['p'], t2cep['amp'], s=4, color='xkcd:steel blue', alpha=0.5)
    ax2.scatter(afo['p'], afo['amp'], s=4, color='xkcd:pale purple', alpha=0.5)
    ax2.scatter(afu['p'], afu['amp'], s=4, color='xkcd:rose', alpha=0.5)
    ax2.scatter(cfo['p'], cfo['amp'], s=4, color='xkcd:sage', alpha=0.5)
    ax2.scatter(cfu['p'], cfu['amp'], s=4, color='xkcd:gray', alpha=0.5)
    ax2.set_xlim(0,3)

    if axes == None:
        ax1.set_xlabel('$\log P$')
        ax1.set_ylabel('I mag')
        ax1.invert_yaxis()
        ax2.set(xlabel='P [days]', ylabel='I amp')
        plt.show()


def plot_lmc_cep_pl(axes=None, offset=0, period_cutoff=0):

    t2cep_dir = config.ogle_dir+'LMC/t2cep/'
    acep_dir = config.ogle_dir+'LMC/acep/'
    ccep_dir = config.ogle_dir+'LMC/ccep/'

    dt = np.dtype([('i', float), ('v', float), ('p', float), ('amp', float)])
    t2cep = np.loadtxt(t2cep_dir+'t2cep.dat.txt', usecols=(1,2,3,6), dtype=dt)
    afu = np.loadtxt(acep_dir+'acepF.dat.txt', usecols=(1,2,3,6), dtype=dt)
    afo = np.loadtxt(acep_dir+'acep1O.dat.txt', usecols=(1,2,3,6), dtype=dt)
    cfu = np.loadtxt(ccep_dir+'cepF.dat.txt', usecols=(1,2,3,6), dtype=dt)
    cfo = np.loadtxt(ccep_dir+'cep1O.dat.txt', usecols=(1,2,3,6), dtype=dt)

    if period_cutoff != 0:
        t2cep['p'][t2cep['p'] > period_cutoff] = np.nan
        afo['p'][afo['p'] > period_cutoff] = np.nan
        afu['p'][afu['p'] > period_cutoff] = np.nan
        cfo['p'][cfo['p'] > period_cutoff] = np.nan
        cfu['p'][cfu['p'] > period_cutoff] = np.nan
    if axes == None:
        fig1, ax1 = plt.subplots(1,1)

    else:
        ax1 = axes


    # classical cepheid lines
    x_fo = np.array([-0.6, 0.8])
    x_fu = np.array([0.0, 2.1])

    y_fo = -3.311*(x_fo-1.0) + 12.897 - 18.477 + offset
    y_fu = -2.912*(x_fu-1.0) + 13.741 - 18.477 + offset
    ax1.scatter(x_fo, y_fo, s=1, color='xkcd:sage')
    ax1.scatter(x_fu, y_fu, s=1, color='xkcd:gray')
    ax1.fill_between(x_fo, y_fo-0.16, y_fo+0.16, color='xkcd:sage', alpha=0.4)
    ax1.fill_between(x_fu, y_fu-0.15, y_fu+0.15, color='xkcd:gray', alpha=0.4)
    # anomalous cepheid lines
    x_fo = np.array([-0.4, 0.07])
    x_fu = np.array([-0.2, 0.37])
    y_fo = -3.302*x_fo + 16.656 - 18.477 + offset
    y_fu = -2.962*x_fu + 17.368 - 18.477 + offset
    ax1.scatter(x_fo, y_fo, s=1, color='xkcd:pale purple')
    ax1.scatter(x_fu, y_fu, s=1, color='xkcd:rose')
    ax1.fill_between(x_fo, y_fo-0.16, y_fo+0.16, color='xkcd:pale purple', alpha=0.4)
    ax1.fill_between(x_fu, y_fu-0.23, y_fu+0.23, color='xkcd:rose', alpha=0.4)
    # type 2 cepheid line
    x_fu = np.array([-0.09, 1.8])
    y_fu = -2.033*x_fu + 18.015 - 18.477 + offset
    ax1.scatter(x_fu, y_fu, s=1, color='xkcd:steel blue')
    ax1.fill_between(x_fu, y_fu-0.4, y_fu+0.4, color='xkcd:steel blue', alpha=0.4)

    if axes == None:
        ax1.set_xlabel('$\log P$')
        ax1.set_ylabel('I mag')
        ax1.invert_yaxis()
        ax2.set(xlabel='P [days]', ylabel='I amp')
        plt.show()

def plot_lmc_rrl(axes=None, offset=0, rrd_fu=False):

    rrl_dir = config.ogle_dir+'LMC/rrl/'

    dt = np.dtype([('i', float), ('v', float), ('p', float), ('amp', float)])

    rrab = np.loadtxt(rrl_dir+'RRab.dat.txt', usecols=(1,2,3,6), dtype=dt)
    rrc = np.loadtxt(rrl_dir+'RRc.dat.txt', usecols=(1,2,3,6), dtype=dt)
    if rrd_fu == False:
        rrd = np.loadtxt(rrl_dir+'RRd.dat.txt', usecols=(1,2,3,6), dtype=dt)
    else:
        rrd = np.loadtxt(rrl_dir+'RRd.dat.txt', usecols=(1,2,11,14), dtype=dt)

    rrab[rrab['i'] < 18] = np.nan
    rrc[rrc['i'] < 18] = np.nan
    rrd[rrd['i'] < 18] = np.nan
    rrab[rrab['i'] > 20] = np.nan
    rrc[rrc['i'] > 20] = np.nan
    rrd[rrd['i'] > 20] = np.nan

    if axes == None:
        fig1, ax1 = plt.subplots(1,1)
        fig2, ax2 = plt.subplots(1,1)
    else:
        ax1 = axes[0]
        ax2 = axes[1]

    x_fo = np.array([-0.7, -0.3])
    x_fu = np.array([-0.6, 0.0])
    y_fo = -2.014*x_fo + 17.743 - 18.477 + offset
    y_fu = -1.889*x_fu + 18.164 - 18.477 + offset
    fo_mag = -2.014*np.log10(periods_fo) + 17.743
    fu_mag = -1.889*np.log10(periods_fu) + 18.164
    ax1.scatter(x_fu, y_fu, s=1, color='xkcd:puce')
    ax1.scatter(x_fo, y_fo, s=1, color='xkcd:eggplant')
    ax1.fill_between(x_fu, y_fu-0.15, y_fu+0.15, color='xkcd:puce', alpha=0.5)
    ax1.fill_between(x_fo, y_fo-0.16, y_fo+0.16, color='xkcd:eggplant', alpha=0.5)


    ax2.scatter(rrab['p'], rrab['amp'], s=1, color='xkcd:puce', alpha=0.5)
    ax2.scatter(rrc['p'], rrc['amp'], s=1, color='xkcd:eggplant', alpha=0.5)
    #ax2.scatter(rrd['p'], rrd['amp'], s=1, color='xkcd:sage', alpha=0.5)

    if axes == None:
        ax1.set_xlabel('$\log P$')
        ax1.set_ylabel('I mag')
        ax1.invert_yaxis()
        ax2.set(xlabel='P [days]', ylabel='I amp')
        plt.show()


def update_variable_list(VAR_FILE, star_id, lcv_dir='lcvs/'):


    dt = np.dtype([('temp', int), ('period', float), ('chisq', float),
        ('epoch', float), ('amp1', float), ('t1', float), ('mag1', float),
        ('amp2', float), ('t2', float), ('mag2', float), ('color', float)])
    props = np.loadtxt(lcv_dir+'c{}.fitlc_props'.format(star_id), dtype=dt,
        skiprows=1)


    # get mean mag and err from fit and phased file
    dt = np.dtype([('phase', float), ('mag1', float), ('mag2', float)])
    fit = np.loadtxt(lcv_dir+'c{}.fitlc_fit'.format(star_id), dtype=dt,
        skiprows=1)
    dt = np.dtype([('filter', int), ('phase', float), ('mag', float),
        ('err', float)])
    lcv = np.loadtxt(lcv_dir+'c{}.fitlc_phase'.format(star_id), dtype=dt,
        skiprows=1, usecols=(0,1,2,3))

    flux1 = 99*np.power(10,-fit['mag1']/2.5)
    flux2 = 99*np.power(10,-fit['mag2']/2.5)
    mean_mag1 = -2.5*np.log10(np.mean(flux1)/99)
    mean_mag2 = -2.5*np.log10(np.mean(flux2)/99)

    # photometric uncertainty of points
    b1 = lcv['filter'] == 0
    b2 = lcv['filter'] == 1
    sigma_p1 = 1./np.sum(1./lcv['err'][b1]**2)
    sigma_p2 = 1./np.sum(1./lcv['err'][b2]**2)
    # scatter around template
    n1 = len(lcv['mag'][b1])
    n2 = len(lcv['mag'][b2])
    template_mag1 = np.interp(lcv['phase'][b1], fit['phase'], fit['mag1'])
    template_mag2 = np.interp(lcv['phase'][b2], fit['phase'], fit['mag2'])
    residuals1 = lcv['mag'][b1] - template_mag1
    residuals2 = lcv['mag'][b2] - template_mag2
    sigma_fit1 = 1./float(n1)*np.std(residuals1)
    sigma_fit2 = 1./float(n2)*np.std(residuals2)

    err1 = np.sqrt(sigma_p1 + sigma_fit1)
    err2 = np.sqrt(sigma_p2 + sigma_fit2)

    # Load list of variables and types to check
    dt = np.dtype([('id', int), ('cat_id', int), ('type', 'U4'),
        ('subtype', 'U4'), ('x', float), ('y', float), ('period', float),
        ('t0', float), ('mag1', float), ('err1', float), ('amp1', float),
        ('mag2', float), ('err2', float), ('amp2', float)])
    var_list = np.loadtxt(VAR_FILE, dtype=dt)

    i = var_list['id'] == int(star_id)
    print('Old period: {}'.format(var_list['period'][i][0]))
    print('New period: {}'.format(props['period']))
    var_list['period'][i] = props['period']
    var_list['t0'][i] = props['epoch']
    var_list['mag1'][i] = mean_mag1
    var_list['err1'][i] = err1
    var_list['amp1'][i] = props['amp1']
    var_list['mag2'][i] = mean_mag2
    var_list['err2'][i] = err2
    var_list['amp2'][i] = props['amp2']


    np.savetxt(VAR_FILE, var_list,
        fmt='%8i %8i %3s %3s %8.3f %8.3f %7.5f %10.4f %6.3f %5.3f %4.2f %6.3f %5.3f %4.2f')
