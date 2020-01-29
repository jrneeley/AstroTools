import numpy as np
import matplotlib.pyplot as plt
import sys
#sys.path.insert(0, '/Users/jill/python/')
#import daophot_tools as dao
from . import config
from astropy.io import fits
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from astropy.visualization import LogStretch, ImageNormalize, PercentileInterval

np.warnings.filterwarnings('ignore')

def compute_variability_index(filters, mjds, mags, errs,
    statistic='WelchStetsonI', max_time=0.02):
    # Currently doesn't allow nans

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

            stat = np.sqrt(1./(npairs*(npairs-1)))*tot

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
    fig, ax = plt.subplots(6,2,constrained_layout=True, figsize=(12,18))
    ax1 = ax[0,0]
    ax2 = ax[0,1]
    ax3 = ax[1,0]
    ax4 = ax[1,1]
    ax5 = ax[2,0]
    ax6 = ax[2,1]
    ax7 = ax[3,0]
    ax8 = ax[3,1]
    gs1 = ax[4,0].get_gridspec()
    for axis in ax[4,:]: axis.remove()
    axbig1 = fig.add_subplot(gs1[4,:])
    gs2 = ax[5,0].get_gridspec()
    for axis in ax[5,:]: axis.remove()
    axbig2 = fig.add_subplot(gs2[5,:])

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
        plot_region(var_list['cat_id'][i], var_list['x'][i], var_list['y'][i],
            image, axes=ax2, xall=allstars['x'], yall=allstars['y'],
            xoff=xoff, yoff=yoff, aperture=10, img_limits=img_limits)


    # Plot Band1 PL
    # ignore NV and EB stars
    vclean = (var_list['type'] != 'BIN') & (var_list['type'] != 'NV') & \
        (var_list['type'] != 'LPV')
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
    axbig1.errorbar(lcv['mjd'][filt], lcv['mag'][filt], yerr=lcv['err'][filt],
        fmt='.', color='xkcd:gray')

    axbig1.errorbar(lcv_clean['mjd'][fil], lcv_clean['mag'][fil],
        yerr=lcv_clean['err'][fil], fmt='.', color='xkcd:ocean blue')
    tt = fit['phase']*var_list['period'][i] + var_list['t0'][i]
    ttt = []
    for j in np.arange(0,40):
        ttt = np.append(ttt, tt+(j-20)*var_list['period'][i])
    mm = np.tile(fit['mag1'], 40)
    axbig1.plot(ttt, mm, color='xkcd:ocean blue')
    axbig1.set_xlim(np.min(lcv['mjd'])-0.1, np.max(lcv['mjd'])+0.1)



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
    tt = fit['phase']*var_list['period'][i] + var_list['t0'][i]
    ttt = []
    for j in np.arange(0,40):
        ttt = np.append(ttt, tt+(j-20)*var_list['period'][i])
    mm = np.tile(fit['mag2'], 40)
    axbig2.plot(ttt, mm, color='xkcd:rose')
    axbig2.set_xlim(np.min(lcv['mjd'])-0.1, np.max(lcv['mjd'])+0.1)

    axbig1.set_xlabel('MJD')
    axbig1.set_ylabel('mag')
    axbig1.invert_yaxis()
    axbig2.set_xlabel('MJD')
    axbig2.set_ylabel('mag')
    axbig2.invert_yaxis()

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
    y_fo = -3.328*x_fo + 16.209 - 18.477 + offset
    y_fu = -2.914*x_fu + 16.672 - 18.477 + offset
    ax1.scatter(x_fo, y_fo, s=1, color='xkcd:sage')
    ax1.scatter(x_fu, y_fu, s=1, color='xkcd:gray')
    ax1.fill_between(x_fo, y_fo-0.23, y_fo+0.23, color='xkcd:sage', alpha=0.4)
    ax1.fill_between(x_fu, y_fu-0.21, y_fu+0.21, color='xkcd:gray', alpha=0.4)
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

    #ax.scatter(np.log10(jill['period']), jill['i_mag'])
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
    y_fo = -1.946*x_fo + 17.784 - 18.477 + offset
    y_fu = -1.894*x_fu + 18.169 - 18.477 + offset
    ax1.scatter(x_fu, y_fu, s=1, color='xkcd:puce')
    ax1.scatter(x_fo, y_fo, s=1, color='xkcd:eggplant')
    ax1.fill_between(x_fu, y_fu-0.2, y_fu+0.2, color='xkcd:puce', alpha=0.5)
    ax1.fill_between(x_fo, y_fo-0.2, y_fo+0.2, color='xkcd:eggplant', alpha=0.5)


    ax2.scatter(rrab['p'], rrab['amp'], s=1, color='xkcd:puce', alpha=0.5)
    ax2.scatter(rrc['p'], rrc['amp'], s=1, color='xkcd:eggplant', alpha=0.5)
    #ax2.scatter(rrd['p'], rrd['amp'], s=1, color='xkcd:sage', alpha=0.5)

    if axes == None:
        ax1.set_xlabel('$\log P$')
        ax1.set_ylabel('I mag')
        ax1.invert_yaxis()
        ax2.set(xlabel='P [days]', ylabel='I amp')
        plt.show()

def plot_region(star, x, y, image, xall=[], yall=[], ext=0,
    axes=None, xoff=0, yoff=0, aperture=None, img_limits=[0,500]):

    image_data = fits.getdata(image, ext=ext)

    #dt = np.dtype([('id', int), ('x', float), ('y', float)])
    #star_data = np.loadtxt(star_list, dtype=dt, usecols=(0,1,2), skiprows=3)

    #x_all = star_data['x'] - (xoff+1)
    #y_all = star_data['y'] - (yoff+1)

    if axes == None:
        fig, ax = plt.subplots(1,1, figsize=(8,5))
    else:
        ax = axes

    norm1 = ImageNormalize(image_data, vmin=img_limits[0], vmax=img_limits[1],
        stretch=LogStretch())

    ax.imshow(image_data, cmap='gray', norm=norm1)

    ax.set_aspect('equal')
    if aperture != None:
        ap = Circle((x, y), aperture, facecolor=None, edgecolor='red', fill=0)
        ax.add_patch(ap)
    if (len(xall) > 0) & (len(yall) > 0):
        x_all = xall - (xoff+1)
        y_all = yall - (yoff+1)
        ax.scatter(x_all, y_all, marker='x', color='green')
    ax.set_xlim(x-20, x+20)
    ax.set_ylim(y-20, y+20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')


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
