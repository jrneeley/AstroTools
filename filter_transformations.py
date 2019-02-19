import numpy as np

def hst_wfc2ground_synthetic(mag1, mag2, hst_band1, hst_band2, mag='vega'):

    filters = np.array(['F435W', 'F475W', 'F555W', 'F606W', 'F625W', 'F775W', 'F814W'])
    if mag == 'vega':
        zeropoints = np.array([25.779, 26.168, 25.724, 26.398, 25.731, 25.256, 25.501])
    if mag == 'ab':
        zeropoints = np.array([25.673, 26.068, 25.718, 26.486, 25.898, 25.654, 25.937])
    if mag == 'st':
        zeropoints = np.array([25.157, 25.757, 25.672, 26.655, 26.206, 26.393, 26.776])

    mag1_0 = mag1 - zeropoints[filters == hst_band1]
    mag2_0 = mag2 - zeropoints[filters == hst_band2]
    color_hst = mag1 - mag2

# B - V
    if (hst_band1 == 'F435W') & (hst_band2 == 'F555W'):
        if color_hst < 0.4: a0, a1, a3 = 25.769, 0.029, -0.156
        else: a0, a1, a2 = 25.709, 0.108, -0.068
        if color_hst < 0.2: b0, b1, b2 = 25.714, -0.083, 0.020
        else: b0, b1, b2 = 25.720, -0.087, 0.004
    elif (hst_band1 == 'F435W') & (hst_band2 == 'F606W'):
        if color_hst < 0.4: a0, a1, a2 = 25.769, 0.029, -0.156
        else: a0, a1, a2 = 25.709, 0.108, -0.068
        b0, b1, b2 = 26.410, 0.170, 0.061
    elif (hst_band1 == 'F475W') & (hst_band2 == 'F555W'):
        a0, a1, a2 = 26.146, 0.389, 0.032
        if color_hst < 0.2: b0, b1, b2 = 25.714, -0.083, 0.020
        else: b0, b1, b2 = 25.720, -0.087, 0.004
    elif (hst_band1 == 'F475W') & (hst_band2 == 'F606W'):
        a0, a1, a2 = 26.146, 0.389, 0.032
        b0, b1, b2 = 26.410, 0.170, 0.061

# B - R
    elif (hst_band1 == 'F435W') & (hst_band2 == 'F625W'):
        if color_hst < 0.6: a0, a1, a2 = 25.769, 0.027, -0.073
        else: a0, a1, a2 = 25.743, 0.022, -0.013
        b0, b1, b2 = 25.717, -0.041, -0.003
    elif (hst_band1 == 'F475W') & (hst_band2 == 'F625W'):
        if color_hst < 0.6: a0, a1, a2 = 26.150, 0.291, -0.110
        else: a0, a1, a2 = 26.015, 0.433, -0.046
        b0, b1, b2 = 25.717, -0.041, -0.003

# B - I
    elif (hst_band1 == 'F435W') & (hst_band2 == 'F775W'):
        if color_hst < 1.0: a0, a1, a2 = 25.768, 0.022, -0.038
        else: a0, a1, a2 = 25.749, 0.008, -0.005
        if color_hst < 1.0: b0, b1, b2 = 25.239, -0.030, -0.002
        else: b0, b1, b2 = 25.216, -0.002, -0.007
    elif (hst_band1 == 'F435W') & (hst_band2 == 'F814W'):
        if color_hst < 1.0: a0, a1, a2 = 25.768, 0.022, -0.038
        else: a0, a1, a2 = 25.749, 0.008, -0.005
        if color_hst < 0.3: b0, b1, b2 = 25.490, 0.013, -0.028
        else: b0, b1, b2 = 25.495, -0.010, 0.006
    elif (hst_band1 == 'F475W') & (hst_band2 == 'F775W'):
        if color_hst < 1.0: a0, a1, a2 = 26.145, 0.220, -0.050
        else: a0, a1, a2 = 25.875, 0.456, -0.053
        if color_hst < 1.0: b0, b1, b2 = 25.239, -0.030, -0.002
        else: b0, b1, b2 = 25.216, -0.002, -0.007
    elif (hst_band1 == 'F475W') & (hst_band2 == 'F814W'):
        if color_hst < 1.0: a0, a1, a2 = 26.145, 0.220, -0.050
        else: a0, a1, a2 = 25.875, 0.456, -0.053
        if color_hst < 0.3: b0, b1, b2 = 25.490, 0.013, -0.028
        else: b0, b1, b2 = 25.495, -0.010, 0.006

# V - R
    elif (hst_band1 == 'F555W') & (hst_band2 == 'F625W'):
        if color_hst < 0.1: a0, a1, a2 = 25.720, -0.197, 0.159
        else: a0, a1, a2 = 25.724, -0.159, 0.018
        if color_hst < 0.2: b0, b1, b2 = 25.720, -0.147, 0.043
        else: b0, b1, b2 = 25.684, 0.042, -0.184
    elif (hst_band1 == 'F606W') & (hst_band2 == 'F625W'):
        if color_hst < 0.1: a0, a1, a2 = 26.392, 0.310, 0.266
        else: a0, a1, a2 = 26.372, 0.477, -0.001
        if color_hst < 0.2: b0, b1, b2 = 25.720, -0.147, 0.043
        else: b0, b1, b2 = 25.684, 0.042, -0.184

# V - I
    elif (hst_band1 == 'F555W') & (hst_band2 == 'F775W'):
        if color_hst < 0.4: a0, a1, a2 = 25.719, -0.088, 0.043
        else: a0, a1, a2 = 25.735, -0.106, 0.013
        if color_hst < 1.2: b0, b1, b2 = 25.241, -0.061, 0.002
        else: b0, b1, b2 = 25.292, -0.105, 0.007
    elif (hst_band1 == 'F555W') & (hst_band2 == 'F814W'):
        if color_hst < 0.4: a0, a1, a2 = 25.719, -0.088, 0.043
        else: a0, a1, a2 = 25.735, -0.106, 0.013
        if color_hst < 0.1: b0, b1, b2 = 25.489,  0.041, -0.093
        else: b0, b1, b2 = 25.496, -0.014, 0.015
    elif (hst_band1 == 'F606W') & (hst_band2 == 'F775W'):
        if color_hst < 0.4: a0, a1, a2 = 26.394, 0.153, 0.096
        else: a0, a1, a2 = 26.331, 0.340, -0.038
        if color_hst < 1.2: b0, b1, b2 = 25.241, -0.061, 0.002
        else: b0, b1, b2 = 25.292, -0.105, 0.007
    elif (hst_band1 == 'F606W') & (hst_band2 == 'F814W'):
        if color_hst < 0.4: a0, a1, a2 = 26.394, 0.153, 0.096
        else: a0, a1, a2 = 26.331, 0.340, -0.038
        if color_hst < 0.1: b0, b1, b2 = 25.489, 0.041, -0.093
        else: b0, b1, b2 = 25.496, -0.014, 0.015

# R - I
    elif (hst_band1 == 'F625W') & (hst_band2 == 'F775W'):
        a0, a1, a2 = 25.720, -0.107, -0.101
        if color_hst < 0.2: b0, b1, b2 = 25.240, -0.119, 0.001
        else: b0, b1, b2 = 25.242, -0.133, 0.002
    elif (hst_band1 == 'F625W') & (hst_band2 == 'F814W'):
        a0, a1, a2 = 25.720, -0.107, -0.101
        if color_hst < 0.2: b0, b1, b2 = 25.489, 0.068, -0.295
        else: b0, b1, b2 = 25.478, 0.042, 0.012


    a = a2 - b2
    b = a1 - b1 - 1
    c = color_hst + a0 + b0

    col1 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    col2 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)

    if col1 > 0 :
        col_new = col1
    if col2 > 0:
        col_new = col2
    mag1_new = mag1_0 + a0 + a1*col_new + a2*col_new**2
    mag2_new = mag2_0 + b0 + b1*col_new + b2*col_new**2



def hst_wfc2ground_observed():
    # Do same as above for observed transformations
    print ' I don\'t do anything yet.'
