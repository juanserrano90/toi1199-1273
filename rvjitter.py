import numpy as np
from scipy import interpolate

def rvjitter(logRhk,sptype='G'):
    """
    Computes stellar jitter given a value for LogR'hk
    and a sptype (F, G or K).
    Uses relations by Santos et al. (2000).
    """
    #
    Rhk = 10**logRhk
    R5 = Rhk*1e5
    #
    if sptype=='F':
        jitmin = 7.9*R5**(0.48)
        jit = 9.2*R5**(0.75)
        jitmax = 10.9*R5**(1.02)
    elif sptype=='G':
        jitmin = 7.2*R5**(0.47)
        jit = 7.9*R5**(0.55)
        jitmax = 8.31*R5**(0.63)
    elif sptype=='K':
        jitmin = 4.90*R5**(0.13)
        jit = 7.8*R5**(0.13)
        jitmax = 11.7*R5**(0.13)
    else:
        print("Sp Type must be either F, G, or K")
        return
    return jitmin,jit,jitmax

def rvjitter_wright(sindex, bmv, teff):
    """
    Compute stellar jitter expected for a star with a given sindex, b-v, and
    teff according to wright (2009)
    """

    # First compute Fca from S
    logCcf = 0.25 * bmv**3 - 1.33 * bmv**2 + 0.43 * bmv + 0.24

    fca = sindex * 10**logCcf * teff**4 * 1e-14

    ## Now interpolate the Fcamin from Table 1 of Wright+2009
    bmvt = list(np.arange(0.4, 1.45, 0.05))
    bmvt.append(1.5)
    bmvt.append(1.6)
    bmvt = np.array(bmvt)

    fcamint = np.array([5.24, 3.80, 2.88, 2.29, 1.82, 1.41, 1.09, 0.831,
                       0.645, 0.489, 0.363, 0.281, 0.218, 0.174, 0.135,
                       0.105, 0.079, 0.063, 0.052, 0.044, 0.038, 0.029,
                       0.025])
    ##
    fcamin = interpolate.interp1d(bmvt, fcamint)(bmv)

    return fca - fcamin