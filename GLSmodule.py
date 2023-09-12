import os

import numpy as np
import pylab as p

# For C++ parsing
from ctypes import cdll, c_double, c_int, POINTER

import fit

# Directory where C++ code is
cppdir = os.path.join(os.getenv('HOME'), 'code/cpp/')


def GLSperiodogram(t, rv, erv, prange=None, fap=None, plot=True,
                   multfactor=10):
    """Compute GLS periodogram."""
    y = rv.copy()
    ey = erv.copy()

    ind = np.argsort(t)
    # t = t[ind]
    # rv = rv[ind]
    # erv = erv[ind]

    if prange is None:
        # # Construct prange from time array as in yorbit
        Pmax = 2*(t.max() - t.min())
        Pmin = np.max([0.25, np.median(np.diff(t))])
        dnu = 0.25*np.min(np.diff(t))

        prange = np.arange(Pmin, Pmax + dnu, dnu)

        print('Using default values: Pmin: {pmin}; Pmax: {pmax}; '
              'dnu : {dnu}; {N} frequencies.'
              ''.format(pmin=Pmin, pmax=Pmax, dnu=dnu, N=len(prange)))

    if fap is None:
        Nsimul = 1

    else:
        Nsimul = multfactor/fap

    powers = np.zeros(Nsimul)

    # ## TEMPORAL (for C-24)
    # cond0 = np.abs(prange - 11.759) < 3*0.0063
    # cond1 = np.abs(prange - 1./(1./11.759 - 0.00277)) < 3*0.0063
    # cond2 = np.2abs(prange - 1./(1./11.759 - 0.005535)) < 3*0.0063
    # cond1b = np.abs(prange - 1./(1./11.759 + 0.00277)) < 3*0.0063
    # cond2b = np.abs(prange - 1./(1./11.759 + 0.005535)) < 3*0.0063

    # cond = cond0 + cond1 + cond2 + cond1b + cond2b
    ####
    W = np.sum(1/ey**2)

    for j in range(int(Nsimul)):

        amplitude = np.zeros(len(prange))
        phase = np.zeros(len(prange))
        pp = np.zeros(len(prange))

        w = 1/ey**2.0/W
        YYhat = np.sum(w*y*y)
        Y = np.sum(w*y)

        for i in range(len(prange)):

            omega = 2 * np.pi / prange[i]

            ccos = np.cos(omega*t)
            ssin = np.sin(omega*t)

            C = np.sum(w*ccos)
            S = np.sum(w*ssin)

            YChat = np.sum(w*y*ccos)
            YShat = np.sum(w*y*ssin)
            CChat = np.sum(w*ccos*ccos)
            SShat = np.sum(w*ssin*ssin)
            CShat = np.sum(w*ccos*ssin)

            YY = YYhat - Y*Y
            YC = YChat - Y*C
            YS = YShat - Y*S
            CC = CChat - C*C
            SS = SShat - S*S
            CS = CShat - C*S

            D = CC*SS - CS*CS

            pp[i] = (SS*YC*YC + CC*YS*YS - 2*CS*YC*YS)/(YY * D)

            amplitude_cosine = (YC*SS - YS*CS)
            amplitude_sine = (YS*SS - YC*CS)

            amplitude[i] = np.sqrt(amplitude_sine**2 + amplitude_cosine**2)/D
            phase[i] = np.atan2(amplitude_sine, amplitude_cosine)

        if fap is None:
            return prange, pp, amplitude, phase

        if j == 0:
            pp0 = pp

        # # TEMPORAL
        # powers[j] = np.sum(pp[cond])

        powers[j] = np.max(pp)

        # # Shuffle data
        ind = np.arange(len(y))
        np.random.shuffle(ind)
        y = y[ind]
        ey = ey[ind]

    # # Compute FAP probabilities
    # if fap < 0.1:
        # Compute also 10% FAP

    if plot:
        fig1 = p.figure()
        ax = fig1.add_subplot(111)
        ax.hist(powers, 10, histtype='step')
        ax.axvline(powers[0], ls=':', color='k')

    faplevels = []

    spow = np.sort(powers)
    i = np.log10(multfactor)
    while fap <= 0.1:
        faplevels.append(spow[-10**i])
        fap = fap*10.0
        i = i + 1

    return prange, pp0, faplevels


def prepareGLSc():
    """PREPARE C FUNCTION."""
    lib = cdll.LoadLibrary(os.path.join(cppdir, 'GLSperiodogram.so'))
    lib.GLS.argtypes = [POINTER(c_double), c_int,
                        POINTER(c_double), POINTER(c_double),
                        POINTER(c_double), c_int, POINTER(c_double)]
    return lib.GLS


def GLSc(t, rv, erv, prange=None, fap=None, plot=True,
         multfactor=10, **kwargs):
    """Run GLS with C function."""
    # C++ GLS function
    glsfunc = kwargs.pop('glsfunc', None)

    # Check
    if (len(t) != len(rv)) or (len(t) != len(erv)):
        raise RuntimeError('Arrays do not have same length')

    if glsfunc is None:
        # Prepare C++ func
        glsfunc = prepareGLSc()

    ###

    # Get prange, if not given
    # Get periods in which to compute the periodogram, if not given.
    if prange is None:
        # Construct prange from time array as in yorbit
        Pmax = 2*(t.max() - t.min())
        Pmin = np.max([0.25, np.median(np.diff(t))])
        dnu = 0.25*np.min(np.diff(t))

        NU = np.arange(1/Pmax, 1/Pmin, 1/Pmax)
        prange = 1/NU
        # np.arange(Pmin, Pmax + dnu, dnu)

        print('Using default values: Pmin: {pmin}; Pmax: {pmax}; '
              'dnu : {dnu}; {N} frequencies.'
              ''.format(pmin=Pmin, pmax=Pmax, dnu=dnu, N=len(prange)))

    y = rv.copy()
    ey = erv.copy()

    if fap is None:
        Nsimul = 1

    else:
        Nsimul = int(multfactor/np.min(fap))

    powers = np.zeros(Nsimul)

    W = np.sum(1/ey**2)

    print('Running {} simulations.'.format(Nsimul))
    print('Progress: ', end="", flush=True)
    for j in range(Nsimul):

        if (j*100/Nsimul) % 1 == 0:

            print('{:02.0f}%'.format(j*100/Nsimul), end="",
                  flush=True)
            print('\b'*3, end="", flush=True)

        pp = np.zeros(len(prange))

        w = 1/ey**2.0/W

        # EXECUTE C++ FUNCTION
        glsfunc(prange.ctypes.data_as(POINTER(c_double)), len(prange),
                t.ctypes.data_as(POINTER(c_double)),
                y.ctypes.data_as(POINTER(c_double)),
                w.ctypes.data_as(POINTER(c_double)),
                len(t), pp.ctypes.data_as(POINTER(c_double))
                )

        if fap is None:
            return pp

        if j == 0:
            pp0 = pp

        # TEMPORAL
        # powers[j] = np.sum(pp[cond])

        powers[j] = np.max(pp)

        # Shuffle data
        ind = np.arange(len(y))
        np.random.shuffle(ind)
        y = y[ind]
        ey = ey[ind]

    print('COMPLETE!')

    # Compute FAP levels
    faplevels = [np.percentile(powers, q) for q in (1 - np.array(fap))*100]

    if plot:
        fig1 = p.figure()
        ax = fig1.add_subplot(111)
        ax.hist(powers, 25, histtype='step')
        ax.axvline(powers[0], ls=':', color='k')

        fig2 = p.figure(figsize=(8, 4))
        ax2 = fig2.add_subplot(111)
        ax2.semilogx(prange, pp0, lw=1)
        for i, ff in enumerate(faplevels):
            ax2.axhline(ff, color=str(fap[i]))

    return pp0, faplevels, powers


def windowfunction(t, erv, prange):

    w = (1/erv**2)/np.sum(1/erv**2)

    wf = np.zeros(len(prange))

    for i in range(len(prange)):

        omega = 2 * np.pi / prange[i]

        ccos = np.cos(omega*t)
        ssin = np.sin(omega*t)

        wC = np.sum(w*ccos)
        wS = np.sum(w*ssin)

        wC = np.sum(ccos)/len(ccos)
        wS = np.sum(ssin)/len(ssin)

        wf[i] = wC * wC + wS * wS
    return wf


def stacked(t, rv, erv, oversampling=10, start=20, prange=None):
    """Compute stacked periodogram."""
    Deltat = t.max() - t.min()

    if prange is None:
        # Define explored periods
        pnu = np.arange(1/Deltat, 1/0.9, 1/Deltat/oversampling)
        prange = 1/pnu

    powers = np.empty((len(t)+1-start, len(prange)))

    assert start < len(t), "Minum must be smaller than the length of the data"
    for i, ind in enumerate(range(start, len(t)+1)):
        print(ind)
        tt = t[:ind]
        y = rv[:ind]
        ey = erv[:ind]

        try:
            powers[i] = GLSc(tt, y, ey, prange=prange, plot=False)[0]
        except:
            powers[i] = GLSperiodogram(tt, y, ey, prange=prange,
                                       plot=False)[1]

    return prange, powers


def linGLS(t, y, ey2, designmatrix, pnu=None, ncomponents=1,
           oversampling=5, include_bias=True, **kwargs):
    """
    Perform a GLS periodogram using an arbitrary set of basis functions.

    :param np.array designmatrix: design matrix of basis functions
    (nsamples x nfeatures)
    :param int ncomponents: number of Fourier components per frequency
    (default:1).
    """
    if pnu is None:
        # Define explored periods
        pnu = autofrequency(t, samples_per_peak=oversampling, **kwargs)

    if include_bias:
        bias = np.ones([len(t), 1])
        phi = np.hstack([bias, designmatrix])
        # designmatrix.append(lambda x: x*0.0 + 1.0)
    else:
        phi = designmatrix.copy()

    amps = np.zeros([len(pnu), phi.shape[1] + 2*ncomponents])
    pow = np.zeros(len(pnu))
    chi = np.zeros(len(pnu))
    y_pred = np.zeros([len(pnu), len(t)])
    # Chi2 of non-sinusoidal function
    popt0, epopt0, cov0 = fit.cuadminlin(t, y, ey2, phi)
    # y_pred0 = np.sum(np.array([popt0[i] * phi[:, i]
    #                            for i in range(len(basefuncs))]), axis=0)

    y_pred0 = np.dot(popt0.reshape(1, -1), phi.T).ravel()
    chi0 = np.mean((y - y_pred0)**2 / ey2)

    for i, f in enumerate(pnu):
        # create new design matrix with columns for sinusoids
        phi_nu = np.random.rand(phi.shape[0], phi.shape[1] + ncomponents*2)
        phi_nu[:, :-ncomponents*2] = phi

        for k in range(ncomponents, 0, -1):
            # Add sinusoidal base functions
            nf = k * f

            phi_nu[:, -2*k] = np.sin(2 * np.pi * nf * t)
            phi_nu[:, -2*k + 1] = np.cos(2 * np.pi * nf * t)

            # phi.append(lambda u: np.sin(2 * np.pi * nf * u))
            # phi.append(lambda u: np.cos(2 * np.pi * nf * u))

        amps[i], epopt, cov = fit.cuadminlin(t, y, ey2, phi_nu)

        y_pred[i] = np.dot(amps[i].reshape(1, -1), phi_nu.T).ravel()
        # y_pred[i] = np.sum(np.array([amps[i][j] * phi_nu[j](t)
        #                              for j in range(len(phi_nu))]), axis=0)

        chi[i] = np.mean((y - y_pred[i])**2 / ey2)
        pow[i] = (chi0 - chi[i])/chi0

    return pnu, pow, amps, chi, y_pred0, y_pred


def autofrequency(t, samples_per_peak=5, nyquist_factor=5,
                  minimum_frequency=None, maximum_frequency=None,
                  return_freq_limits=False):
    """
    Define array for frequencies based on data.

    Script copied from astropy
    """
    baseline = t.max() - t.min()
    n_samples = t.size

    df = 1.0 / baseline / samples_per_peak

    if minimum_frequency is None:
        minimum_frequency = 0.5 * df

    if maximum_frequency is None:
        avg_nyquist = 0.5 * n_samples / baseline
        maximum_frequency = nyquist_factor * avg_nyquist

    Nf = 1 + int(np.round((maximum_frequency - minimum_frequency) / df))

    if return_freq_limits:
        return minimum_frequency, minimum_frequency + df * (Nf - 1)
    else:
        return minimum_frequency + df * np.arange(Nf)
