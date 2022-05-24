# cython: language_level=3
cimport numpy as np
import numpy as np
from libc.math cimport pow, fabs
cimport cython

cdef double ROCP = 0.28571426       # R over Cp
cdef double ZEROCNK = 273.15        # Zero Celsius in Kelvins

@cython.cdivision(True)
cdef double _wobf(double t):
    cdef double npol, ppol
    t = t - 20
    if t <= 0:
        npol = 1. + t * (-8.841660499999999e-3 + t * ( 1.4714143e-4 + t * (-9.671989000000001e-7 + t * (-3.2607217e-8 + t * (-3.8598073e-10)))))
        npol = 15.13 / (pow(npol,4))
        return npol
    else:
        ppol = t * (4.9618922e-07 + t * (-6.1059365e-09 + t * (3.9401551e-11 + t * (-1.2588129e-13 + t * (1.6688280e-16)))))
        ppol = 1 + t * (3.6182989e-03 + t * (-1.3603273e-05 + ppol))
        ppol = (29.93 / pow(ppol,4)) + (0.96 * t) - 14.8
        return ppol

cdef np.ndarray _wobf_arr(np.ndarray t):
    cdef np.ndarray correction
    cdef Py_ssize_t xshape
    xshape = t.shape[0]
    correction = np.zeros(xshape, dtype=np.float64)
    for x in range(xshape):
        correction[x] = _wobf(t[x])
    return correction

def wobf(t):
    if not isinstance(t, np.ndarray):
        return _wobf(t)
    else:
        return _wobf_arr(t)

@cython.cdivision(True)
cdef double _satlift(double p, double thetam, double conv):
    if fabs(p - 1000.) - 0.001 <= 0:
        return thetam
    cdef double eor = 999
    cdef double pwrp, t1, e1, rate, t2, e2
    while fabs(eor) - conv > 0:
        if eor == 999:                  # First Pass
            pwrp = pow((p / 1000.),ROCP)
            t1 = (thetam + ZEROCNK) * pwrp - ZEROCNK
            e1 = _wobf(t1) - _wobf(thetam)
            rate = 1
        else:                           # Successive Passes
            rate = (t2 - t1) / (e2 - e1)
            t1 = t2
            e1 = e2
        t2 = t1 - (e1 * rate)
        e2 = (t2 + ZEROCNK) / pwrp - ZEROCNK
        e2 += _wobf(t2) - _wobf(e2) - thetam
        eor = e2 * rate
    return t2 - eor

def satlift(p, thetam, conv=0.1):
    if not isinstance(p, np.ndarray) and not isinstance(thetam, np.ndarray):
        return _satlift(p, thetam, conv)
    else:
        # If p and thetam are arrays
        short = np.fabs(p - 1000.) - 0.001 <= 0
        lft = np.where(short, thetam, 0)
        if np.all(short):
            return lft

        eor = 999
        first_pass = True
        while np.fabs(np.min(eor)) - conv > 0:
            if first_pass:                  # First Pass
                pwrp = np.power((p[~short] / 1000.),ROCP)
                t1 = (thetam[~short] + ZEROCNK) * pwrp - ZEROCNK
                e1 = wobf(t1) - wobf(thetam[~short])
                rate = 1
                first_pass = False
            else:                           # Successive Passes
                rate = (t2 - t1) / (e2 - e1)
                t1 = t2
                e1 = e2
            t2 = t1 - (e1 * rate)
            e2 = (t2 + ZEROCNK) / pwrp - ZEROCNK
            e2 += wobf(t2) - wobf(e2) - thetam[~short]
            eor = e2 * rate
        lft[~short] = t2 - eor
        return lft

@cython.cdivision(True)
cdef double _vappres(double t):
    cdef double pol
    pol = t * (1.1112018e-17 + (t * -3.0994571e-20))
    pol = t * (2.1874425e-13 + (t * (-1.789232e-15 + pol)))
    pol = t * (4.3884180e-09 + (t * (-2.988388e-11 + pol)))
    pol = t * (7.8736169e-05 + (t * (-6.111796e-07 + pol)))
    pol = 0.99999683 + (t * (-9.082695e-03 + pol))
    return 6.1078 / pol**8

cdef np.ndarray _vappres_arr(np.ndarray t):
    cdef np.ndarray pres
    cdef Py_ssize_t xshape
    xshape = t.shape[0]
    pres = np.zeros(xshape, dtype=np.float64)
    for x in range(xshape):
        pres[x] = _vappres(t[x])
    return np.ma.array(pres)

def vappres(t):
    if isinstance(t, np.ndarray):
        return _vappres_arr(t)
    else:
        return _vappres(t)

@cython.cdivision(True)
cdef double _mixratio(double p, double t):
    cdef double x, wfw, fwesw
    x = 0.02 * (t - 12.5 + (7500. / p))
    wfw = 1. + (0.0000045 * p) + (0.0014 * x * x)
    fwesw = wfw * _vappres(t)
    return 621.97 * (fwesw / (p - fwesw))

cdef np.ndarray _mixratio_arr(np.ndarray p, np.ndarray t):
    cdef np.ndarray mr
    cdef Py_ssize_t xshape
    xshape = t.shape[0]
    mr = np.zeros(xshape, dtype=np.float64)
    for x in range(xshape):
        mr[x] = _mixratio(p[x], t[x])
    return np.ma.array(mr)

def mixratio(p, t):
    if isinstance(p, np.ndarray) and isinstance(t, np.ndarray):
        return _mixratio_arr(p, t)
    else:
        return _mixratio(p, t)