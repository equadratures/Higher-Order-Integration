import numpy as np
import math


def legpts_asy2_bdy(n, npts):
    rho = n + 0.5
    rho2 = n - 0.5

    jk = np.array([2.404825557695773, 5.520078110286310, 8.653727912911012,
                   11.791534439014281, 14.930917708487785, 18.071063967910922,
                   21.211636629879258, 24.352471530749302, 27.493479132040254,
                   30.634606468431975, 33.775820213573568])

    if npts > 11:
        p = ((np.arange(len(jk) + 1, npts) - 0.25) * np.pi).astype(float)
        aa = np.array([0.000813005721543268, 0, 0.0245988241803681, 0,
                       0.131420807470708, 0, 0.0682894897349453])
        bb = np.array([0.00650404577261471, 0, 0.200991122197811, 0,
                       1.16837242570470, 0, 1, 0])
        jk = np.concatenate((jk, p + np.polyval(aa, p) / np.polyval(bb, p)))
    jk = jk[:npts]

    phik = jk / rho
    t = phik + (phik * np.cot(phik) - 1) / (8 * phik * rho ** 2)

    tB1, A2, tB2, A3 = asy2_higherterms(0, 0)

    dt = np.inf
    j = 0
    while np.linalg.norm(dt, np.inf) > np.sqrt(np.finfo(float).eps) / 200:
        vals, ders = feval_asy(n, t, 0)
        dt = vals / ders
        t += dt
        j += 1
        if j > 10:
            dt = 0

    _, ders = feval_asy(n, t, 1)

    t = t[::-1][:npts]
    ders = ders[::-1][:npts]
    x = np.cos(t)
    w = (2 / ders ** 2).reshape(-1, 1)
    v = np.sin(t) / ders

    return x, w, v, t, ders


def feval_asy(n, t, flag):
    Ja = besselj(0, n * t)
    Jb = besselj(1, n * t)
    Jbb = besselj(1, (n - 1) * t)

    if not flag:
        Jab = besselj(0, (n - 1) * t)
    else:
        Jab = besseltaylor(-t, n * t)

    gt = 0.5 * (np.cot(t) - 1 / t)
    gtdt = 0.5 * (-1 / np.sin(t) ** 2 + 1 / t ** 2)
    tB0 = 0.25 * gt
    A1 = gtdt / 8 - 1 / 8 * gt / t