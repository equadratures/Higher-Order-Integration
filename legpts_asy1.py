import numpy as np
import math

def legpts_asy1(n, mint=None):
    is_odd = n % 2
    k = np.arange((n - 2 + is_odd) / 2 + 1, 0, -1)
    theta = np.pi * (4 * k - 1) / (4 * n + 2)
    x = (1 - (n - 1) / (8 * n ** 3) - 1 / (384 * n ** 4) * (39 - 28 / np.sin(theta) ** 2)) * np.cos(theta)
    t = np.arccos(x)

    if mint is None:
        mint = t[-10] if len(t) > 10 else t[-1]
    elif mint == 0:
        mint = t[-1]

    idx = np.max(np.where(t < mint)[0]) - 1
    dt = np.inf
    j = 0

    while np.linalg.norm(dt, np.inf) > np.sqrt(np.finfo(float).eps) / 1000:
        vals, ders = feval_asy(n, t, mint, 0)
        dt = vals / ders
        t -= dt
        j += 1
        dt = dt[:idx]
        if j > 10:
            dt = 0

    vals, ders = feval_asy(n, t, mint, 1)
    t -= vals / ders

    x = np.cos(t)
    w = 2 / ders ** 2
    v = np.sin(t) / ders

    if is_odd:
        x = np.concatenate((-x[::-1][1:], x))
        w = np.concatenate((w[::-1][1:], w))
        v = np.concatenate((-v[::-1][1:], v))
        ders = np.concatenate((ders[::-1][1:], ders))
    else:
        x = np.concatenate((-x[::-1], x))
        w = np.concatenate((w[::-1], w))
        v = np.concatenate((-v[::-1], v))
        ders = np.concatenate((ders[::-1], ders))

    return x, w, v, ders, t


def feval_asy(n, t, mint, flag):
    M = 20
    c = np.cumprod(np.arange(1, 2 * M, 2) / np.arange(2, 2 * M + 2, 2))
    d = np.cumprod(np.arange(1, 2 * M, 2) / np.arange(2 * n + 3, 2 * (n + M) + 2, 2))
    c *= d
    R = (8 / np.pi) * c / (2 * np.sin(mint)) ** np.arange(0.5, M + 1) / 10
    R = R[np.abs(R) > np.finfo(float).eps]
    M = len(R)
    c = c[:M]

    ds = -1 / (8 * n)
    s = ds
    j = 1

    while np.abs(ds / s) > np.finfo(float).eps / 100:
        j += 1
        ds = -0.5 * (j - 1) / (j + 1) / n * ds
        s += ds

    p2 = np.exp(s) * np.sqrt(4 / (n + 0.5) / np.pi)
    g = np.array([1, 1 / 12, 1 / 288, -139 / 51840, -571 / 2488320, 163879 / 209018880,
                  5246819 / 75246796800, -534703531 / 902961561600, -4483131259 / 86684309913600,
                  432261921612371 / 514904800886784000])
    fn = np.sum(g * np.append(1, np.cumprod(np.ones(9) / n)))
    fn5 = np.sum(g * np.append(1, np.cumprod(np.ones(9) / (n + 0.5))))
    C = p2 * (fn / fn5)

    onesT = np.ones_like(t)
    onesM = np.ones(M)
    M05 = np.arange(M) + 0.5
    onesMcotT = onesM * np.tan(t)
    M05onesT = np.outer(M05, onesT)
    alpha = np.outer(onesM, n * t) + M05onesT * (np.outer(onesM, t - 0.5 * np.pi))
    cosAlpha = np.cos(alpha)
    sinAlpha = np.sin(alpha)

    if flag:
        k = np.arange(t.size - 1, -1, -1)
        rho = n + 0.5
        ta = t.astype(np.float32)
        tb = t - ta
        hi = rho * ta
        lo = rho * tb
        pia = np.float32(np.pi)
        pib = np.float32(-8.742278000372485e-08)
        dh = (hi - (k - 0.25) * pia) + lo - (k - 0.25) * pib
        tmp = np.zeros_like(t, dtype=np.float64)
        sgn = 1
        fact = 1
        DH = dh
        dh2 = dh * dh

        for j in range(21):
            dc = sgn * DH / fact
            tmp += dc
            sgn = -sgn
            fact *= (2 * j + 3) * (2 * j + 2)
            DH *= dh2
            if np.linalg.norm(dc, np.inf) < np.finfo(float).eps / 1000:
                break

        tmp[1::2] = -tmp[1::2]
        tmp *= np.sign(cosAlpha[0, 1] * tmp[1])
        cosAlpha[0] = tmp

    twoSinT = 2 * np.sin(t)
    denom = np.cumprod(twoSinT) / np.sqrt(twoSinT)
    vals = C * np.dot(c, cosAlpha / denom)
    numer = M05onesT * (cosAlpha * onesMcotT + sinAlpha) + n * sinAlpha
    ders = -C * np.dot(c, numer / denom)

    return vals, ders
