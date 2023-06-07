# % Computes the lLegendre nodes and weights using asymptotic formulae and
# % Newton's method.

import numpy as np
import math

def legpts_asy(n):
    nbody = 10;
    bdyidx1 = np.arrange(n-(nbody-1),n);
    bdyidx2 = np.arrange(nbody, 1, -1);

    if n <= 2*nbody:
        [xbody, wbody] = legpts_asy2_bdy(n, math.ceil(n/2));
        [xbody2, wbody2] = legpts_asy2_bdy(n, math.floor((n/2)));
        x = np.array(xbody2[::-1], xbody);
        w = np.array(wbody2[::-1], wbody);
        return [x, w];


    # Interior algorithm:
    [x, w] = legpts_asy1(n);

    # Boundary algorithm:
    [xbody, wbody] = legpts_asy2_bdy(n,nbody);

    # Output the result:
    x[bdyidx1] = xbody;
    w[bdyidx1] = wbody;
    x[bdyidx2] = -xbody;
    w[bdyidx2] = wbody;