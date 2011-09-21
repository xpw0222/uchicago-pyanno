#!/usr/bin/env python
"""
  estimates model parameters under models A and Bt

  *modelnumber*  -- 1 for model A,
                    2 for Bt
  *numberOfRuns* -- number of independent runs with a randomly
                    chosen starting point in parameter space
  *usepriors* -- 1 or 0; if set to 1, uses beta-priors for
                    theta-parameters with (a,b) set to (2,1)
  *estimatealphas* -- 1 or 0; if set to 1, alpha-parameters are used
                    explicitly in the likelihood function
                    (to get proper ML or MAP estimates, in the text
                    the model is described as A-with-Alphas); if set to 0,
                    alpha-parameters are computed as a function of
                    estimates of omega-parameters
    *useomegas* -- 1 or 0 (use or don't estimated code frequencies in
                    initializing gamma-parameters of model Bt)


  *dim* -- total number of admissible annotation values

  model A requires:
                        *estimateOmegas8*
                        *expPat*
                        *getAlphas*
                        *getBeta*
                        *likeA*
                        *likeA8*
                        *random_start8*
                        *computePosteriorA*

 model B requires:
                        *likeBt*
                        *likeBt8*
                        *patternFrequenciesBt*
                        *random_startBt8*
                        *computePosteriorBt*

 shared service functions:

                        *check_data*
                        *compute_counts*
                        *format_posteriors*
                        *string_wrap*
                        *print_wrap*
                        *array_format*
                        *unique*
                        *num2comma*

    *Reference to the modeling details:*

    Rzhetsky A, Shatkay H, Wilbur WJ (2009)
    How to Get the Most out of Your Curation Effort.
    PLoS Comput Biol 5(5): e1000391.
    doi:10.1371/journal.pcbi.1000391
"""

from numpy import *
import numpy as np
import scipy as sp
from scipy import *
from pylab import *
import time
import random
import shelve
import scipy.optimize
import scipy.stats
from time import strftime
from matplotlib import *
from enthought.traits.api import HasTraits, Str, Int, Range, Bool,\
    Array, Enum, Dict, File, on_trait_change, Button
from enthought.traits.ui.api import View, Item, Group, Handler
from enthought.traits.ui.menu import CancelButton, ApplyButton, Action

#========================================================
#========================================================
#                Model Bt
#--------------------------------------------------------
def random_startBt8(dim, alphaslikeomegas, counts, report):
    """Returns a random initial set of parameters.

    Input:
    dim -- number of annotation values
    alphaslikeomega -- use omegas; 0: False, 1: True
    counts -- input data
    report -- verbosity level; one of 'Essentials', 'Everything', 'Nothing'

    Output:
    x -- array of initial, random parameters
         dim-1 + 8 parameters:
           dim-1 are the parameters gamma_i, i.e. P(label=i)
                 the missing one is computed as 1-sum(gamma_1:dim-1)
           8 are the parameters theta_i
    """
    x = sp.zeros((dim - 1) + 8, float)
    ii = 0

    if alphaslikeomegas == 0:
        # don't use omegas
        tmp = sp.random.rand(dim)
        tmp.sort()

        for i in range(1, dim):
            x[ii] = tmp[i] - tmp[i - 1]
            ii += 1

    else:
        omegas = array(estimateOmegas8(counts, dim, report), float)
        x[0:dim - 1] = omegas[0:dim - 1]
        ii = dim - 1

    for k in range(8):
        x[ii] = 0.95 + random.random() * 0.05
        ii += 1

    if cmp(report, 'Nothing') != 0:
        print "Start:" + str(x)

    return x


#--------------------------------------------------------
#
def likeBt8(x, arguments):
    """Log-likelihood of B-with-theta model, for 3 annotators, 8 annotations.

    Input:
    x -- current model parameters
    arguments -- (counts, dim, usepriors)
                 counts is the input data in count format

    Output:
    l -- log-likelihood
    """
    ind = sp.array([[0, 1, 2],
                    [1, 2, 3],
                    [2, 3, 4],
                    [3, 4, 5],
                    [4, 5, 6],
                    [5, 6, 7],
                    [0, 6, 7],
                    [0, 1, 7]], dtype=int)

    counts, dim, usepriors = arguments

    l = 0
    # xx holds the 3 gamma parameters and 3 parameters for each combination
    # of annotators
    xx = sp.zeros(dim-1 + 3, dtype=float)
    xx[0:dim-1] = x[0:dim-1]
    # loop over the 8 combinations of annotators
    for i in range(8):
        param_indices = ind[i,:] + dim - 1
        xx[dim-1:dim+2] = x[param_indices]
        l += likeBt(xx, counts[:,i], dim, usepriors)

    return l


#--------------------------------------------------------
def _patternFrequenciesBt(dim, gam, t, v_ijk_combinations):
    """Compute vector of P(v_{ijk} | params) for each combination of v_{ijk}."""
    pf = 0.
    not_theta = (1.-t) / (dim-1.)
    for psi in range(dim):
        p_v_ijk_given_psi = np.empty_like(v_ijk_combinations, dtype=float)
        for j in range(3):
            p_v_ijk_given_psi[:,j] = np.where(v_ijk_combinations[:,j]==psi,
                                              t[j], not_theta[j])
        pf += p_v_ijk_given_psi.prod(1) * gam[psi]
    return pf


#-------------------------------------------------------
def likeBt(x, data, dim, usepriors):
    """Compute the log likelihood of data for one triplet of annotators.

    Input:
    x -- model parameters (for one triplet of annotators)
    data -- input data for one combination of annotators in count format
    dim -- number of different annotation values
    usepriors -- use prior? 0: False, 1: True
    """

    # unpack parameters vector

    # gamma is x, with last element fixed to sum to one
    gam = sp.zeros(dim, dtype=float)
    gam[0:dim - 1] = x[0:dim - 1]
    gam[dim - 1] = 1 - sum(gam[0:dim - 1])

    # only the theta parameters
    t = x[dim-1:dim+2]

    # TODO: check if it's possible to replace these constraints with bounded optimization
    if min(min(gam), min(t)) < 0.0 or max(max(gam), max(t)) > 1.0:
        return Inf

    # TODO: replace log(beta) with expression using scipy.special.betaln
    if usepriors == 1:
        # if requested, add prior over theta to log likelihood
        l = (log(scipy.stats.beta.pdf(t[0], 2, 1))
             + log(scipy.stats.beta.pdf(t[1], 2, 1))
             + log(scipy.stats.beta.pdf(t[2], 2, 1)))
    else:
        l = 0.0

    # log \prod_n P(v_{ijk}^{n} | params)
    # = \sum_n log P(v_{ijk}^{n} | params)
    # = \sum_v_{ijk}  count(v_{ijk}) P( v_{ijk} | params )
    #
    # where n is n-th annotation of triplet {ijk}]

    # list of all possible combinations of v_i, v_j, v_k elements
    v_ijk_combinations = np.array([i for i in np.ndindex(dim,dim,dim)])
    # compute P( v_{ijk} | params )
    pf = _patternFrequenciesBt(dim, gam, t, v_ijk_combinations)

    l += (data * sp.log(pf)).sum()


    return -l


#-------------------------------------------------------
def computePosteriorBt(v1, v2, v3, t1, t2, t3, gam, dim):
    dist = zeros(dim, float)

    for i in range(dim):
        dist[i] = probGivenTrueB(v1, v2, v3, t1, t2, t3, i, gam[i], dim)

    summa = sum(dist)
    dist = dist / summa

    return dist


#-------------------------------------------------------
def probGivenTrueB(v1, v2, v3, t1, t2, t3, vt, gamt, dim):
    if v1 == vt:
        tt1 = t1
    else:
        tt1 = (1 - t1) / (dim - 1)

    if v2 == vt:
        tt2 = t2
    else:
        tt2 = (1 - t2) / (dim - 1)

    if v3 == vt:
        tt3 = t3
    else:
        tt3 = (1 - t3) / (dim - 1)

    return gamt * tt1 * tt2 * tt3


#=======================================================
#=======================================================
#                Model A
#-------------------------------------------------------
def computePosteriorA(vijk, tijk, alphas, dimension):
#
# *vijk* -- values provided by annotators *i*, *j*, and *k*
# *tijk* -- correctness values for annotators *i*, *j*, and *k*
#  *alphas* -- pre-computed values of alpha-parameters
#  *dimension* -- the number of permissible annotation values
#
#  *posteriors*  -- posterior probability values for annotations 1, 2, ..., *dimension*
#  note: annotation values *vijk(i)* -- *integer* values between 1 and *dimension*

    posteriors = zeros(dimension, float)

    #-----------------------------------------------
    # aaa
    if vijk[0] == vijk[1] and vijk[1] == vijk[2]:
        x1 = tijk[0] * tijk[1] * tijk[2]
        x2 = (1 - tijk[0]) * (1 - tijk[1]) * (1 - tijk[2])
        p1 = x1 / (x1 + alphas[3] * x2)
        p2 = (1 - p1) / (dimension - 1)

        for j in range(dimension):
            if vijk[0] == j:
                posteriors[j] = p1
            else:
                posteriors[j] = p2
            #-----------------------------------------------
            # aaA
    elif vijk[0] == vijk[1] and vijk[1] != vijk[2]:
        x1 = tijk[0] * tijk[1] * (1 - tijk[2])
        x2 = (1 - tijk[0]) * (1 - tijk[1]) * tijk[2]
        x3 = (1 - tijk[0]) * (1 - tijk[1]) * (1 - tijk[2])

        # a is correct
        p1 = x1 / (x1 + alphas[2] * x2 + alphas[4] * x3)

        # A is correct
        p2 = (alphas[2] * x2) / (x1 + alphas[2] * x2 + alphas[4] * x3)

        # neither
        p3 = (1 - p1 - p2) / (dimension - 2)

        for j in range(dimension):
            if vijk[0] == j:
                posteriors[j] = p1
            elif vijk[2] == j:
                posteriors[j] = p2
            else:
                posteriors[j] = p3
            #-----------------------------------------------
            # aAa
    elif vijk[0] == vijk[2] and vijk[1] != vijk[2]:
        x1 = tijk[0] * (1 - tijk[1]) * tijk[2]
        x2 = (1 - tijk[0]) * tijk[1] * (1 - tijk[2])
        x3 = (1 - tijk[0]) * (1 - tijk[1]) * (1 - tijk[2])

        # a is correct
        p1 = x1 / (x1 + alphas[1] * x2 + alphas[5] * x3)

        # A is correct
        p2 = (alphas[1] * x2) / (x1 + alphas[1] * x2 + alphas[5] * x3)

        # neither
        p3 = (1 - p1 - p2) / (dimension - 2)

        for j in range(dimension):
            if vijk[0] == j:
                posteriors[j] = p1
            elif vijk[1] == j:
                posteriors[j] = p2
            else:
                posteriors[j] = p3
            #-----------------------------------------------
            # Aaa
    elif vijk[1] == vijk[2] and vijk[0] != vijk[2]:
        x1 = (1 - tijk[0]) * tijk[1] * tijk[2]
        x2 = tijk[0] * (1 - tijk[1]) * (1 - tijk[2])
        x3 = (1 - tijk[0]) * (1 - tijk[1]) * (1 - tijk[2])

        # a is correct
        p1 = x1 / (x1 + alphas[0] * x2 + alphas[6] * x3)

        # A is correct
        p2 = (alphas[0] * x2) / (x1 + alphas[0] * x2 + alphas[6] * x3)

        # neither
        p3 = (1 - p1 - p2) / (dimension - 2)

        for j in range(dimension):
            if vijk[0] == j:
                posteriors[j] = p2
            elif vijk[2] == j:
                posteriors[j] = p1
            else:
                posteriors[j] = p3
            #-----------------------------------------------
            # aAb
    elif vijk[0] != vijk[1] and vijk[1] != vijk[2]:
        x1 = tijk[0] * (1 - tijk[1]) * (1 - tijk[2])
        x2 = (1 - tijk[0]) * tijk[1] * (1 - tijk[2])
        x3 = (1 - tijk[0]) * (1 - tijk[1]) * tijk[2]
        x4 = (1 - tijk[0]) * (1 - tijk[1]) * (1 - tijk[2])

        summa1 = 1 - alphas[3] - alphas[4] - alphas[5] - alphas[6]
        summa2 = (1 - alphas[0]) * x1 + (1 - alphas[1]) * x2 + (1 - alphas[
                                                                    2]) * x3 + summa1 * x4

        # a is correct
        p1 = (1 - alphas[0]) * x1 / summa2

        # A is correct
        p2 = (1 - alphas[1]) * x2 / summa2

        # b is correct
        p3 = (1 - alphas[2]) * x3 / summa2

        # (a, A, b) are all incorrect
        p4 = (summa1 * x4 / summa2) / (dimension - 3)

        for j in range(dimension):
            if vijk[0] == j:
                posteriors[j] = p1
            elif vijk[1] == j:
                posteriors[j] = p2
            elif vijk[2] == j:
                posteriors[j] = p3
            else:
                posteriors[j] = p4

    # check posteriors: non-negative, sum to 1
    if sum(posteriors) - 1.0 > 0.0000001 or min(posteriors) < 0:
        print 'Aberrant posteriors!!!', posteriors
        print sum(posteriors)
        print min(posteriors)
        time.sleep(60)

    return posteriors


#-------------------------------------------------------------
def random_startA8(estimatealphas, report):
    if estimatealphas == 1:
        x = zeros(11, float)
    else:
        x = zeros(8, float)

    for i in range(8):
        x[i] = 0.5 + random.random() * 0.5

    if estimatealphas == 1:
        for i in range(8, 11):
            x[i] = random.random() / 4.

    if cmp(report, 'Nothing') != 0:
        print "Start:" + str(x)
    return x


#-------------------------------------------------------------
def getBeta(omegas, i1, i2, i3):
    n = len(omegas)
    onebeta = []

    if i1 == i2 and i2 == i3:
        tmp3 = 0
        for i in range(n):
            tmp3 += omegas[i] ** 3

    if (i1 == i2 and i2 != i3) or\
       (i1 == i3 and i2 != i3) or\
       (i2 == i3 and i1 != i3):
        tmp = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    tmp += (omegas[i] ** 2) * omegas[j]

    if i1 != i2 and i2 != i3 and i1 != i3:
        tmp1 = 0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i != j and j != k and i != k:
                        tmp1 += omegas[i] * omegas[j] * omegas[k]

    if i1 == i2 and i2 == i3:
        return (omegas[i1] ** 3) / tmp3

    elif i1 == i2 and i2 != i3:
        return (omegas[i1] ** 2) * omegas[i3] / tmp

    elif i1 == i3 and i2 != i3:
        return (omegas[i1] ** 2) * omegas[i2] / tmp

    elif i2 == i3 and i1 != i3:
        return omegas[i1] * omegas[i2] ** 2 / tmp

    elif i1 != i2 and i2 != i3 and i1 != i3:
        if tmp1 == 0:
            tmp1 = 1
        return omegas[i1] * omegas[i2] * omegas[i3] / tmp1

    else:
        print 'Unexpected condition!!!'

    return onebeta


#-------------------------------------------------------------
def expPat(alphas, omegas, theta1, theta2, theta3, i1, i2, i3):
    if i1 == i2 and i2 == i3:
        p = (theta1 * theta2 * theta3 +\
             (1 - theta1) * (1 - theta2) * (1 - theta3) * alphas[3])
        p *= getBeta(omegas, i1, i2, i3)
        return p

    elif i1 == i2 and i2 != i3:
        p = (theta1 * theta2 * (1 - theta3) +\
             (1 - theta1) * (1 - theta2) * theta3 * alphas[2] +\
             (1 - theta1) * (1 - theta2) * (1 - theta3) * alphas[4])
        p *= getBeta(omegas, i1, i2, i3)
        return p

    elif i1 == i3 and i2 != i1:
        p = (theta1 * (1 - theta2) * theta3 +\
             (1 - theta1) * theta2 * (1 - theta3) * alphas[1] +\
             (1 - theta1) * (1 - theta2) * (1 - theta3) * alphas[5])
        p *= getBeta(omegas, i1, i2, i3)
        return p

    elif i2 == i3 and i1 != i2:
        p = ((1 - theta1) * theta2 * theta3 +\
             theta1 * (1 - theta2) * (1 - theta3) * alphas[0] +\
             (1 - theta1) * (1 - theta2) * (1 - theta3) * alphas[6]);
        p *= getBeta(omegas, i1, i2, i3)
        return p

    elif i1 != i2 and i2 != i3 and i1 != i3:
        p = ( (1 - theta1) * (1 - theta2) * (1 - theta3) *\
              (1 - alphas[3] - alphas[4] - alphas[5] - alphas[6]) +\
              theta1 * (1 - theta2) * (1 - theta3) * (1 - alphas[0]) +\
              (1 - theta1) * theta2 * (1 - theta3) * (1 - alphas[1]) +\
              (1 - theta1) * (1 - theta2) * theta3 * (1 - alphas[2]) )
        p *= getBeta(omegas, i1, i2, i3)
        return p

    else:
        print 'Unexpected condition!!'

    return p


#----------------------------------------------------------
def estimateOmegas8(counts, dim, report):
    omegas = zeros(dim, float)
    ii = 0

    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for nchunks in range(8):
                    omegas[i] += counts[ii, nchunks]
                    omegas[j] += counts[ii, nchunks]
                    omegas[k] += counts[ii, nchunks]
                ii += 1

    omegas = omegas / (3. * sum(sum(counts)))
    if cmp(report, 'Nothing') != 0:
        print "Omegas: " + str(omegas)

    return omegas


#----------------------------------------------------------
def likeA8(x, arguments):
    ind = array([[0, 1, 2],\
        [1, 2, 3],\
        [2, 3, 4],\
        [3, 4, 5],\
        [4, 5, 6],\
        [5, 6, 7],\
        [0, 6, 7],\
        [0, 1, 7]], int)

    alphas, omegas, counts, usepriors, estimatealphas, dim = arguments
    l = 0

    for i in range(8):
        if estimatealphas == 1:
            xx = zeros(6, float)
            for j in range(3):
                xx[j] = x[int(ind[i, j])]
                xx[j + 3] = x[8 + j]
        else:
            xx = zeros(3, float)
            for j in range(3):
                xx[j] = x[int(ind[i, j])]

        l -= likeA(xx, alphas, omegas, counts[:, i], usepriors, estimatealphas,
                   dim)

    # returns - log L
    return l


#----------------------------------------------------------
def likeA(x, alphas, omegas, counts, usepriors, estimatealphas, dim):
    like = -Inf

    if estimatealphas == 1 and x[4] + 3 * x[5] > 1:
        return like

    if amin(x) <= 0 or amax(x) > 1:
        return like

    al = zeros(len(alphas))

    for i in range(len(alphas)):
        al[i] = alphas[i]

    if estimatealphas == 1:
        al[0] = x[3]
        al[1] = x[3]
        al[2] = x[3]
        al[3] = x[4]
        al[4] = x[5]
        al[5] = x[5]
        al[6] = x[5]


    # prior on *thetas*
    if usepriors == 1:
        like = log(scipy.stats.beta.pdf(x[0], 2, 1))\
               + log(scipy.stats.beta.pdf(x[1], 2, 1))\
        + log(scipy.stats.beta.pdf(x[2], 2, 1))
    else:
        like = 0

    ii = 0
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                like += counts[ii] * log(
                    expPat(al, omegas, x[0], x[1], x[2], i, j, k))
                ii += 1
    return like


#----------------------------------------------------------
def getAlphas(omega):
    n = len(omega)
    alphas = zeros(8, float)
    # dublet sums
    s2 = zeros(n, float)
    s3 = zeros(n, float)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if i != k and j != k:
                    s2[k] += omega[i] * omega[j]
                    # triplet sums

    for k in range(n):
        for i in range(n):
            for j in range(n):
                for l in range(n):
                    if i != k and j != k and l != k:
                        s3[k] += omega[i] * omega[j] * omega[l]

    # a1 / a2 / a3
    a1 = 0
    for i in range(n):
        tmp = 0
        for j in range(n):
            if j != i:
                tmp += omega[j] ** 2
        a1 += omega[i] * tmp / s2[i]

    alphas[0] = a1
    alphas[1] = a1
    alphas[2] = a1

    # a4

    a4 = 0
    for i in range(n):
        tmp = 0
        for j in range(n):
            if j != i:
                tmp += omega[j] ** 3
        a4 += omega[i] * tmp / s3[i]

    alphas[3] = a4

    # a5, a6, a7

    a5 = 0
    for i in range(n):
        tmp = 0
        for j in range(n):
            for k in range(n):
                if j != i and k != i and j != k:
                    tmp += omega[k] * omega[j] ** 2
        a5 += omega[i] * tmp / s3[i]

    alphas[4] = a5
    alphas[5] = a5
    alphas[6] = a5

    return alphas


#==========================================================
#==========================================================
#              Service functions
#------ compute unique list:  --------------------------
def unique(seq, keepstr=True):
    se = set(seq)
    s = list(se)
    return s


#-------------------------------------------------------------
def num2comma(num):
    if num == Inf:
        return "Infinity"
    elif num == -Inf:
        return "-Infinity"
    elif num == 0:
        return "0"

    order = log10(num)
    if (order / 3) > int(order / 3):
        groups = int(order / 3 + 1)
    else:
        groups = int(order / 3)

    x = zeros(groups)
    prev = num
    s = ""
    for i in range(groups):
        x[i] = int(prev / 10 ** (3 * (groups - i - 1)))
        prev -= int(x[i] * 10 ** (3 * (groups - i - 1)))
        if i < groups - 1:
            s += str(int(x[i])) + ","
        else:
            s += str(int(x[i]))

    return s


#----------------------------------------------------------
def format_posteriors(mat, alphas, dimension, thetas, gam, modelnumber):
    m = mat.shape[0]
    posteriors = zeros([m, dimension], float)
    mat = array(mat)
    thetas = array(thetas)

    i = 0
    for row in mat:
        ind = where(row >= 0)
        ind1 = where(row < 0)
        vijk = row[ind]
        tijk = thetas[ind].copy()
        if modelnumber == 1:
            p = computePosteriorA(vijk, tijk, alphas, dimension)
        else:
            p = computePosteriorBt(vijk[0], vijk[1], vijk[2], tijk[0], tijk[1],
                                   tijk[2], gam, dimension)
        posteriors[i, :] = p
        i += 1

    return posteriors


#----------------------------------------------------------
def string_wrap(st, mode):
    st = str(st)

    if mode == 1:
        st = "\033[1;29m" + st + "\033[0m"
    elif mode == 2:
        st = "\033[1;34m" + st + "\033[0m"
    elif mode == 3:
        st = "\033[1;44m" + st + "\033[0m"
    elif mode == 4:
        st = "\033[1;35m" + st + "\033[0m"
    elif mode == 5:
        st = "\033[1;33;44m" + st + "\033[0m"
    elif mode == 5:
        st = "\033[1;47;34m" + st + "\033[0m"
    else:
        st = st + ' '
    return st


#----------------------------------------------------------
def print_wrap(st):
    print "\033[01;34m" + str(st) + "\033[0m"
    return 0


#----------------------------------------------------------
def array_format(arr, format):
    s = ""
    for a in arr:
        s += str(format % a)
    return s


#----------------------------------------------------------
def read_labels(filein):
    annotators = []
    codes = []

    if not os.path.isfile(filein):
        print "Label definition file <" + filein + "> does not exist!!!"
        return annotators, codes

    for line in open(filein):
        if not line:
            break
        elif line.strip() == "":
            continue

        line = line.strip()

        if cmp(line, 'ANNOTATORS:') == 0:
            tmplist = annotators
        elif cmp(line, 'CODES:') == 0:
            tmplist = codes
        else:
            tmplist.append(line)

    return annotators, codes


#-------------------------------------------------------------
def check_data(filename, optional_labels, report):
    """
    Output
    mat -- array of annotations (integer array, nitems x 8)
    dim -- number of distinct annotation values
    values -- list of possible annotation values (same as in file - 1
    imap -- value-to-index map
    originaldata -- same as mat, without subtracting 1
    annotators -- ??? (from read_labels)
    codes -- ??? (from read_labels)
    n -- number of annotations
    """
    n = 0

    # TODO use numpy loadtxt
    # open file a first time to check consistency
    a = zeros(8)
    for line in open(filename):
        if not line:
            break
        elif line.strip() == "":
            continue
        n += 1
        j = 0
        for word in line.split():
            a[j] = int(word.replace(',', '').strip())
            j += 1

        if j != 8:
            print "Line #" + str(n) + " is ill-formed: " + str(j)
            print line

    if cmp(report, 'Nothing') != 0:
        print string_wrap(num2comma(n) + " lines in file " + filename + '...',
                          1)

    # open file a second time and read the data
    aa = zeros([n, 8], int)
    i = 0
    for line in open(filename):
        if not line:
            break
        elif line.strip() == "":
            continue

        j = 0
        for word in line.split():
            aa[i, j] = int(word.replace(',', '').strip())
            j += 1
        i += 1

    # Try to read labels
    annotators = None
    codes = None
    if optional_labels is not None:
        annotators, codes = read_labels(optional_labels)

    # Creating image
    #====================================================
    m, dd, ii = plot_annotators(aa, annotators, n, filename)
    #====================================================

    #----------------------------
    # create list of annotation values
    values = sp.zeros(m, int)
    j = 0
    for i in range(len(ii)):
        if ii[i] >= 0:
            values[j] = int(ii[i]) - 1
            j += 1

    # values of annotations have to be integer and positive
    # imap is a value --> to index map
    mm = sp.amax(values)
    imap = sp.zeros(mm + 1, int)
    # ??? I think this is broken: it should be imap[values[i]] = i
    for i in range(m):
        imap[values[i] - 1] = i

    if cmp(report, 'Nothing') != 0:
        print " "

    # data: make sure that -1 stands for no annotation
    # the rest of codes orea 0 to dim-1
    mat = sp.zeros([n, 8], int)

    for i in range(n):
        for j in range(8):
            if aa[i, j] > 0:
                mat[i, j] = aa[i, j] - 1
            else:
                mat[i, j] = aa[i, j]

    return mat, m, values, imap, aa, annotators, codes, n


#------------------------------------------------------
def plot_annotators(aa, annotators, n, filename):
    pyplot.figure(num=None, dpi=100, facecolor='w', edgecolor='k')
    hold(True)
    xlabel('Annotators')
    ylabel('Units of annotation')

    bb = zeros([n, 8])

    for i in range(n):
        for j in range(8):
            if aa[i, j] >= 0:
                bb[i, j] = 1
            else:
                bb[i, j] = 0

    cc = [0]
    k = 0

    for i in range(1, n):
        l = 0
        for j in range(8):
            if bb[i, j] != bb[cc[k], j] and l == 0:
                cc.append(i)
                k += 1
                l = 1

    l = 0
    dd = zeros([k + 1, 8])
    for i in range(k + 1):
        for j in range(8):
            dd[i, j] = 1 - bb[cc[i], j]
    spy(dd)
    hot()
    # plot

    # dimension is the number of distinct values excluding "-1" (no value)

    ee = list(aa.flatten(1))
    ii = unique(ee)  # list of annotations values(including -1)
    m = len(ii) - 1  # number of distinct values for annotations

    title(filename + ": " + str(m) + ' distinct annotation values')

    tty = []
    for i in range(k + 1):
        tty.append(num2comma(cc[i]))

    tty.append(num2comma(n))

    yticks(arange(k + 2) - 0.5, tty)

    if annotators is None:
        ttx = []
        for i in range(8):
            ttx.append(str(i + 1))
    else:
        ttx = annotators

    xticks(arange(8), ttx)
    show()
    return m, dd, ii


#----------------------------------------------------------
def compute_counts(mat, dim):
    """Transform data in counts format.
    Input:
    mat -- Input data (integer array, nitems x 8)
    dim -- number of annotation values (# classes)

    Ouput:
    data -- data[i,j] is the number of times the combination of annotators
             number `j` voted according to pattern `i`
             (integer array, dim^3 x 8)
    """
    index = array([[0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7],
        [0, 6, 7],
        [0, 1, 7]], int)
    m = mat.shape[0]
    n = mat.shape[1]
    mat = sp.asarray(mat, dtype=int)

    if n != 8:
        print 'Strange: ' + str(n) + 'annotator number !!!'


    # compute counts of 3-annotator patterns for 8 triplets
    # of annotators

    data = sp.zeros([dim ** 3, 8], dtype=int)

    # transform each triple of annotations into a code in base `dim`
    for i in range(m):
        ind = sp.where(mat[i, :] >= 0)

        code = mat[i, ind[0][0]] * (dim ** 2) +\
               mat[i, ind[0][1]] * dim +\
               mat[i, ind[0][2]]

        # o is the index of possible combination of annotators in the loop design
        o = -100
        for j in range(8):
            k = 0
            for l in range(3):
                if index[j, l] == ind[0][l]:
                    k += 1
            if k == 3:
                o = j

        if o >= 0:
            data[code, o] += 1
        else:
            print str(code) + " " + str(ind) + " = homeless code"

    return data


#------------------------------------------------------------
#============================================================
#     general-purpose parameter sampling routine for MCMC

def q_lower_upper(theta, psi, lower, upper):
    """
def q_lower_upper(theta,psi,lower,upper):
return (theta,q)

*theta*          -- current parameter value (also, the new value on return)
*lower*, *upper* -- lower/upper boundaries for the parameter
*psi*            -- MCMC max jump size with regard to *theta*
*q*              -- log-ratio of probabilities of sampling 
                       *new value given old value* 
                                  to  
                       *old value given new value*
    """
    q = 0

    if theta < lower or theta > upper:
        print '**g_lower_upper**: Value out of range!!'
        print theta, lower, upper
        raw_input('What now???')
        return theta, q

    # *a* is a uniform random number
    a = random.random()

    # boundary conditions
    if theta == upper:
        theta = upper - a * min(psi, upper - lower)
        q_new_to_old = -log(2 * min(psi, upper - theta))
        q_old_to_new = -log(min(psi, upper - lower))
        q = q_new_to_old - q_old_to_new
        return theta, q

    if theta == lower:
        theta = lower + a * min(psi, upper - lower)
        q_new_to_old = -log(2. * min(psi, theta - lower))
        q_old_to_new = -log(min(psi, upper - lower))
        q = q_new_to_old - q_old_to_new
        return theta, q

    # go to the 'left'
    if  a > 0.5:
        theta_old = theta

        #jump interval is *theta*, choose uniformly
        theta -= random.random() * min(psi, theta_old - lower)

        #transition probability from old to new
        q_old_to_new = -log(min(psi, theta_old - lower))
        q_new_to_old = -log(min(psi, upper - theta))
        q = q_new_to_old - q_old_to_new
        return theta, q

    # go to the 'right'
    else:
        # jump interval is *upper_limit*-*theta*, choose uniformly
        theta_old = theta
        theta += random.random() * min(psi, upper - theta_old)

        q_old_to_new = -log(min(psi, upper - theta_old))
        q_new_to_old = -log(min(psi, theta - lower))
        q = q_new_to_old - q_old_to_new
        return theta, q

    #------------------------------------------------------------


def form_filename(oldf, suffix):
    fi = oldf.split('.')
    fileout = fi[0] + suffix
    return fileout


#------------------------------------------------------------
# compute optimum jump for MCMC estimation of 
# credible intervals
def optimum_jump(likelihood, x0, arguments,
                 x_upper, x_lower,
                 evaluation_jumps, recomputing_cycle, targetreject, Delta,
                 report):
    m = len(x0)
    dx = zeros(m, float)
    Rej = zeros(m, float)

    # initial jump sizes are random
    for i in range(m):
        dx[i] = (x_upper[i] - x_lower[i]) / 100.

    # *x_curr* is the current version of arguments; 
    # this assignemt should produce an array copy 
    x_curr = x0[:]
    logLold = -likelihood(x0, arguments)

    for i in range(evaluation_jumps):
        for j in range(m):
            xj_old = x_curr[j]
            xj, q = q_lower_upper(xj_old, dx[j], x_lower[j], x_upper[j])
            if xj < x_lower[j] or xj > x_upper[j]:
                print xj
                print j
                print xj_old
                raw_input('What now???')

            x_curr[j] = xj
            logLnew = -likelihood(x_curr, arguments)

            alpha = min(1, exp(logLnew - logLold + q))

            if random.random() < alpha:
                logLold = logLnew
            else:
                Rej[j] += float(1. / recomputing_cycle)
                x_curr[j] = xj_old

        if i % recomputing_cycle == 0 and i > 0:
            if cmp(report, 'Nothing') != 0:
                print i
                print Rej
            dx, check = adjustJump(dx, Rej, targetreject, Delta)
            if check == True:
                return dx
            Rej *= 0

    return dx


#----------------------------------------------------------
#==========================================================
def adjustJump(dx, Rej, targetreject, Delta):
    param = len(dx)

    check = True
    for j in range(param):
        if dx[j] == 0:
            dx[j] = 0.000001

        if Rej[j] != 0:
            if abs(Rej[j] - targetreject) > Delta:
                dx[j] *= (targetreject / Rej[j])
                check = False
        elif Rej[j] == 0:
            dx[j] *= 5.

    return dx, check


#=================================================================================================
def sample_distribution(likelihood, x0, arguments, dx, Metropolis_jumps, x_lower
, x_upper, report):
    m = len(x0)
    Samples = zeros([Metropolis_jumps, m], float)
    x_curr = zeros(len(x0), float)
    x_curr[:] = x0[:]
    logLold = -likelihood(x0, arguments)
    Rej = zeros(m, float)

    #print report

    for i in range(Metropolis_jumps):
        if (i + 1) % 100 == 0:
            if cmp(report, 'Nothing') != 0:
                print string_wrap(str(i + 1), 4)

        for j in range(m):
            xj_old = x_curr[j]
            xj, q = q_lower_upper(xj_old, dx[j], x_lower[j], x_upper[j])
            x_curr[j] = xj
            logLnew = -likelihood(x_curr, arguments)

            alpha = min(1, exp(logLnew - logLold + q))

            if random.random() < alpha:
                logLold = logLnew
            else:
                Rej[j] += float(1. / Metropolis_jumps)
                x_curr[j] = xj_old
        Samples[i, :] = x_curr[:]

    return Samples


#------------------------------------------------------------------------------------------
#==========================================================================================
def analyse_parameter_distribution(values, confidence, nbs):
    alpha = (1. - confidence) / 2.
    nn, bins = histogram(values, bins=nbs, range=None, normed=False,
                         weights=None)
    binsize = bins[1] - bins[0]
    binmin = bins[0]
    n = len(nn)
    x = zeros(n, float)
    y = zeros(n, float)
    x[0] = binmin + binsize / 2
    sn = sum(nn)

    vmean = numpy.mean(values, axis=0)
    vmode = scipy.stats.mode(values, axis=0)[0][0]
    vmedian = numpy.median(values, axis=0)
    vskew = scipy.stats.skew(values, axis=0)
    vstd = numpy.std(values, axis=0)


    #  vtable = freqtable(values,axis=0)

    #  print vmean, vmode, vmedian, vskew, 2*vstd

    for i in range(n):
        y[i] = float(nn[i] / (1. * sn))
    for i in range(1, n):
        x[i] = x[i - 1] + binsize

    vs = sort(values)
    nm = len(values)

    n1 = int(nm * alpha)
    n2 = int(nm * (1. - alpha))
    x_left = vs[n1]
    x_right = vs[n2]

    if vmode < x_left or vmode > x_right:
        print 'Mode is outside two-sided CI. Skewed distribution!\n  Switching to one-sided CI...'
        if  vmode < x_left:
            n3 = int(2 * nm * alpha)
            print vmode, '|', x_left, x_right
            x_right = vs[n3]
            print 'theta < ', x_right
            x_left = None
        else:
            n3 = int(nm * (1. - 2 * alpha))
            print vmode, '|', x_left, x_right
            x_left = vs[n3]
            print 'theta > ', x_left
            x_right = None

    print x_left, x_right
    #   print vmode-2*vstd, vmode+2*vstd
    #   print '--------------------------------'

    return x, y, x_left, x_right


#==========================================================================
def get_y(x0, x, y):
    # find x_i, x_i+1 such that x_i < x0 < x_i+1  -->  y0 = y(x_i)/2 + y(x_i+1)/2
    # or x0 == x_i  --> y0 = y(x0)
    n = len(x)
    if x0 < x[0]:
        return y[0]
    elif x0 > x[n - 1]:
        return y[n - 1]

    for i in range(n):
        if x0 == x[i]:
            return y[i]
        elif x0 > x[i] and x0 < x[i + 1]:
            if x0 - x[i] > x[i + 1] - x0:
                return y[i + 1]
            else:
                return y[i]

    print 'get_y:  a problem!!!'


#==========================================================================
def plot_modelA(values, confidence, dpidpi, numbins, x_best):
    fig1 = pyplot.figure(num=None, dpi=dpidpi, facecolor='w', edgecolor='k')
    rect = [0.1, 0.1, 0.8, 0.8]
    ax1 = fig1.add_axes(rect)

    x_left = zeros(len(x_best), float)
    x_right = zeros(len(x_best), float)
    delta = 0
    leg0 = ['$A_1$', '$A_2$', '$A_3$', '$A_4$', '$A_5$', '$A_6$', '$A_7$',
            '$A_8$']
    pyplot.title(
        'Annotator-specific accuracy ($\\theta_i$ $\\rightarrow$ $A_i$)')

    increment_delta = False

    for i in range(8):
        v_tmp = values[:, i]
        x, y, xl, xr = analyse_parameter_distribution(v_tmp, confidence,
                                                      numbins)
        x_left[i] = xl
        x_right[i] = xr
        col = cm.jet(i / 8.)
        step(x, y + delta, linewidth=3, color=col, linestyle='-', where='mid',
             label=leg0[i])
        ax1.plot([xl, xl], [0 + delta, get_y(xl, x, y) + delta], [xr, xr],
            [0 + delta, get_y(xr, x, y) + delta], color=col, linestyle='-',
                         label='_nolegend_')
        ax1.plot([x_best[i], x_best[i]],
            [0, get_y(x_best[i], x, y) + delta], linewidth=1, color=col,
                                       linestyle='--', label='_nolegend_')
        if increment_delta == True:
            delta += max(y)

    xlabel('$\\theta_i$', {'fontsize': 'large'})
    ylabel('$\it{Frequency}$', {'fontsize': 'large'})
    #pyplot.legend()

    leg = ('$\\alpha_1=\\alpha_2=\\alpha_3$', '$\\alpha_4$',
           '$\\alpha_5=\\alpha_6=\\alpha_7$')
    leg1 = ax1.legend(fancybox=True, loc='best')
    leg1.draw_frame(False)

    if len(x_best) > 8:
        fig2 = pyplot.figure(num=None, dpi=dpidpi, facecolor='w', edgecolor='k')
        ax = fig2.add_axes(rect)

        hold(True)
        delta = 0
        xlabel('$\\alpha_i$', {'fontsize': 'large'})
        ylabel('$\it{Frequency}$', {'fontsize': 'large'})
        title('$\\alpha$-parameters: posterior distributions')

        for i in range(8, 11):
            col = cm.jet((i - 8) / 3.)

            v_tmp = values[:, i]
            x, y, xl, xr = analyse_parameter_distribution(v_tmp, confidence,
                                                          numbins)
            x_left[i] = xl
            x_right[i] = xr
            ax.step(x, y + delta, linewidth=3, color=col, linestyle='-',
                    where='mid', label=leg[i - 8])
            ax.plot([xl, xl], [0 + delta, get_y(xl, x, y) + delta], [xr, xr],
                [0 + delta, get_y(xr, x, y) + delta], color=col, linestyle='-',
                            label='_nolegend_')
            ax.plot([x_best[i], x_best[i]],
                [0, get_y(x_best[i], x, y) + delta], linewidth=1, color=col,
                                          linestyle='--', label='_nolegend_')
            if increment_delta == True:
                delta += max(y)

        leg2 = ax.legend(fancybox=True)
        leg2.draw_frame(False)
        #pyplot.legend(loc='best')

    return x_left, x_right


#--------------------------------------------------------------------------
def put_triplet(dicti, like0, x0, mode):
    dicti[mode + '_f_best'] = like0
    dicti[mode + '_x_best'] = x0
    t_best = strftime("%Y-%m-%d %H:%M:%S")
    dicti[mode + '_update_time'] = t_best
    return 0


#-------------------------------------------------------------------------
def get_triplet(dicti, mode):
    f0 = dicti.get(mode + '_f_best', -inf)
    x0 = dicti.get(mode + '_x_best', None)
    t0 = dicti.get(mode + '_update_time', None)
    return f0, x0, t0


#==========================================================================
def load_save_parameters(filename, modelname, f_best, x_best, report):
    filebest = form_filename(filename, '.history')
    print 'load-save'
    curr = modelname + '_' + str(len(x_best))

    if not os.path.isfile(filebest + '.db'):
        print 'new file'
        f = shelve.open(filebest + '.db')
        for model in ['A_model_8', 'A_model_11', 'Bt_model_12']:
            if model == curr:
                put_triplet(f, f_best, x_best, model)
            else:
                put_triplet(f, -inf, None, model)
        f.close()
        return f_best, x_best, strftime("%Y-%m-%d %H:%M:%S")
    else:
        print 'old file'
        f = shelve.open(filebest + '.db')
        f0, x0, t0 = get_triplet(f, curr)

        print 'Stored values:'
        print f0
        print x0
        print t0

        if f0 > f_best:
            f.close()
            return f0, x0, t0
        else:
            put_triplet(f, f_best, x_best, curr)
            f.close()
            return f_best, x_best, strftime("%Y-%m-%d %H:%M:%S")


#------------------------------------------------------------------------------
#==============================================================================
def save_metadata(filename, originaldata, annotators, codes, n, omegas, report):
    filebest = form_filename(filename, '.history')
    print 'Saving metadata ...'

    if not os.path.isfile(filebest + '.db'):
        print 'Your database file <%s> does not exist!!!' % (filebest + '.db')

    f = shelve.open(filebest)
    f['originaldata'] = originaldata
    f['annotators'] = annotators
    f['codes'] = codes
    f['omegas'] = omegas
    f['n'] = n
    f.close()

    return 0


#===========================
class ABmodelGUI(HasTraits):
    model = Enum('A', 'Bt')
    use_priors = Bool(True)
    estimate_alphas = Bool(True)
    number_of_runs = Range(1, 10000, 1)
    run_computation_now = Button('Run')
    use_omegas = Bool(True)
    input_data = File(None)
    optional_labels = File(None)
    estimate_variances = Bool(True)
    Metropolis_jumps = Range(100, 500000, 1000)
    report = Enum('Essentials', 'Everything', 'Nothing')
    target_reject_rate = Range(1, 99, 30)
    evaluation_jumps = Range(200, 5000, 500)
    recomputing_cycle = Range(50, 1000, 100)
    per_cent_delta = Range(0, 30, 20)
    significance = Enum('95%', '99%', '90%')
    figure_dpi = Range(100, 1200, 100)
    # look_at_raw_data = Button('Show raw data')
    raw_annotations = Array(dtype=int32, shape=(None, None))


    def _run_computation_now_fired(self):
        self.run_estimation(self.model,
                            self.use_priors,
                            self.estimate_alphas,
                            self.number_of_runs,
                            self.use_omegas,
                            self.input_data,
                            self.optional_labels,
                            self.estimate_variances,
                            self.Metropolis_jumps,
                            self.report,
                            self.target_reject_rate,
                            self.evaluation_jumps,
                            self.recomputing_cycle,
                            self.per_cent_delta,
                            self.significance,
                            self.figure_dpi,
                            #self.look_at_raw_data,
                            self.raw_annotations)
        return 1


    def _look_at_raw_data_fired(self):
        data_view = View(
            Group(Item(name='raw_annotations', label='Units of annotation'),
                  show_border=True),
            title='Annotators',
            buttons=[CancelButton],
            x=100, y=100, dock='vertical',
            width=700,
            resizable=True
        )
        mat, dim, values, imap, originaldata, annotators, codes, n = check_data(
            self.input_data, self.optional_labels, self.report)
        self.raw_annotations = array(originaldata[0:80, :])

        self.configure_traits(view=data_view)
        print "What now???"


    #============================================================================
    def run_estimation(self, model,
                       use_priors,
                       estimate_alphas,
                       number_of_runs,
                       use_omegas,
                       input_data,
                       optional_labels,
                       estimate_variances,
                       Metropolis_jumps,
                       report,
                       target_reject_rate,
                       evaluation_jumps,
                       recomputing_cycle,
                       per_cent_delta,
                       significance,
                       figure_dpi,
                       #look_at_raw_data,
                       raw_annotations):
        #------------------ getting parameters and deciding what to do... ----------------------
        if cmp(report, 'Nothing') != 0:
            print ''
            print string_wrap('Estimating ...', 3)
            print 'Model: ' + string_wrap(model, 4)
            print 'Use priors: ' + string_wrap(str(use_priors), 4)
            print 'Estimate alphas: ' + string_wrap(str(estimate_alphas), 4)
            print 'Number of runs: ' + string_wrap(str(number_of_runs), 4)
            print 'Use omegas: ' + string_wrap(str(use_omegas), 4)
            print 'Input file: ' + string_wrap(input_data, 4)
            print 'Optional file with labels for figures '\
            + string_wrap(optional_labels, 4)
            print 'Estimate variances: ' + string_wrap(str(estimate_variances),
                                                       4)
            print 'Metropolis jumps: ' + string_wrap(str(Metropolis_jumps), 4)
            print 'Report: ' + string_wrap(str(report), 4)
            print 'Target rejection rate (%): ' + string_wrap(
                str(target_reject_rate), 4)
            print 'Evaluation jumps: ' + string_wrap(str(evaluation_jumps), 4)
            print 'Recomputing cycle: ' + string_wrap(str(recomputing_cycle), 4)
            print 'Per cent of admissible deviation from target rejection: '\
            + string_wrap(str(per_cent_delta), 4)
            print 'Significance: ' + string_wrap(str(significance), 4)
            print 'Figure resolution (dpi): ' + string_wrap(str(figure_dpi), 4)
            print''

        Delta = per_cent_delta * 0.01
        targetreject = target_reject_rate * 0.01
        if cmp(significance, '90%') == 0:
            Level = 0.9
        elif cmp(significance, '95%') == 0:
            Level = 0.95
        else:
            Level = 0.99

        dpi = int(figure_dpi)

        if cmp(model, 'A') == 0:
            modelnumber = 1
            if cmp(report, 'Nothing') != 0:
                print string_wrap("*****A-model*****", 4)
            modelname = 'A_model'
        else:
            if cmp(report, 'Nothing') != 0:
                print string_wrap("*****Bt-model*****", 4)
            modelname = 'Bt_model'
            modelnumber = 2

        tic = time.time()

        estimatealphas = int(estimate_alphas)
        numberOfRuns = int(number_of_runs)
        useomegas = int(use_omegas)
        usepriors = int(use_priors)
        filename = input_data

        random.seed()

        #------ prepare data ---------------------------------------------------------------------------------
        mat, dim, values, imap, originaldata, annotators, codes, n = check_data(
            filename, optional_labels, report)
        data = compute_counts(mat, dim)
        FF = zeros(numberOfRuns, float)
        alphas = []
        gammas = []
        omegas = estimateOmegas8(data, dim, report)


        #---------
        # model A
        if modelnumber == 1:
            alphas = getAlphas(omegas)
            if estimatealphas == True:
                Res = zeros([numberOfRuns, 11], float)
            else:
                Res = zeros([numberOfRuns, 8], float)
            #---------
            # model Bt
        else:
            Res = zeros([numberOfRuns, dim - 1 + 8], float)
        best_f = -Inf

        fileout = form_filename(filename, modelname + '_' + str(
            numberOfRuns) + '_posteriors.txt')
        ffile = open(fileout, "w")

        if cmp(report, 'Nothing') != 0:
            toc = time.time()
            print_wrap(str(toc - tic) + ' seconds has elapsed')


        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        for j in range(numberOfRuns):
            if cmp(report, 'Nothing') != 0:
                print_wrap(str(j + 1))

            # optimize parameters by maximizing the log likelihood of the model
            if modelnumber == 1:
                # model A
                arguments = ((alphas, omegas, data, 1, estimatealphas, dim),)
                arguments1 = (alphas, omegas, data, 1, estimatealphas, dim)
                x0 = random_startA8(estimatealphas, report)
                x_best = scipy.optimize.fmin(likeA8,
                                             x0, args=arguments,
                                             xtol=1e-8, ftol=1e-8, disp=False,
                                             maxiter=1e+10, maxfun=1e+30)
                FF[j] = -likeA8(x_best, arguments1)
            else:
                # model B
                arguments = ((data, dim, 1),)
                arguments1 = (data, dim, 1)
                x0 = random_startBt8(dim, useomegas, data, report)
                x_best = scipy.optimize.fmin(likeBt8,\
                                             x0, args=arguments,\
                                             xtol=1e-8, ftol=1e-8,\
                                             disp=False, maxiter=1e+10,
                                             maxfun=1e+30)
                FF[j] = -likeBt8(x_best, arguments1)

            Res[j, :] = x_best[:]
            toc = time.time()
            if cmp(report, 'Nothing') != 0:
                print_wrap(str((toc - tic) / 60) + ' minutes has elapsed')
                print '     '
                print_wrap(' Log-likelihood = ' + str(FF[j]))
            ffile.write(' Log-likelihood = ' + str(FF[j]) + '\n')

            #------------------------------------
            fs, xs, ts = load_save_parameters(filename, modelname, FF[j], x_best
                                              , report)
            #------------------------------------

            if modelnumber == 1:
                ffile.write(
                    ' Thetas: ' + array_format(Res[j, 0:8], '%4.3f ') + '\n')
                if estimatealphas == 1:
                    ffile.write(' Alphas: ' + array_format(Res[j, 8:11],
                                                           '%4.3f ') + '\n')

                if cmp(report, 'Nothing') != 0:
                    print_wrap(
                        ' Thetas: ' + array_format(Res[j, 0:8], '%4.3f '))
                    if estimatealphas == 1:
                        print_wrap(
                            ' Alphas: ' + array_format(Res[j, 8:11], '%4.3f '))
            else:
                ffile.write(' Gammas: ' + array_format(Res[j, 0:dim - 1],
                                                       '%4.3f ') + '\n')
                ffile.write(
                    ' Thetas: ' + array_format(Res[j, dim - 1:dim - 2 + 9],
                                               '%4.3f ') + '\n')
                if cmp(report, 'Nothing') != 0:
                    print_wrap(
                        ' Gammas: ' + array_format(Res[j, 0:dim - 1], '%4.3f '))
                    print_wrap(
                        ' Thetas: ' + array_format(Res[j, dim - 1:dim - 2 + 9],
                                                   '%4.3f '))

            maxL = -Inf
            thetas = zeros(8, float)
            #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


        #--------------------------------------------------------------------------
        # --------------- save mea-data -------------------------------------------
        save_metadata(filename, originaldata, annotators, codes, n, omegas,
                      report)
        #--------------------------------------------------------------------------
        #-------------------------
        # define best estimates
        x_best = zeros(len(Res[0, :]), float)
        for i in range(numberOfRuns):
            if maxL < FF[i]:
                maxL = FF[i]
                x0 = Res[i, :]
                x_best[:] = x0[:]
        f_best = maxL
        if cmp(report, 'Nothing') != 0:
            print 'Best:'
            print x0

        # try to read the past best estimates for these data
        # save the best results 
        fs, xs, ts = load_save_parameters(filename, modelname, f_best, x_best,
                                          report)
        x0[:] = xs[:]
        x_best[:] = xs[:]

        if cmp(report, 'Nothing') != 0:
            print ''
            print_wrap('Best Log-likelihood so far: ' + str(fs))
            print_wrap('Best parameters so far:')
            print_wrap(array_format(xs, '%4.3f '))

        #----------- now we do have point estimates, but still need to compute
        #--------------- distributions -- with MCMC
        #-------------------------
        # estimate variances and credible intervals of parameters

        if estimate_variances == True:
            if cmp(report, 'Nothing') != 0:
                print string_wrap('**Preparing to compute credible intervals**',
                                  4)
            x_upper = zeros(len(x0), float) + 1.
            x_lower = zeros(len(x0), float)

            if modelnumber == 1:
                likelihood = likeA8
                arguments = (alphas, omegas, data, 1, estimatealphas, dim)
            else:
                likelihood = likeBt8
                arguments = (data, dim, 1)

            if cmp(report, 'Nothing') != 0:
                print string_wrap(
                    'Optimizing jump (to get closer to the target rejection)...'
                    , 4)
            dx = optimum_jump(likelihood, x0, arguments,
                              x_upper, x_lower,
                              evaluation_jumps, recomputing_cycle, targetreject,
                              Delta, report)

            if cmp(report, 'Nothing') != 0:
                print string_wrap('**Computing credible intervals**', 4)

            Samples = sample_distribution(likelihood, x0, arguments,
                                          dx, Metropolis_jumps, x_lower, x_upper
                                          , report)
            print "Save samples!!!"
            fi = filename.split('.')
            filearray = fi[0] + modelname + '_' + str(
                numberOfRuns) + '_MCMC.txt'
            sp.save(filearray, Samples)
            print "Saved samples ..."

            if cmp(report, 'Nothing') != 0:
                print Samples

            numbins = 25
            dpidpi = 150
            if modelnumber == 1:
                x_left, x_right = plot_modelA(Samples, Level, dpidpi, numbins,
                                              x0)
                for i in range(len(x_best)):
                    print str(x_best[i]), ', CI : [', str(x_left[i]), ',', str(
                        x_right[i]), ']'
                print ' '
                show()

            # -------- now we can return to annotations and provide them with posterior probabilities:
            # ---------------------------------------------------------------------------------------------
            # ---------------------------------------------------------------------------------------------
            # compute MAP annotations under model A
        maxL = -Inf
        if modelnumber == 1:
            for i in range(numberOfRuns):
                if estimatealphas == True:
                    if cmp(report, 'Nothing') != 0:
                        print "Run #" + str(i + 1) + ": "\
                              + str(FF[i]) + "|" + str(Res[i, 0:8])\
                              + "|" + str(Res[i, 8:])
                else:
                    if cmp(report, 'Nothing') != 0:
                        print "Run #" + str(i + 1)\
                              + ": "\
                              + str(FF[i])\
                              + "|" + str(Res[i, :])

                if maxL < FF[i]:
                    maxL = FF[i]
                    thetas = Res[i, 0:8]

                if estimatealphas == True:
                    alphas[0] = Res[i, 8]
                    alphas[1] = Res[i, 8]
                    alphas[2] = Res[i, 8]
                    alphas[3] = Res[i, 9]
                    alphas[4] = Res[i, 10]
                    alphas[5] = Res[i, 10]
                    alphas[6] = Res[i, 10]
                #--------------------------
                # posteriors for model Bt
        else:
            gammas = zeros(dim, float)
            for i in range(numberOfRuns):
                if cmp(report, 'Nothing') != 0:
                    print "Run #" + str(i + 1) + ": " + str(FF[i]) + "|" + str(
                        Res[i, :])

                if maxL < FF[i]:
                    maxL = FF[i]
                    thetas = Res[i, dim - 1:8 + dim - 1]
                    gammas[0:dim - 1] = Res[i, 0:dim - 1]
                    gammas[dim - 1] = 1 - sum(gammas[0:dim - 1])
                    alphas = []

                # compute posteriors
        post = format_posteriors(mat, alphas, dim, thetas, gammas, modelnumber)
        post = array(post, float)

        for i in range(post.shape[0]):
            ffile.write(
                array_format(originaldata[i, :], ' %d,') + '|' + array_format(
                    post[i, :], '%5.4f,') + '\n')
            if cmp(report, 'Everything') == 0:
                ind = array(where(originaldata[i, :] > 0), int).flatten()
                s1 = array_format(originaldata[i, :], '%d ')
                s2 = array_format(post[i, :], '%4.3f ')
                s3 = array_format(thetas[ind], '%5.4f ')
                print string_wrap(s1, 1) + "|" + string_wrap(s2,
                                                             2) + "|" + string_wrap(
                    s3, 3)

        if cmp(report, 'Everything') == 0:
            print string_wrap(
                '*Data* | *Posteriors* for all annotation values [1, 2, ...N_max] | *thetas* for the three evaluators'
                , 1)

        ffile.close()


#==========================================================
#              Main
#----------------------------------------------------------
if __name__ == '__main__':
    import sys, os


    main_view = View(Group(Item(name='input_data'),
                           Item(name='optional_labels',
                                label='Optional file with labels for codes'),
                           Item(name='model', style='custom'),
                           Item(name='number_of_runs'),
                           Item(name='use_priors',
                                tooltip='Use informative prior distribution on accuracy parameters')
                           ,
                           Item(name='_'),
                           Item(name='estimate_alphas',
                                label='A: estimate *alphas*',
                                enabled_when="model=='A'",
                                tooltip='Estimate *alphas* with the maximum likelihood method')
                           ,
                           Item(name='use_omegas',
                                label='Bt: use *omegas*',
                                enabled_when="model=='Bt'",
                                tooltip='Use *omegas* to initialize *gammas*'),
                           Item(name='_'),
                           Item(name='estimate_variances',
                                enabled_when="input_data!=None"),
                           Item(name='Metropolis_jumps',
                                label='Number of Metropolis-Hastings jumps',
                                enabled_when="estimate_variances==True"),
                           Item(name='_'),
                           Item(name='report', label='Output verbosity level '),
                           Item(name='_'),
                           Item(name='target_reject_rate', style='custom',
                                format_str='%4.3f',
                                enabled_when="input_data!=None and estimate_variances==True")
                           ,
                           Item(name='evaluation_jumps',
                                enabled_when="input_data!=None and estimate_variances==True")
                           ,
                           Item(name='recomputing_cycle',
                                enabled_when="input_data!=None and estimate_variances==True")
                           ,
                           Item(name='per_cent_delta',
                                label='Maximum deviation from the target rejection rate (%)'
                                ,
                                enabled_when="input_data!=None and estimate_variances==True")
                           ,
                           Item(name='significance',
                                enabled_when="input_data!=None and estimate_variances==True")
                           ,
                           Item(name='figure_dpi',
                                label='Resolution of figures (dpi)',
                                enabled_when="input_data!=None and estimate_variances==True")
                           ,
                           Item(name='_'),
                           Item(name='run_computation_now',
                                enabled_when="input_data!=None",
                                show_label=False),
                           #Item(name='look_at_raw_data',enabled_when="input_data!=None",show_label=False),
                           show_border=True),
                     title='Options:',
                     buttons=[CancelButton],
                     x=100, y=100, dock='vertical',
                     width=700,
                     resizable=True
    )

    data_view = View(Group(Item(name='raw_annotations'),
                           show_border=True),
                     title='Raw annotations:',
                     buttons=[CancelButton],
                     x=100, y=100, dock='vertical',
                     width=700,
                     resizable=True
    )
    gui = ABmodelGUI()
    gui.configure_traits(view=main_view)

    if cmp(gui.report, 'Nothing') != 0:
        print '''    

    *Reference:*

    Rzhetsky A, Shatkay H, Wilbur WJ (2009) 
            How to Get the Most out of Your Curation Effort. 
            PLoS Comput Biol 5(5): e1000391. 
            doi:10.1371/journal.pcbi.1000391
            '''
