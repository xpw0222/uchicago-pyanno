"""This file contains functions to sample from a distribution given its
log likelihood.
"""

import numpy as np


# TODO need to change sign convention for likelihood: positive, not negative
from pyanno.util import string_wrap


def q_lower_upper(theta, psi, lower, upper):
    """General-purpose parameter sampling routine for MCMC

    Input:
    theta          -- current parameter value (also, the new value on return)
    lower, upper -- lower/upper boundaries for the parameter
    psi            -- MCMC max jump size with regard to *theta*

     Output:
    theta
    q              -- log-ratio of probabilities of sampling
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
    a = np.random.random()

    # boundary conditions
    if theta == upper:
        theta = upper - a * min(psi, upper - lower)
        q_new_to_old = -np.log(2 * min(psi, upper - theta))
        q_old_to_new = -np.log(min(psi, upper - lower))
        q = q_new_to_old - q_old_to_new
        return theta, q

    if theta == lower:
        theta = lower + a * min(psi, upper - lower)
        q_new_to_old = -np.log(2. * min(psi, theta - lower))
        q_old_to_new = -np.log(min(psi, upper - lower))
        q = q_new_to_old - q_old_to_new
        return theta, q

    # go to the 'left'
    if  a > 0.5:
        theta_old = theta

        #jump interval is *theta*, choose uniformly
        theta -= np.random.random() * min(psi, theta_old - lower)

        #transition probability from old to new
        q_old_to_new = -np.log(min(psi, theta_old - lower))
        q_new_to_old = -np.log(min(psi, upper - theta))
        q = q_new_to_old - q_old_to_new
        return theta, q

    # go to the 'right'
    else:
        # jump interval is *upper_limit*-*theta*, choose uniformly
        theta_old = theta
        theta += np.random.random() * min(psi, upper - theta_old)

        q_old_to_new = -np.log(min(psi, upper - theta_old))
        q_new_to_old = -np.log(min(psi, theta - lower))
        q = q_new_to_old - q_old_to_new
        return theta, q

    #------------------------------------------------------------


def optimum_jump(likelihood, x0, arguments,
                 x_upper, x_lower,
                 evaluation_jumps, recomputing_cycle, targetreject, Delta,
                 report):
    """Compute optimum jump for MCMC estimation of credible intervals.

    Adjust jump size in Metropolis-Hasting MC to achieve given rejection rate.
    Jump size is estimated for each argument separately.

    Input:
    likelihood -- likelihood function (??? describe arguments)

    Output:
    dx --
    """
    m = len(x0)
    dx = np.zeros(m, float)
    Rej = np.zeros(m, float)

    # initial jump sizes are random
    for i in range(m):
        dx[i] = (x_upper[i] - x_lower[i]) / 100.

    # *x_curr* is the current version of arguments;
    # this assignment should produce an array copy
    x_curr = x0[:]
    logLold = -likelihood(x0, arguments)

    for i in range(evaluation_jumps):
        for j in range(m):
            xj_old = x_curr[j]
            xj, q = q_lower_upper(xj_old, dx[j], x_lower[j], x_upper[j])
            if xj < x_lower[j] or xj > x_upper[j]:
                # FIXME: take care of this case
                print xj
                print j
                print xj_old
                raw_input('What now???')

            x_curr[j] = xj
            logLnew = -likelihood(x_curr, arguments)

            alpha = min(1, np.exp(logLnew - logLold + q))

            # rejection step
            if np.random.random() < alpha:
                logLold = logLnew
            else:
                Rej[j] += float(1. / recomputing_cycle)
                x_curr[j] = xj_old

        if i % recomputing_cycle == 0 and i > 0:
            if cmp(report, 'Nothing') != 0:
                print i
                print Rej
                print dx
            dx, check = adjustJump(dx, Rej, targetreject, Delta)
            if check == True:
                print i
                print Rej
                print dx
                return dx
            Rej *= 0

    return dx


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
            check = False

    return dx, check


def sample_distribution(likelihood, x0, arguments, dx, Metropolis_jumps, x_lower
, x_upper, report):
    m = len(x0)
    Samples = np.zeros([Metropolis_jumps, m], float)
    x_curr = np.zeros(len(x0), float)
    x_curr[:] = x0[:]
    logLold = -likelihood(x0, arguments)
    Rej = np.zeros(m, float)

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

            alpha = min(1, np.exp(logLnew - logLold + q))

            if np.random.random() < alpha:
                logLold = logLnew
            else:
                Rej[j] += float(1. / Metropolis_jumps)
                x_curr[j] = xj_old
        Samples[i, :] = x_curr[:]

    return Samples