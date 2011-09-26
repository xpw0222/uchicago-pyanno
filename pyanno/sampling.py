"""This file contains functions to sample from a distribution given its
log likelihood.
"""

import numpy as np
from pyanno.util import string_wrap


# TODO need to change sign convention for likelihood: positive, not negative
# TODO add parameters for burn-in, thinning
# ??? vectorizing this and optimum_jump gives a 10x speedup
#     (evaluation only after sampling the *full* parameters space)
def sample_distribution(likelihood, x0, arguments, dx,
                        nsamples, x_lower, x_upper, report):
    """General-purpose sampling routine for MCMC sampling.

    Draw samples from a distribution given its unnormalized log likelihood.

    Input:
    likelihood -- unnormalized log likelihood function. the function accepts two
                   arguments: likelihood(params, values); `params` is a vector
                   containing a vector of parameters that will be sampled over
                   by this function; `values` contains the observed values.
                  The function should return the *negative* log likelihood
    """
    m = len(x0)
    samples = np.zeros((nsamples, m), dtype=float)
    x_curr = x0.copy()
    llhood = -likelihood(x0, arguments)
    #rejection = np.zeros((m,), dtype=float)

    for i in range(nsamples):
        if (i + 1) % 100 == 0:
            if cmp(report, 'Nothing') != 0:
                print string_wrap(str(i + 1), 4)

        for j in range(m):
            xj_old = x_curr[j]
            xj, q = q_lower_upper(xj_old, dx[j], x_lower[j], x_upper[j])
            x_curr[j] = xj
            llhood_new = -likelihood(x_curr, arguments)

            # rejection step; reject with probability `alpha`
            alpha = min(1, np.exp(llhood_new - llhood + q))
            if np.random.random() < alpha:
                llhood = llhood_new
            else:
                #rejection[j] += 1. / nsamples
                x_curr[j] = xj_old
        samples[i,:] = x_curr

    return samples


# TODO: simplify, move stopping condition ("check") to optimum_jump
def optimum_jump(likelihood, x0, arguments,
                 x_upper, x_lower,
                 evaluation_jumps, recomputing_cycle, target_rejection_rate,
                 delta, report):
    """Compute optimum jump for MCMC estimation of credible intervals.

    Adjust jump size in Metropolis-Hasting MC to achieve target rejection rate.
    Jump size is estimated for each argument separately.

    Input:
    likelihood -- log likelihood function. the function accepts two
                   arguments: likelihood(params, values); `params` is a vector
                   containing a vector of parameters that will be sampled over
                   by this function; `values` contains the observed values.
                  The function should return the *negative* log likelihood

    Output:
    dx -- the final jump size
    """
    m = len(x0)
    dx = np.zeros((m,), dtype=float)
    rejection = np.zeros(m, float)

    # initial jump sizes are random
    for i in range(m):
        dx[i] = (x_upper[i] - x_lower[i]) / 100.

    # *x_curr* is the current version of arguments
    x_curr = x0.copy()
    llhood = -likelihood(x0, arguments)

    for i in range(evaluation_jumps):
        for j in range(m):
            xj_old = x_curr[j]
            xj, q = q_lower_upper(xj_old, dx[j], x_lower[j], x_upper[j])
            if xj < x_lower[j] or xj > x_upper[j]:
                raise ValueError('Parameter values out or range')

            x_curr[j] = xj
            llhood_new = -likelihood(x_curr, arguments)
            alpha = min(1, np.exp(llhood_new - llhood + q))

            # rejection step; accept with probability `alpha`
            if np.random.random() < alpha:
                llhood = llhood_new
            else:
                rejection[j] += 1. / recomputing_cycle
                x_curr[j] = xj_old

        # adjust step size every `recomputing cycle` steps
        if i % recomputing_cycle == 0 and i > 0:
            if cmp(report, 'Nothing') != 0:
                print i
                print rejection
                print dx
            dx, check = _adjust_jump(dx, rejection, target_rejection_rate, delta)
            if check == True:
                # all rejection rates within range
                print i
                print rejection
                print dx
                return dx
                # reset all rejection rates
            rejection *= 0.

    return dx


def _adjust_jump(dx, rejection, target_reject_rate, delta):
    """Adapt step size to get closer to target rejection rate.

    Output:
    dx -- new step sizes
    check -- True if all rejection rates are within the required limits
    """
    check = True
    for j in xrange(dx.shape[0]):
        if dx[j] == 0:
            dx[j] = 0.000001

        if rejection[j] != 0:
            if abs(rejection[j] - target_reject_rate) > delta:
                dx[j] *= (target_reject_rate / rejection[j])
                check = False
        elif rejection[j] == 0:
            dx[j] *= 5.
            check = False

    return dx, check


# TODO vectorize this to sample from all dimensions at once
def q_lower_upper(theta, psi, lower, upper):
    """Returns a sample from the proposal distribution.

    Input:
    theta -- current parameter value (also, the new value on return)
    psi -- MCMC max jump size with regard to *theta*
    lower, upper -- lower/upper boundaries for the parameter

     Output:
    theta
    q              -- log-ratio of probabilities of sampling
                        *new value given old value*
                                   to
                        *old value given new value*
    """

    if theta < lower or theta > upper:
        raise ValueError('Parameter values out or range')

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
