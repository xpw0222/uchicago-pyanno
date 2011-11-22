Model definitions
=================

At present, pyAnno implements three probabilistic models of data annotation:

    1. `Model A`_ , a three-step generative model from the paper
    [Rzhetsky2009]_.

    2. `Model B-with-theta`_ , a multinomial generative model from the paper
    [Rzhetsky2009]_.

    3. `Model B`_, a Bayesian generalization of the model proposed in
    [Dawid1979]_.

Definitions
-----------

annotations
    The values emitted by the annotator on the available data items.
    In the documentation,
    :math:`x_i^j`, indicates the i-th annotation for annotator j.

labels
    The possible annotations. They may be numbers, or strings, or any
    discrete set of objects.

label class, or just class
    Every set of labels is ordered and numbered from 0 to K. The number
    associated with each label is the label class. The ground truth label class
    for each data item, i, is indicated in the documentation as :math:`y_i`.

prevalence
    The prior probability of label classes

accuracy
    The probability of an annotator reporting the correct label class


Model A
-------

descr

Model B-with-theta
------------------

Model B-with-theta is a multinomial generative model of the annotation
process. The process begins with the generation of "true" label classes,
drawn from a fixed categorical distribution. Each annotator reports
a label class with some additional noise.

There are two sets of parameters: :math:`\gamma_k` controls the
prior probability of generating a label of class :math:`k`.
The accuracy parameter :math:`\\theta^j_k` controls the probability of annotator
:math:`j` reporting class :math:`k'` given that the true label is :math:`k`.
An important part of the model is that the error probability is controlled
by just one parameter per annotator, making estimation more robust and
efficient.

Formally, for annotations :math:`x_i^j` and true label classes :math:`y_i`:

* The probability of the true label classes is

  :math:`P(\mathbf{y} | \gamma) = \prod_i P(y_i | \gamma)`,

  :math:`P(y_i | \gamma) = \mathrm{Categorical}(y_i; \gamma) = \gamma_{y_i}`.

* The prior over the accuracy parameters is

  :math:`P(\theta_j) = \mathrm{Beta}(\theta_j; 1, 2)`.

* And finally the distribution over the annotations is

  :math:`P(\mathbf{x} | \mathbf{y}, \theta) = \prod_i \prod_j P(x^j_i | y_i, \theta_j)`,

  :math:`P(x^j_i | y_i, \theta_j) = \left\{\begin{array}{l l} \theta_j & \quad \text{if } x_i^j = y_i\\ \frac{1-\theta_j}{\sum_n \theta_n} & \quad \text{otherwise}\\ \end{array} \right.`.

See [Rzhetsky2009]_ for more details.

Model B
-------

Model B is a more general form of B-with-theta, and is also a Bayesian
generalization of the earlier model proposed in [Dawid1979]_. The generative
process is identical to the one in model B-with-theta, except that
a) 


References
----------

.. [Rzhetsky2009] Rzhetsky A., Shatkay, H., and Wilbur,
    W.J. (2009). "How to get the most from
    your curation effort", PLoS Computational Biology, 5(5).

.. [Dawid1979] Dawid, A. P. and A. M. Skene. 1979.  Maximum likelihood
    estimation of observer error-rates using the EM algorithm.  Applied
    Statistics, 28(1):20--28.


DAWID AND SKENE'S MULTINOMIAL MODEL W. ARBITRARY DESIGN
------------------------------------------------------------

Our first model generalizes the model in (Dawid and Skene 1979) to
arbitrary designs, including that of (Rzhetsky et al. 2009).  There
are no priors, so estimation is necessarily through maximum
likelihood.


Example math:
:math:`P(x_i = y^t)`


Data
--------------------
I                 number of items being annotated
J                 number of annotators
N                 number of annotations
K                 number of categories
jj[n] in 0:(J-1)  annotator for annotation n in 1:N
ii[n] in 0:(I-1)  item for annotation n in 1:N
y[n]  in 0:(K-1)  category for annotation n in 1:N

Parameters
--------------------
z[i]  in 1:K                 (latent) category for item i in 1:I
pi[k] in [0,1]               prevalence of category k in 1:K; 
                                 SUM_k pi[k] = 1
theta[j][kRef][k] in [0,1]   probability of annotator j in 1:J 
                                 returning category k in 1:K for 
                                 item of true category kRef; 
                                 SUM_k theta[j,kRef,k] = 1
Model
--------------------
z[i] ~ Categorical(pi)
y[n] ~ Categorical(theta[ jj[n] ][ z[ii[n]] ])

Complete Data Likelihood
--------------------
p(y,z|theta,pi)
    = p(z|pi) * p(y|theta,z)
    = PROD_{i in 1:I} p(z[i]|pi)
      * PROD_{n in 1:N} p(y[n]|theta,z)
    = PROD_{i in 1:K} pi[ z[i] ]
      * PROD_{n in 1:N} theta[ jj[n] ][ z[ii[n]] ][ y[n] ]

Observed Data Likelihood
--------------------
p(y|theta,pi) = INTEGRAL_z p(y,z|theta,pi) dz

Maximum Likelihood Estimate (MLE)
--------------------
(theta*,pi*) = ARGMAX_{theta,pi} p(y|theta,pi)



DAWID AND SKENE'S MODEL WITH PRIORS
------------------------------------------------------------
The second model adds priors to the Dawid and Skene model, which
corresponds to the full Model B in (Rzhetsky et al. 2009).

Priors
--------------------
This model basically adds Dirichlet priors for the categorical
parameters.  There is one prior beta for prevalence pi, and
K priors alpha[k] for annotator response for items of reference
category k.  

   beta in (0,infty)^K

   alpha[k] in (0,infty)^K

For maximum a posteriori fitting, all values must be 
greater than or equal to 1.0.


Model
--------------------
In BUGS-like notation, we add the following:

pi ~ Dirichlet(beta)

for (j in 1:J) {
    for (k in 1:K) {
    	theta[j][k] ~ Dirichlet(alpha[k])
    }
}

Complete Likelihood
--------------------
We just add in terms for the priors to the data likelihood
above, giving us:

p(y,z,theta,pi|alpha,beta)
    = p(theta|alpha) * p(pi|beta) * p(y,z|theta,pi)

where

     p(theta|alpha) = Dirichlet(theta|alpha)

and

     p(theta|alpha) 
         = PROD_{j in 1:J} PROD_{k in 1:K} 
	     Dirichlet(theta[j][k]|alpha[k]).

EM ALGORITHM
------------------------------------------------------------

All of the expecation-maximization (EM) algorithms work the
same way for computing either maximum likelihood estimates (MLE)
or maximum a posterioiri (MAP estimates).  The basic idea is
to treat the the unknown category labels as missing data,
alternating between estimating the category expecations and
then maximizing the parameters for those expectations.

0. Initialize parameters (pi(0),theta(0))

1. for n = 1; ; ++n

   1.a  (E Step)
        Calculate observed data likelihood given previous params
             p(cat|pi(n-1),theta(n-1),y)

   1.b  (M Step)
        Set next params pi(n), theta(n) to maximize observed 
        data likelihood w.r.t. previous params
         
   1.c  (convergence test)
        if log likelihood doesn't change much, exit



