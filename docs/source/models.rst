Model definitions
=================

At present, pyAnno implements three probabilistic models of data annotation:

    1. `Model A`_ , a three-step generative model from the paper
    [Rzhetsky2009]_.

    2. `Model B-with-theta`_ , a multinomial generative model from the paper
    [Rzhetsky2009]_.

    3. `Model B`_, a Bayesian generalization of the model proposed in
    [Dawid1979]_.

Glossary
--------

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

Model A  defines a probability distribution over data annotations with
a loop design in which each item is annotated by three users.
The distributions over annotations is defined by a
three-steps generative model:

    1. First, the model independently generates correctness values for the
       triplet of annotators (e.g., CCI where C=correct, I=incorrect)

    2. Second, the model generates an agreement pattern compatible with
       the correctness values (e.g., CII is compatible with the agreement
       patterns 'abb' and 'abc', where different letters correspond to
       different annotations

    3. Finally, the model generates actual observations compatible with
       the agreement patterns

More in detail, the model is described as follows:

* Parameters :math:`\theta_j` control the prior probability that annotator
  :math:`j` is correct. Thus, for each triplet of annotations
  for annotators :math:`m`, :math:`n`, and :math:`l`, we have

  :math:`P( \mathrm{X}_m \mathrm{X}_n \mathrm{X}_l | \theta ) = P( \mathrm{X}_m | \theta ) P( \mathrm{X}_n | \theta ) P( \mathrm{X}_l | \theta )`

  where

  :math:`P( \mathrm{X}_j ) =\left\{\begin{array}{l l} \theta_j & \quad \text{if } \mathrm{X}_j = \mathrm{C} \\ 1-\theta_j & \quad \text{if } \mathrm{X}_j = \mathrm{I}\\ \end{array} \right.`

* Parameters :math:`\omega_k` control the probability of observing an
  annotation of class :math:`k` over all items and annotators. From these
  one can derive the parameters :math:`\alpha`, which correspond
  to the probability
  of each agreement pattern according to the tables published in
  [Rzhetsky2009]_.

See [Rzhetsky2009]_ for a more complete presentation of the model.

Model B-with-theta
------------------

Model B-with-theta is a multinomial generative model of the annotation
process. The process begins with the generation of "true" label classes,
drawn from a fixed categorical distribution. Each annotator reports
a label class with some additional noise.

There are two sets of parameters: :math:`\gamma_k` controls the
prior probability of generating a label of class :math:`k`.
The accuracy parameter :math:`\theta^j_k` controls the probability of annotator
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
a) the accuracy parameters are represented by a full tensor
:math:`\theta_{j,k,k'} = P(x^j = k' | y = k)`, and b) it defines prior
probabilities over the model parameters, :math:`\theta`, and :math:`\pi`.

The complete model description is as follows:

* The probability of the true label classes is

  :math:`P(\pi | \beta) = \mathrm{Dirichlet} (\pi ; \beta)`

  :math:`P(\mathbf{y} | \pi) = \prod_i P(y_i | \pi)`,

  :math:`P(y_i | \pi) = \mathrm{Categorical}(y_i; \pi) = \pi_{y_i}`

* The distribution over accuracy parameters is

  :math:`P(\theta_{j,k,:} | \alpha_k) = \mathrm{Dirichlet} ( \theta_{j,k,:} ; \alpha_k)`

  The hyper-parameters :math:`\alpha_k` define what kind of error distributions
  are more likely for an annotator. For example, they can be defined such that
  :math:`\alpha_{k,k'}` peaks at :math:`k = k'` and decays for :math:`k'`
  becoming increasingly dissimilar to :math:`k`. Such a prior is adequate
  for ordinal data, where the label classes have a meaningful order.

* The distribution over annotation is defined as

  :math:`P(\mathbf{x} | \mathbf{y}, \theta) = \prod_i \prod_j P(x^j_i | y_i, \theta_{j,:,:})`,

  :math:`P(x^j_i = k' | y_i = k, \theta_{j,:,:}) = \theta{j,k,k'}`.

References
----------

.. [Rzhetsky2009] Rzhetsky A., Shatkay, H., and Wilbur,
    W.J. (2009). "How to get the most from
    your curation effort", PLoS Computational Biology, 5(5).

.. [Dawid1979] Dawid, A. P. and A. M. Skene. 1979.  Maximum likelihood
    estimation of observer error-rates using the EM algorithm.  Applied
    Statistics, 28(1):20--28.


