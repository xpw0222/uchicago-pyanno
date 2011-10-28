.. pyanno documentation master file, created by
   sphinx-quickstart on Tue Oct 25 13:48:31 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyAnno's documentation!
==================================

pyAnno 2.0 is a Python library for the analysis and diagnostic testing of
annotation and curation efforts. pyAnno implements statistical models for
inferring from categorical data annotated by multiple annotators

    * annotator accuracies and biases,
    * gold standard categories of items,
    * prevalence of categories in population, and
    * population distribution of annotator accuracies and biases.

The models include a generalization of Dawid and Skene's (1979) multinomial
model with Dirichlet priors on prevalence and estimator accuracy,
and the two models introduces in Rzhetsky et al.'s (2009). The implementation
allows Maximum Likelihood and Maximum A Posteriori estimation of parameters,
and to draw samples from the full posterior distribution over annotator
accuracy.


Contents:

.. toctree::
   :maxdepth: 2

   Installation guide <installation>
   User guide <user_guide>
   pyAnno models <models>
   Developer guide <developer_guide>
   Library Reference <modules>


Licensing
---------

pyAnno is licensed under the Apache License, Version 2.0.


Contributors
------------

* Pietro Berkes    (Enthought, Ltd.)
* Bob Carpenter    (Columbia University,   Statistics)
* Andrey Rzhetsky  (University of Chicago, Medicine)
* James Evans      (University of Chicago, Sociology)


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

