User guide
==========

This user guide contains information about getting started
with pyAnno, running simulations and running model estimation.


General organization of pyAnno
------------------------------

pyAnno can be used as a Python library from within your Python code, or as
a standalone application by executing the pyAnno GUI.

The pyAnno library is composed of several modules:

* :py:mod:`pyanno.models` defines a number of models for the
  annotation  process

* :py:mod:`pyanno.annotations` offers a :class:`~AnnotationsContainer` object
  to load annotation files

* :py:mod:`pyanno.measures` contains a number of statistical functions to
  measure aggreement and covariance between annotators

* :py:mod:`pyanno.plots` defines functions to give a graphical representations
  of annotations and model parameters

* :py:mod:`pyanno.database` defines a
  :class:`~PyannoDatabase` object that can be used to store and retrieve
  the result of applying models to annotations

* :py:mod:`pyanno.ui` contains the definition of the pyAnno GUI,
  which is used by the pyAnno application and is
  typically not used directly from Python


Using the library from a Python shell
-------------------------------------

Creating a new model
^^^^^^^^^^^^^^^^^^^^

Generating data
^^^^^^^^^^^^^^^

Estimating parameters
^^^^^^^^^^^^^^^^^^^^^

Plots
^^^^^


Using the library from the pyanno GUI
-------------------------------------

Starting the GUI
^^^^^^^^^^^^^^^^

Navigating the main window
^^^^^^^^^^^^^^^^^^^^^^^^^^

The database window
^^^^^^^^^^^^^^^^^^^




Finding help
------------

* Online API: [REF]

* Another way to see how the functions are intended to work
  is to have a look at the unit tests, which can be found in
  the directory `pyanno\tests` of the pyanno library

* If everything else fails, please describe your issue in
  pyAnno's
  `issue tracker <https://github.com/enthought/uchicago-pyanno/issues>`_.
