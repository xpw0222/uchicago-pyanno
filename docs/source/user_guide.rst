User guide
==========

This user guide contains information about getting started
with pyAnno, running simulations and running model estimation.

pyAnno can be used as a Python library from within your Python code, or as
a standalone application by executing the pyAnno GUI.

Using the library from a Python shell
-------------------------------------

General organization of pyAnno
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


Annotations
^^^^^^^^^^^

In pyAnno, annotations are two-dimensional arrays of integers. Rows
correspond to data items, and columns to annotators. Each entry :math:`x_i^j`
in an annotation array is the label class assigned by annotator :math:`j` to
item :math:`i`, or :attr:`pyanno.util.MISSING_VALUE` for missing values.

[how to build annotation arrays from raw data]

Creating a new model
^^^^^^^^^^^^^^^^^^^^

Annotations: missing values have to be indicated as
:attr:`pyanno.util.MISSING_VALUE` (which evaluates to -1).

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

Plot tools
''''''''''

* To save a plot as displayed in the window, click on the plot and press Ctrl-S
  (Cmd-S on Mac). A dialog will open, asking for a destination file
  and the resolution of the saved image (in dpi).

* It is possible to copy the *data* underlying the plots by pressing Ctrl-C
  (Cmd-C on Mac). The data is copied on the clipboard as a Python string,
  which can be copied in a text file, or in a Python shell to further analyze
  it. For most plots, the copied data will be a numpy array. Make sure to
  type `from numpy import array` in your Python shell so that Python can create
  an array object when you paste the string.


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
