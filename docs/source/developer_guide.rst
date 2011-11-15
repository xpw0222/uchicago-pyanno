Developer guide
===============

**Under construction**

git repository
--------------

pyAnno is hosted on github at https://github.com/enthought/uchicago-pyanno .

*describe workflow: fork, pull request*


Running tests
-------------

We recommend the use of `nosetests`. From the root directory of the
pyanno project, type ::

    $ nosetests -v pyanno/test


Compiling the documentation
---------------------------

The documentation is based on `Sphinx <http://sphinx.pocoo.org/index.html>`_.
It can be found in `pyanno\docs`.

After editing the documentation, you whould compile it, revise the result,
and publish it online:

1) In `pyanno\docs` type ::

    $ make html

[under construction]

Adding a new model
------------------

This is a checklist when implementing a new model in pyAnno:

1) Write a new model implementation as a subclass of
   :class:`pyanno.abstract_model.AbstractModel`, implementing the full
   interface.

2) Add an import statement in :py:mod:`pyanno.models`

To add the model to the UI:

1) Write a new subclass of :class:`pyanno.ui.model_view.PyannoModelView`,
   the `traitsui` graphical view of your model parameters, and of
   :class:`pyanno.ui.model_view.NewModelDialog`, which creates a dialog
   requesting the parameters to create a new instance of the model.
   See the classes defined in :py:mod:`pyanno.ui.model_a_view` for
   reference.

2) Add the name of the model, and a reference to the class and model view
   in `model_name`, `_model_name_to_class`, and `_model_class_to_view`,
   at the beginning of the :class:`pyanno.ui.model_data_view.ModelDataView`.


BUILD AND INSTALL
------------------------------------------------------------
You only need to build and install if you want to run
the contained scripts, because they import the packages
assuming they are installed.  Unit testing, for instance,
doesn't require a build or install.

Building pyanno
----------------------------------------
To build a gzipped tarball (suffix .tar.gz)
and zipped archive (suffix .zip), use:

    % cd $PYANNO_HOME
    % python setup.py sdist --formats=gztar,zip

The result will be to create two distribution
files:

    $PYANNO_HOME/dist/pyanno-1.0.tar.gz
                     /pyanno-1.0.zip

Developer Installation of pyanno
----------------------------------------
After following the build instructions above,

    % cd $PYANNO_HOME
    % python setup.py install

End-user installation instructions in Install.txt.


Scripted Operation
----------------------------------------
There are Windows batch (.bat) and Unix shell (.sh) scripts
to build and install.  

On Windows:

     % cd $PYANNO_HOME
     % build.bat

On Unix:

     % cd $PYANNO_HOME
     % sh build.sh



