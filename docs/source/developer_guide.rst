Developer guide
===============

.. contents:: Index


git repository
--------------

pyAnno is hosted on GitHub at
`https://github.com/enthought/uchicago-pyanno <https://github.com/enthought/uchicago-pyanno>`_.

The recommended workflow to extend and improve pyAnno is the following:

1. Fork the project on GitHub

2. Implement your changes on your repository

3. Submit a pull request on GitHub

.. _testing:

Running tests
-------------

Even though the tests are written using `unittest`, we recommend the use of
`nosetests` to execute the library test suites. From the root directory of the
pyanno project, type ::

    $ nosetests -v pyanno/test

.. note::

    The tests that verify some of the models' functionality (e.g.,
    the estimation of the model parameters), are stochastic. This has the advantage
    that they test different, general scenarios at each round, but occasionally
    lead to test failures. If one of the test fails, please run it a second time.
    If the failures are consistent, please report a bug using the
    `issue tracker <https://github.com/enthought/uchicago-pyanno/issues>`_.


Adding a new model
------------------

This is a checklist of things to do when implementing a new model in pyAnno:

1) Write a new model implementation as a subclass of
   :class:`pyanno.abstract_model.AbstractModel`, implementing the full
   interface.

2) Add an import statement in :py:mod:`pyanno.models`

3) Add some documentation about the class in the `models` page.


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


Publishing the documentation
----------------------------

The documentation is based on `Sphinx <http://sphinx.pocoo.org/index.html>`_.
It can be found in `pyanno\docs`.

After editing the documentation, you should compile it, revise the result,
and publish it online:

1) Check out the `gh-pages` branch from the git repository in a new
   directory, `DOCPATH`.

2) Enter the directory `pyanno\docs` and
   edit the `BUILDDIR` variable in the local `Makfile` to `DOCPATH`.

3) Type ::

    $ make html

   Make sure the the `pyanno` package
   is in the `PYTHONPATH`, or Shpinx will fail to generate the API
   documentation.

4) Enter `DOCPATH`, check that the documentation has been correctly generated,
   and push the branch back to GitHub: ::

    $ git add *
    $ git commit -m"DOC: describe your changes here"
    $ git push origin gh-pages

The new documentation will be published within seconds.


Uploading a new version of pyAnno to PyPI
-----------------------------------------

1) Revise the `setup.py` file and update the version number

2) :ref:`Run the tests <testing>` and correct any bug

3) Push to PyPI: ::

    $ python setup.py register bdist_egg bdist_wininst sdist upload

