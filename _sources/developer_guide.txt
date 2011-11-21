Developer guide
===============

git repository
--------------

pyAnno is hosted on GitHub at
`https://github.com/enthought/uchicago-pyanno <https://github.com/enthought/uchicago-pyanno>`_.

The recommended workflow to extend and improve pyAnno is the following:

1. Fork the project on GitHub

2. Implement your changes on your repository

3. Submit a pull request on GitHub


Running tests
-------------

Even though the tests are written using `unittest`, we recommend the use of
`nosetests` to execute the library test suites. From the root directory of the
pyanno project, type ::

    $ nosetests -v pyanno/test

**Important** The tests that verify some of the models' functionality (e.g.,
the estimation of the model parameters), are stochastic. This has the advantage
that they test different, general scenarios at each round, but occasionally
lead to test failures. If one of the test fails, please run it a second time.
If it continues failing, please report a bug using the
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


Compiling the documentation
---------------------------

The documentation is based on `Sphinx <http://sphinx.pocoo.org/index.html>`_.
It can be found in `pyanno\docs`.

After editing the documentation, you should compile it, revise the result,
and publish it online:

1) In `pyanno\docs` type ::

    $ make html

   To compile the pyAnno documentation.

2) Check out the `gh-pages` branch from the git repository, and copy the
   newly built HTML docs

3) push the branch back to GitHub

