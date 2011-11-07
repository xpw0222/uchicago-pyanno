Installation guide
==================

**[under construction]**

pyAnno can be installed on any platform that runs Python, including
all recent flavors of Windows, Mac, and Linux.

Install Dependencies
--------------------

To use pyAnno you will need to have the following Python libraries installed:

   - Python 2.7
     http://www.python.org/

   - numpy 1.6
     http://numpy.scipy.org/

   - scipy 0.9.0
     http://www.scipy.org/

   - traits 4.0.1
     http://code.enthought.com/projects/traits/

   - chaco 4.1.0
     http://code.enthought.com/chaco/

The easiest way to install all of these dependencies is to install the EPD
Free Python distribution:
http://enthought.com/products/epd_free.php


Installing with `easy_install` or `pip`
---------------------------------------

pyAnno is hosted on PyPi_, so it can be installed simply with either
pip_ (preferred) or easy_install_ . Both tools are very easy to install,
and are already available on most Python distributions.

To install pyAnno, simply type in your command line:

::

   pip install pyanno

or

::

   easy_install pyanno

If you do not have administrative permissions in the Python folder,
you should pre-pend the commands above with `sudo` (e.g.,
`sudo pip install pyanno`).


.. _PyPi: http://pypi.python.org/pypi
.. _pip: http://www.pip-installer.org/en/latest/index.html
.. _easy_install: http://peak.telecommunity.com/DevCenter/EasyInstall.html


Mac OS X and Windows installers
-------------------------------

[Where to find the installers, and how to use them.]


Installing from source
----------------------

1. Clone pyAnno's git repository_ ::

    git clone git://github.com/enthought/uchicago-pyanno.git

2. Enter the repository ::

    cd uchicago-pyanno

3. Install pyAnno ::

    python setup.py install

.. _repository: https://github.com/enthought/uchicago-pyanno
