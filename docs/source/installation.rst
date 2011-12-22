Installation guide
==================

pyAnno can be installed on any platform that runs Python, including
all recent flavors of Windows, Mac, and Linux.

.. contents:: Index

Install dependencies
--------------------

To use pyAnno you will need the following:

   - Python 2.7
     http://www.python.org/

   - numpy 1.6
     http://numpy.scipy.org/

   - scipy 0.9.0
     http://www.scipy.org/

   - traits 4.1
     http://code.enthought.com/projects/traits/

   - chaco 4.1.0
     http://code.enthought.com/chaco/

   - wxPython 2.8.10
     http://www.wxpython.org/

The easiest way to satisfy all of these dependencies is to install the
`EPD Free Python distribution <http://enthought.com/products/epd_free.php>`_.
For your convenience, we prepared installers for Mac OS X and Windows
that include both pyAnno and the EPD Free distribution:

[ADD LINK TO INSTALLERS]

.. note::

    The 64-bit EPD Free distribution for Mac OS X does not include
    wxPython (because of `lack of support from Apple
    <http://enthought.com/products/epdfaq.php#mac>`_). Please install the
    fully-featured 32-bit version.


.. _binary_installers:

Binary installers
-----------------

The most convenient way to install pyAnno is to use our custom
EPD Free installers for Mac OS X and Windows, which include the latest
version of Python and all scientific libraries necessary to run pyAnno:

[ADD LINK TO INSTALLERS]


If you already have a Python installation with all the dependencies listed
above, you will find Windows and Mac OS X binary installers at

[ADD LINK TO INSTALLERS]

To executable scripts to start the pyAnno GUI will be installed in the
scripts path of your Python installation. On Windows, this will usually be at
:file:`C:\\Python27\\Scripts\\pyanno-ui` , and on Mac OS X at
:file:`/Library/Frameworks/Python.framework/Versions/Current/bin/pyanno-ui`.

You can also start the pyAnno GUI from a terminal with the command ::

   pyanno-ui


Installing with `pip` or `easy_install`
---------------------------------------

pyAnno is hosted on PyPi_, so it can be installed simply with either
pip_ (recommended) or easy_install_ . Both tools are very easy to install,
and are often available out of the box on many Python distributions.

To install pyAnno, simply type on the command line:

::

   pip install pyanno

or

::

   easy_install pyanno

If you do not have administrator permissions in the Python folder,
you should pre-pend the commands above with `sudo` (e.g.,
`sudo pip install pyanno`).

The pyAnno GUI can then be started from the terminal by typing ::

   pyanno-ui


Installing from source
----------------------

pyAnno's source code repository is hosted on GitHub at
https://github.com/enthought/uchicago-pyanno . To install the library
from the latest source:

1. Clone pyAnno's git repository ::

    git clone git://github.com/enthought/uchicago-pyanno.git

2. Enter the repository ::

    cd uchicago-pyanno

3. Install pyAnno ::

    python setup.py install


.. _PyPi: http://pypi.python.org/pypi
.. _pip: http://www.pip-installer.org/en/latest/index.html
.. _easy_install: http://peak.telecommunity.com/DevCenter/EasyInstall.html
