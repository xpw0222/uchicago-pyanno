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


Installing with `easy_install`
------------------------------

::

   easy_install pyanno


Mac OS X and Windows installers
-------------------------------

Where to find the installers, and how to use them.


Installing from source
----------------------

1. Create Directory $PYANNO_DIR for pyanno


2. Download pyanno-1.0.zip into $PYANNO_DIR


3. Unpack pyanno-1.0.zip into $PYANNO_DIR

   You can do this from the Explorer in Windows by:
       a. navigating to $PYANNO_DIR, 
       b. right clicking on pyanno-1.0.zip,
       c. selecting [extract all]
       d. entering $PYANNO_DIR path or browsing for it
       e. click [Extract] button

   Or from a shell with:
       % cp pyanno-1.0.zip $PYANNO_DIR
       % cd $PYANNO_DIR
       % unzip pyanno-1.0.zip

   We like the Cygwin distribution to get tools like
   unzip in Windows: http://www.cygwin.com/


4. Install pyanno into Python:
   This you'll need to do from a shell:

       % cd pyanno-1.0
       % python setup.py install
