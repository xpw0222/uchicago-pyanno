"""pyanno package setup definition"""


from setuptools import setup, find_packages


setup(name = "pyanno",
      version = "2.0dev-2",
      packages = find_packages(),

      package_data = {
          '': ['*.txt', '*.rst', 'data/*'],
      },
      include_package_data = True,

      install_requires = [],

      entry_points = {
          'console_scripts': [],
          'gui_scripts': [
              'pyanno-ui = pyanno.ui.main:main',
          ],
          'setuptools.installation': [
            'eggsecutable = pyanno.ui.main:main',
          ]
      },

      #scripts = ['examples/mle_sim.py',
      #           'examples/map_sim.py',
      #           'examples/rzhetsky_2009/mle.py',
      #           'examples/rzhetsky_2009/map.py' ],

      author = ['Pietro Berkes', 'Bob Carpenter',
                'Andrey Rzhetsky', 'James Evans'],
      author_email = ['pberkes@enthought.com', 'carp@lingpipe.com'],

      description = 'Package for curating data annotation efforts.',

      url = 'https://github.com/enthought/uchicago-pyanno',
      download_url = 'https://github.com/enthought/uchicago-pyanno',

      license='LICENSE.txt'
      )
