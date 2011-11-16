"""pyanno package setup definition"""


from setuptools import setup, find_packages

# ---- add ETS recipes for py2app

import types
import py2app.recipes

def ets_check(cmd, mf):
    m = mf.findNode('pyface')
    if m is None or m.filename is None:
        return None
    return dict(
        packages = ['pyface','enable','kiva','traits','wx','traitsui','chaco']
    )

py2app.recipes.ets = types.ModuleType('py2app.recipes.ets')
py2app.recipes.ets.check = ets_check


# ---- /add ETS recipes for py2app



APP = ['pyanno/ui/main.py']
OPTIONS = {'argv_emulation': True, 'packages': 'pyanno'}


with open('README') as f:
    LONG_DESCRIPTION = f.read()

setup(name = "pyanno",
      version = "2.0dev-10",
      packages = find_packages(),

      author = 'pyAnno developers',

      description = 'Package for curating data annotation efforts.',
      long_description = LONG_DESCRIPTION,

      url = 'https://github.com/enthought/uchicago-pyanno',
      download_url = 'https://github.com/enthought/uchicago-pyanno',

      license='LICENSE.txt',
      platforms = ["Any"],


      package_data = {
          '': ['*.txt', '*.rst', 'data/*'],
          'pyanno.ui': ['images/*'],
          'pyanno.plots': ['images/*']
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

      app=APP,
      options={'py2app': OPTIONS},
      setup_requires=['py2app'],

      #scripts = ['examples/mle_sim.py',
      #           'examples/map_sim.py',
      #           'examples/rzhetsky_2009/mle.py',
      #           'examples/rzhetsky_2009/map.py' ],
  )
