"""pyanno package setup definition"""


from setuptools import setup, find_packages

# ---- add ETS recipes for py2app

import types
import py2app.recipes

py2app.recipes.pyface = types.ModuleType('py2app.recipes.pyface')
py2app.recipes.traitsui = types.ModuleType('py2app.recipes.traitsui')
py2app.recipes.chaco = types.ModuleType('py2app.recipes.chaco')

def pyface_check(cmd, mf):
    m = mf.findNode('pyface')
    if m is None or m.filename is None:
        return None
    return dict(
        packages = ['pyface','enable','kiva','traits','wx']
    )

def traitsui_check(cmd, mf):
    m = mf.findNode('traitsui')
    if m is None or m.filename is None:
        return None
    return dict(
        packages = ['traitsui']
    )

def chaco_check(cmd, mf):
    m = mf.findNode('chaco')
    if m is None or m.filename is None:
        return None
    return dict(
        packages = ['chaco']
    )

py2app.recipes.pyface.check = pyface_check
py2app.recipes.traitsui.check = traitsui_check
py2app.recipes.chaco.check = chaco_check

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
