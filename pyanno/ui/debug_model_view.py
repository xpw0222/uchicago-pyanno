""" Base class for Model/View's to provide a python shell for debugging. """

# Copyright 2011 Mark Dickinson and Martin Chilvers, Enthought

from enthought.traits.api import PythonValue
from enthought.traits.ui.api import Group, Include, Item, ModelView
from enthought.traits.ui.api import ShellEditor, VGroup, View


class DebugModelView(ModelView):
    """ Base class for Model/View's to provide a python shell for debugging.
    """

    #### Private protocol #####################################################

    # A dummy python value to allow us to display a Python shell.
    _python_value = PythonValue

    #### Traits UI ############################################################

    python_shell_group = Group(
        Item(
            '_python_value',
            editor     = ShellEditor(share=True),
            show_label = False,
        ),

        label = 'Python Shell'
    )

    debug_group = python_shell_group

    debug_view  = View(
        VGroup(
            Include('body'),
            Include('debug_group'),

            layout = 'split'
        ),

        resizable = True,
        width     = 800,
        height    = 600
    )
