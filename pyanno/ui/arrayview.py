import numpy as np

from traits.api import HasTraits, Property, Array
from traitsui.api import View, Item, TabularEditor
from traitsui.tabular_adapter import TabularAdapter
from traitsui.menu import NoButtons


def create_array2d_adapter(width, show_index=True, data_format='%f'):
    _columns = [('%d' % (i+1), i) for i in range(width)]
    if show_index:
        _columns.insert(0, ('row\col', 'index'))

    class Array2DAdapter(TabularAdapter):
        columns = _columns

        font = 'Courier 10'
        alignment = 'right'
        format = data_format
        index_text = Property

        def _get_index_text(self):
            return str(self.row)

    return Array2DAdapter()


def create_array2d_editor(width, show_index=True, data_format='%f'):
    adapter = create_array2d_adapter(width, show_index, data_format)
    return TabularEditor(adapter=adapter)


#### Testing and debugging ####################################################

def main():
    """Entry point for standalone testing/debugging."""

    class TestShowArray(HasTraits):

        data = Array(shape=(None, 9))

        view = View(
            Item(
                'data',
                editor=create_array2d_editor(9,
                                             show_index=True),
                show_label=False
            ),
            title     = 'Array2D editor',
            width     = 0.3,
            height    = 0.8,
            resizable = True,
            buttons   = NoButtons
        )


    data = np.random.rand(100,9)
    blah = TestShowArray(data=data)
    blah.data = data
    print blah.data
    blah.configure_traits()


if __name__ == '__main__':
    main()
