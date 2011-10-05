from chaco.data_label import DataLabel
from chaco.data_range_2d import DataRange2D
from chaco.text_box_overlay import TextBoxOverlay
from chaco.tooltip import ToolTip
from traits.trait_numeric import Array
from traits.trait_types import Button, Range, Int
from traitsui.api import ModelView, View, Item, Group, Tabbed, VGroup, HGroup
from traitsui.editors.array_editor import ArrayEditor
from traitsui.editors.instance_editor import InstanceEditor
from traitsui.item import Heading
from traitsui.menu import OKButton, OKCancelButtons
from traits.api import Instance
from enable.component_editor import ComponentEditor
from chaco.api import ArrayPlotData, Plot
from chaco.base import n_gon
import numpy as np

from pyanno.ui.arrayview import create_array2d_editor

def _hinton_diagram(distr, title=None):

    plot_data = ArrayPlotData()
    # Polygon Plot holding the square of the Hinton diagram
    polyplot = Plot(plot_data)

    distr_len = distr.shape[0]
    centers = [(i, 0.5) for i in xrange(1, distr_len+1)]

    for idx, center in enumerate(centers):
        # draw square with area proportional to probability mass
        r = np.sqrt(distr[idx] / 2.)
        npoints = n_gon(center=center, r=r, nsides=4, rot_degrees=45)
        nxarray, nyarray = np.transpose(npoints)
        # save in dataarray
        plot_data.set_data("x" + str(idx), nxarray)
        plot_data.set_data("y" + str(idx), nyarray)

        plot = polyplot.plot(("x"+str(idx), "y"+str(idx)),
                             type="polygon",
                             face_color='black',
                             edge_color='black')[0]

    # tweak some of the plot properties
    polyplot.padding = 50
    if title is not None:
        polyplot.title = title
    range2d = DataRange2D(low=(0.5, -0.2), high=(distr_len+0.5, 1.2))
    polyplot.range2d = range2d
    polyplot.aspect_ratio = ((range2d.x_range.high - range2d.x_range.low)
                             / (range2d.y_range.high - range2d.y_range.low))

    polyplot.x_grid.visible = False
    polyplot.y_grid.visible = False
    polyplot.x_axis.tick_interval = 1.
    polyplot.y_axis.tick_interval = 1.
    #polyplot.y_axis.tick_visible = False
    polyplot.border_visible = False
    #polyplot.x_axis.visible = False
    polyplot.y_axis.visible = False

    #label = TextBoxOverlay(text='test', border_color='white')
    #polyplot.overlays.append(label)

    return polyplot


def create_gamma_view(gamma):
    width = len(gamma)
    class GammaView(ModelView):
        data = Array(dtype=float, shape=(1, width))
        def _data_default(self):
            return self.model.gamma[np.newaxis,:]
        traits_view = View(
            Group(
                Item('data',
                     editor=create_array2d_editor(width, show_index=False),
                     show_label=False),
                ),
            title     = 'Model B-with-Theta, gamma parameters',
            width     = 500,
            height    = 200,
            resizable = True,
            buttons   = OKCancelButtons
       )
    return GammaView


class ModelBtView(ModelView):
    """ Traits UI Model/View for 'ModelBt' objects.
    """

    #### Model properties #######
    pass

    #### Traits UI view #########
    gamma_button = Button(label='Edit...')
    gamma_plot = Instance(Plot)

    # TODO check that gamma is a distribution, and that bounds are btw 0 and 1
    body = VGroup(
        Item('model.nclasses',
             label='number of labels',
             style='readonly'),
        Item('model.nannotators',
             label='number of annotators',
             style='readonly'),
        HGroup(
            Item(label='Gamma[k] = P(label=k)'),
            Item('gamma_plot',
                 editor=ComponentEditor(width=300, height=150),
                 resizable=False,
                 show_label=False
            ),
            Item('gamma_button', show_label=False),
            show_border=True
        ),
        Item(
            'model.theta',
            editor=InstanceEditor(editable=True)
        ),
        Item('model.theta', label="Theta[j] = P(annotation_j=k | label=k)")
    )

    traits_view = View(body, buttons=[OKButton], resizable=True)

    def _gamma_plot_default(self):
        return _hinton_diagram(self.model.gamma)

    def _gamma_button_fired(self):
        gamma_view = create_gamma_view(self.model.gamma)
        gamma_view = gamma_view(model=self.model)
        gamma_view.edit_traits(kind='livemodal')


#### Testing and debugging ####################################################

def main():
    """ Entry point for standalone testing/debugging. """

    from pyanno.modelBt import ModelBt

    model = ModelBt.create_initial_state(5)
    model_view = ModelBtView(model=model)
    model_view.configure_traits(view='traits_view')


if __name__ == '__main__':
    main()
