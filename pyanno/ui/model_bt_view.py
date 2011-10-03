from chaco.data_range_2d import DataRange2D
from traitsui.api import ModelView, View, Item, Group, Tabbed, VGroup, HGroup
from traitsui.menu import OKButton
from traits.api import Instance
from enable.component_editor import ComponentEditor
from chaco.api import ArrayPlotData, Plot
from chaco.base import n_gon
import numpy as np


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

    return polyplot


class ModelBtView(ModelView):
    """ Traits UI Model/View for the basic properties of 'Formation' objects.

    """

    #### Model properties #######
    pass

    #### Traits UI view #########
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
                 show_label=False),
            Item('model.gamma')
        ),
        Item('model.theta', label="Theta[j] = P(annotation_j=k | label=k)")
    )

    traits_view = View(body, buttons=[OKButton], resizable=True)

    def _gamma_plot_default(self):
        return _hinton_diagram(self.model.gamma)

def main():
    """ Entry point for standalone testing/debugging. """

    from pyanno.modelBt import ModelBt

    model      = ModelBt.random_model(5, 100)
    model_view = ModelBtView(model=model)
    model_view.configure_traits(view='traits_view')


if __name__ == '__main__':
    main()
