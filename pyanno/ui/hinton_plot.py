from chaco.data_range_2d import DataRange2D
from chaco.label_axis import LabelAxis
from chaco.scales.scales import FixedScale
from chaco.scales_tick_generator import ScalesTickGenerator
from enable.component_editor import ComponentEditor
from traits.has_traits import HasTraits, on_trait_change
from traits.trait_types import Str, ListFloat
from traits.api import Instance
from chaco.api import ArrayPlotData, Plot
from chaco.base import n_gon
import numpy as np
from traitsui.group import VGroup
from traitsui.item import Item
from traitsui.view import View


class HintonDiagramPlot(HasTraits):
    """Defines a Hinton diagram view of a discrete probability distribution.

    A Hinton diagram displays a probability distribution as a series of
    squares, with area proportional to the probability mass of each element.
    """

    data = ListFloat

    plot_data = Instance(ArrayPlotData)
    plot = Instance(Plot)

    title = Str


    @on_trait_change('data', post_init=True)
    def _update_data(self):
        distr_len = len(self.data)
        plot_data = self.plot_data

        # centers of the squares
        centers = [(i, 0.5) for i in xrange(1, distr_len + 1)]

        for idx, center in enumerate(centers):
            # draw square with area proportional to probability mass
            r = np.sqrt(self.data[idx] / 2.)
            npoints = n_gon(center=center, r=r, nsides=4, rot_degrees=45)
            nxarray, nyarray = np.transpose(npoints)
            # save in dataarray
            plot_data.set_data('x%d' % idx, nxarray)
            plot_data.set_data('y%d' % idx, nyarray)


    def _plot_data_default(self):
        self.plot_data = ArrayPlotData()
        self._update_data()
        return self.plot_data


    def _create_probability_axis(self, plot):
        """Create plot axis for probability values."""
        prob_axis = LabelAxis(
            plot,
            orientation='left',
            positions=[0.5, 0.5 + np.sqrt(0.25) / 2., 1.0],
            labels=['0', '0.25', '1']
        )
        prob_axis.tick_generator = ScalesTickGenerator(scale=FixedScale(0.001))
        return prob_axis


    def _create_label_axis(self, len_, plot):
        """Create plot axis for labels at fixed intervals."""
        ids = range(1, len_ + 1)
        label_list = [str(id) for id in ids]
        label_axis = LabelAxis(
            plot,
            orientation='bottom',
            positions=ids,
            labels=label_list,
            label_rotation=0,
            small_haxis_style=True
        )
        # use a FixedScale tick generator with a resolution of 1
        label_axis.tick_generator = ScalesTickGenerator(scale=FixedScale(1.))
        return label_axis


    def _plot_default(self):
        distr_len = len(self.data)

        # PolygonPlot holding the square of the Hinton diagram
        polyplot = Plot(self.plot_data)
        for idx in range(distr_len):
            polyplot.plot(('x%d' % idx, 'y%d' % idx),
                          type="polygon",
                          face_color='black',
                          edge_color='black')

        # remove grids and axes
        polyplot.underlays = []

        # create x axis for labels
        label_axis = self._create_label_axis(distr_len, polyplot)
        polyplot.index_axis = label_axis
        polyplot.underlays.append(label_axis)

        # create y axis for probability density
        #prob_axis = self._create_probability_axis(polyplot)
        #polyplot.value_axis = prob_axis
        #polyplot.underlays.append(prob_axis)

        # tweak some of the plot properties
        #polyplot.padding = 50
        if self.title is not None:
            polyplot.title = self.title
        range2d = DataRange2D(low=(0.5, 0.), high=(distr_len+0.5, 1.))
        polyplot.range2d = range2d
        polyplot.aspect_ratio = ((range2d.x_range.high - range2d.x_range.low)
                                 / (range2d.y_range.high - range2d.y_range.low))

        polyplot.border_visible = False
        # some padding right, on the bottom
        polyplot.padding = [15, 15, 15, 25]

        return polyplot


    #### View definition #####################################################

    body = Item('plot',
             editor=ComponentEditor(),
             resizable=False,
             show_label=False,
             height=-100,
            )

    traits_view = View(body)



#### Testing and debugging ####################################################

def main():
    """ Entry point for standalone testing/debugging. """

    from pyanno.modelBt import ModelBt
    import numpy as np

    model = ModelBt.create_initial_state(5)
    mv = HintonDiagramPlot(data=model.gamma.tolist())
    mv.configure_traits(view='traits_view')

    return model, mv


if __name__ == '__main__':
    model, mv = main()
