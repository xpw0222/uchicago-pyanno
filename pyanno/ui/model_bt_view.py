from chaco.data_label import DataLabel
from chaco.data_range_2d import DataRange2D
from chaco.label_axis import LabelAxis
from chaco.scales.scales import FixedScale
from chaco.scales_tick_generator import ScalesTickGenerator
from chaco.text_box_overlay import TextBoxOverlay
from chaco.tooltip import ToolTip
from traits.has_traits import HasTraits, on_trait_change
from traits.trait_numeric import Array
from traits.trait_types import Button, Range, Int, Str, ListFloat, List, CFloat, CList, Float, File
from traitsui.api import ModelView, View, Item, Group, Tabbed, VGroup, HGroup
from traitsui.editors.array_editor import ArrayEditor
from traitsui.editors.instance_editor import InstanceEditor
from traitsui.editors.tabular_editor import TabularEditor
from traitsui.group import HFlow
from traitsui.item import Heading, Spring
from traitsui.menu import OKButton, OKCancelButtons
from traitsui.file_dialog import open_file
from traits.api import Instance
from enable.component_editor import ComponentEditor
from chaco.api import ArrayPlotData, Plot
from chaco.base import n_gon
import numpy as np

from pyanno.ui.arrayview import Array2DAdapter


class HintonDiagramPlot(HasTraits):

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
        ids = range(1, distr_len+1)
        label_list = [ 'Gamma[{}]'.format(id) for id in ids]

        label_axis = LabelAxis(
            polyplot,
            orientation = 'bottom',
            positions = ids,
            labels = label_list,
            label_rotation = 0,
            small_haxis_style=True
        )
        # use a FixedScale tick generator with a resolution of 1
        label_axis.tick_generator = ScalesTickGenerator(scale=FixedScale(1.))

        polyplot.index_axis = label_axis
        polyplot.underlays.append(label_axis)

        # create y axis for probability density
        prob_axis = LabelAxis(
            polyplot,
            orientation = 'left',
            positions = [0.5, 0.5+np.sqrt(0.25)/2., 1.0],
            labels = ['0', '0.25', '1']
        )
        prob_axis.tick_generator = ScalesTickGenerator(scale=FixedScale(0.001))

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

        return polyplot


class GammaView(HasTraits):
    data = List

    def traits_view(self):
        return View(
            Group(Item('data',
                       editor=TabularEditor
                           (
                           adapter=Array2DAdapter(ncolumns=len(self.data[0]),
                                                  show_index=False)),
                       show_label=False)),
            title     = 'Model B-with-Theta, gamma parameters',
            width     = 500,
            height    = 200,
            resizable = True,
            buttons   = OKCancelButtons
            )


def vcenter(item):
    return VGroup(Spring(),item,Spring())

class ModelBtView(ModelView):
    """ Traits UI Model/View for 'ModelBt' objects.
    """

    #### Model properties #######
    gamma = ListFloat


    def update_from_model(self):
        """Update view parameters to the ones in the model."""
        print 'update from model'
        self.gamma = self.model.gamma.tolist()


    def _gamma_default(self):
        return self.model.gamma.tolist()


    @on_trait_change('gamma[]', post_init=True)
    def _update_gamma(self):
        self.model.gamma = np.asarray(self.gamma)
        self.gamma_hinton.data = self.gamma


    #### Traits UI view #########
    gamma_hinton = Instance(HintonDiagramPlot)
    gamma_plot = Instance(Plot)

    ## Actions ##
    edit_gamma = Button(label='Edit...')
    generate_data = Button(label='Generate annotations...')

    def _edit_gamma_fired(self):
        """Create editor for gamma parameters"""
        gamma_view = GammaView(data=[self.gamma])
        gamma_view.edit_traits(kind='livemodal')


    # TODO check that gamma is a distribution, and that bounds are btw 0 and 1
    body = VGroup(
        Item('model.nclasses',
             label='number of labels',
             style='readonly'),
        Item('model.nannotators',
             label='number of annotators',
             style='readonly'),
        HGroup(
            vcenter(Item(label='Gamma[k] = P(label=k)')),
            Item('handler.gamma_plot',
                 editor=ComponentEditor(),
                 resizable=False,
                 show_label=False,
                 height=-200,
                 width=-500
            ),
            vcenter(Item('handler.edit_gamma', show_label=False)),
            show_border=True
        ),
        Item('model.theta', label="Theta[j] = P(annotation_j=k | label=k)"),
        HFlow(
            Item('handler.ml_estimate', show_label=False)
        )
    )

    traits_view = View(body, buttons=[OKButton], resizable=True)


    def _gamma_plot_default(self):
        self.gamma_hinton = HintonDiagramPlot(data = self.gamma)
        return self.gamma_hinton.plot


#### Testing and debugging ####################################################

def main():
    """ Entry point for standalone testing/debugging. """

    from pyanno.modelBt import ModelBt

    model = ModelBt.create_initial_state(5)
    model_view = ModelBtView(model=model)
    model_view.configure_traits(view='traits_view')

    return model, model_view


if __name__ == '__main__':
    model, model_view = main()
