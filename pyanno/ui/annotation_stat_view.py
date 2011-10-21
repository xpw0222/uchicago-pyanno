from chaco.array_plot_data import ArrayPlotData
from chaco.color_bar import ColorBar
from chaco.data_range_1d import DataRange1D
from chaco.default_colormaps import jet
from chaco.linear_mapper import LinearMapper
from chaco.plot import Plot
from chaco.plot_containers import HPlotContainer
from chaco.tools.save_tool import SaveTool
from enable.component_editor import ComponentEditor
from traits.has_traits import HasTraits, on_trait_change, HasStrictTraits
from traits.trait_numeric import Array
from traits.trait_types import Enum, Button, Instance, Int, Str, Float
from traitsui.group import VGroup, HGroup
from traitsui.item import Item
from traitsui.view import View

import numpy as np
import pyanno.measures as measures


# statistical measures for pairs of annotators
from pyanno.ui.plot_tools import SaveToolPlus, CopyDataToClipboardTool


PAIRWISE_STATS = {
    "Scott's Pi": measures.scotts_pi,
    "Cohen's Kappa": measures.cohens_kappa,
    "Cohen's weighted Kappa": measures.cohens_weighted_kappa,
    "Pearson's Rho": measures.pearsons_rho,
    "Spearman's Rho": measures.spearmans_rho
}

# statistical measures for all annotators
GLOBAL_STATS = {
    "Fleiss' Kappa": measures.fleiss_kappa,
    "Krippendorff's Kappa": measures.krippendorffs_alpha,
    "Cronbach's Alpha": measures.cronbachs_alpha
}

ALL_STATS_NAMES = PAIRWISE_STATS.keys() + GLOBAL_STATS.keys()


class _SingleStatView(HasTraits):
    value = Float
    name = Str
    txt = Str

    @on_trait_change('value')
    def _update_txt(self):
        txt = "{name} = {val:.4}".format(
            name = self.name,
            val = self.value
        )
        self.txt = txt

    traits_view = View(
        Item('txt', style='readonly', show_label=False)
    )

class _MatrixStatView(HasStrictTraits):
    matrix = Array
    name = Str
    matrix_plot = Instance(HPlotContainer)

    def _matrix_plot_default(self):
        matrix = np.nan_to_num(self.matrix)
        width = matrix.shape[0]

        # Create a plot data obect and give it this data
        plot_data = ArrayPlotData()
        plot_data.set_data("values", matrix)

        # Create the plot
        plot = Plot(plot_data)
        colormap = jet(DataRange1D(low=-1.0, high=1.0))
        img_plot = plot.img_plot("values",
                                 interpolation='nearest',
                                 xbounds=(0, width),
                                 ybounds=(0, width),
                                 colormap=colormap)[0]

        plot.title = self.name

        plot.aspect_ratio = 1.
        # padding [left, right, up, down]
        plot.padding = [0, 0, 25, 25]

        # Create the colorbar, handing in the appropriate range and colormap
        colormap = img_plot.color_mapper
        colorbar = ColorBar(index_mapper=LinearMapper(range=colormap.range),
                            color_mapper=colormap,
                            plot=img_plot,
                            orientation='v',
                            resizable='v',
                            width=20,
                            padding=[0,20,0,0])
        colorbar.padding_top = plot.padding_top
        colorbar.padding_bottom = plot.padding_bottom

        # Create a container to position the plot and the colorbar side-by-side
        container = HPlotContainer(use_backbuffer = True)
        container.add(plot)
        container.add(colorbar)
        container.bgcolor = "lightgray"

        save_tool = SaveToolPlus(component=container,
                                filename='/Users/pberkes/del/test.png')
        plot.tools.append(save_tool)
        copy_tool = CopyDataToClipboardTool(component=container, data=matrix)
        plot.tools.append(copy_tool)

        return container

    traits_view = View(
        Item('matrix_plot',
             editor=ComponentEditor(),
             resizable=False,
             show_label=False,
             height=-200,
             width=-200
        ),
    )


class AnnotationsStatisticsView(HasTraits):

    statistics_name = Enum(ALL_STATS_NAMES)

    annotations = Array

    nclasses = Int

    info_button = Button("Info...")

    stats_view = Instance(HasTraits)


    @on_trait_change('statistics_name')
    def _update_stats_view(self):
        if self.statistics_name in GLOBAL_STATS:
            stat_func = GLOBAL_STATS[self.statistics_name]
            res = stat_func(self.annotations, nclasses=self.nclasses)

            self.stats_view = _SingleStatView(value=res,
                                              name=self.statistics_name)

        else:
            stat_func = PAIRWISE_STATS[self.statistics_name]
            res = measures.pairwise_matrix(stat_func, self.annotations,
                                           nclasses=self.nclasses)

            self.stats_view = _MatrixStatView(matrix=res,
                                              name=self.statistics_name)


    def traits_view(self):
        self.statistics_name = "Cohen's Kappa"
        traits_view = View(
            VGroup(
                HGroup(
                    Item("statistics_name", show_label=False),
                    Item("info_button", show_label=False, enabled_when="False")
                ),
                Item("stats_view",
                     style="custom",
                     show_label=False),
            ),
            resizable = True
        )
        return traits_view


#### Testing and debugging ####################################################

def main():
    """ Entry point for standalone testing/debugging. """

    from pyanno import ModelBt
    model = ModelBt.create_initial_state(5)
    annotations = model.generate_annotations(model.generate_labels(20))

    stats_view = AnnotationsStatisticsView(annotations=annotations, nclasses=5)
    stats_view.configure_traits()
    return model, annotations, stats_view


if __name__ == '__main__':
    m, a, sv = main()
