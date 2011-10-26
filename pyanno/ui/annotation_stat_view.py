from traits.has_traits import HasTraits, on_trait_change
from traits.trait_numeric import Array
from traits.trait_types import Enum, Button, Instance, Int, Str, Float
from traitsui.group import VGroup, HGroup
from traitsui.item import Item
from traitsui.view import View

import numpy as np
import pyanno.measures as measures
from pyanno.plots.matrix_plot import MatrixPlot



# statistical measures for pairs of annotators
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

            self.stats_view = MatrixPlot(matrix=res,
                                              colormap_low=-1.0,
                                              colormap_high=1.0,
                                              title=self.statistics_name)


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