from tables.array import Array
from traits.has_traits import HasTraits, on_trait_change
from traits.trait_types import Enum, Button, Instance, Int
from traitsui.group import VGroup, HGroup
from traitsui.item import Item
from traitsui.view import View

import pyanno.measures as measures


# statistical measures for pairs of annotators
PAIRWISE_STATS = {
    "Scott's Pi": measures.scotts_pi,
    "Cohen's Kappa": measures.cohens_kappa,
    "Cohen's weightd Kappa": measures.cohens_weighted_kappa,
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

class AnnotationStatisticsView(HasTraits):

    statistics_name = Enum(ALL_STATS_NAMES)

    annotations = Array

    nclasses = Int

    info_button = Button("Info...")

    stats_view = Instance(View)


    @on_trait_change('statistics_names')
    def _update_stats_view(self):
        if self.statistics_name in GLOBAL_STATS:
            stat_func = GLOBAL_STATS[self.statistics_name]
            res = stat_func(self.annotations, nclasses=self.nclasses)

            self.stats_view = self._create_single_value_view(res)

        else:
            stat_func = PAIRWISE_STATS[self.statistics_name]
            res = measures.pairwise_matrix(stat_func, self.annotations,
                                           nclasses=self.nclasses)

            self.stats_view = self._create_matrix_view(res)


    def _create_single_value_view(self, val):
        return View()


    def _create_matrix_view(self, mat):
        return View()


    traits_view = View(
        VGroup(
            HGroup(
                Item("statistics_name", show_label=False),
                Item("info_button", show_label=False, enabled_when="False")
            ),
            Item("stats_view", style="custom", show_label=False)
        )
    )


#### Testing and debugging ####################################################

def main():
    """ Entry point for standalone testing/debugging. """

    from pyanno import ModelBt
    model = ModelBt.create_initial_state(5)
    annotations = model.generate_annotations(model.generate_labels(2))

    stats_view = AnnotationStatisticsView(annotations=annotations)
    stats_view.configure_traits()
    return model, annotations, stats_view


if __name__ == '__main__':
    m, a, sv = main()
