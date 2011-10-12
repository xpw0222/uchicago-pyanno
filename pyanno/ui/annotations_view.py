from traits.has_traits import HasTraits, HasStrictTraits
from traits.trait_numeric import Array
from traits.trait_types import Instance, Int, List, ListFloat
from traits.traits import Property
from traitsui.api import ModelView, View, VGroup
from traitsui.item import Item
from pyanno.ui.hinton_plot import HintonDiagramPlot
from pyanno.util import annotations_frequency


class AnnotationsView(HasStrictTraits):
    """ Traits UI Model/View for annotations."""

    ### Model-related traits ###
    annotations = Array
    nclasses = Int

    frequency = Property(ListFloat, depends_on='annotations,nclasses')

    def _get_frequency(self):
        return annotations_frequency(self.annotations, self.nclasses).tolist()

    ### Traits UI definitions ###
    frequency_plot = Instance(HintonDiagramPlot)

    def _frequency_plot_default(self):
        return HintonDiagramPlot(data = self.frequency)

    ### View definition ###
    body = VGroup(
        Item('frequency_plot',
             style='custom',
             resizable=False,
             show_label=False
        )
    )

    traits_view = View(body)



#### Testing and debugging ####################################################

def main():
    """ Entry point for standalone testing/debugging. """

    from pyanno.modelBt import ModelBt
    model = ModelBt.create_initial_state(5)
    annotations = model.generate_annotations(model.generate_labels(100))
    model_view = AnnotationsView(annotations=annotations, nclasses=5)
    model_view.configure_traits()
    return model, annotations, model_view


if __name__ == '__main__':
    m, a, mv = main()
