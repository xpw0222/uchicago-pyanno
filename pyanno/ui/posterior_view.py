# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

from traits.has_traits import HasTraits
from traits.trait_numeric import Array
from traits.trait_types import Bool, Instance
from traitsui.editors.instance_editor import InstanceEditor
from traitsui.group import VGroup, HGroup
from traitsui.item import Item
from traitsui.view import View
from pyanno.plots.annotations_plot import PosteriorPlot
from pyanno.ui.appbase.wx_utils import is_display_small
from pyanno.util import majority_vote


class PosteriorView(HasTraits):

    show_maximum = Bool(False)
    show_majority_vote = Bool(False)
    posterior_plot = Instance(PosteriorPlot)
    annotations = Array

    def traits_view(self):
        height = 760 if is_display_small() else 900
        traits_view = View(
            VGroup(
                VGroup(
                    Item('show_maximum',
                         label='Show MAP estimate (circle)'),
                    Item('show_majority_vote',
                         label='Show majority vote (triangle)'),
                ),
                VGroup(
                    Item('posterior_plot',
                         editor=InstanceEditor(),
                         style='custom',
                         show_label=False),
                ),
                padding=0
            ),
            height = height,
            scrollable=True,
            resizable=True
        )

        return traits_view

    def _show_maximum_changed(self):
        plot = self.posterior_plot
        if self.show_maximum:
            maximum = plot.posterior.argmax(1)
            plot.add_markings(maximum, 'MAP',
                              'circle', 0., 0.,
                              marker_size=3,
                              line_width=2.,
                              marker_color='gray')
        else:
            plot.remove_markings('MAP')
        plot.plot_posterior.request_redraw()


    def _show_majority_vote_changed(self):
        plot = self.posterior_plot
        if self.show_majority_vote:
            majority = majority_vote(self.annotations)
            plot.add_markings(majority, 'majority',
                              'triangle', 0., 0.,
                              marker_size=3,
                              line_width=1.,
                              marker_color='green')
        else:
            plot.remove_markings('majority')
        plot.plot_posterior.request_redraw()


#### Testing and debugging ####################################################

def main():
    """ Entry point for standalone testing/debugging. """

    from pyanno.modelBt_loopdesign import ModelBtLoopDesign

    model = ModelBtLoopDesign.create_initial_state(5)
    annotations = model.generate_annotations(400)
    posterior = model.infer_labels(annotations)

    post_plot = PosteriorPlot(posterior=posterior,
                              title='Posterior over classes')

    post_view = PosteriorView(
        posterior_plot=post_plot,
        annotations=annotations
    )
    post_view.configure_traits()
    return post_view


if __name__ == '__main__':
    mv = main()
