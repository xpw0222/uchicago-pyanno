"""View for model and data pair."""
from traits.has_traits import HasTraits, on_trait_change
from traits.trait_types import Any, File, Instance, Button, Enum, Str, Range, Bool, Float, Event, List
from traits.traits import Property
from traitsui.group import HGroup, VGroup
from traitsui.handler import ModelView
from traitsui.item import Item, Spring
from traitsui.menu import OKCancelButtons
from traitsui.view import View
from pyanno import ModelBt, ModelB
from pyanno.annotations import AnnotationsContainer
from pyanno.ui.annotation_stat_view import AnnotationsStatisticsView
from pyanno.ui.annotations_view import AnnotationsView
from pyanno.ui.model_bt_view import ModelBtView
from pyanno.ui.model_b_view import ModelBView

import numpy as np



# TODO remember last setting of parameters


class ModelDataView(HasTraits):

    model_name = Enum('Model B-with-theta',
                      'Model B',
                      'Model A [not ready yet]')
    _model_name_to_class = {
        'Model B-with-theta': ModelBt,
        'Model B': ModelB
    }
    _model_name_to_view = {
        'Model B-with-theta': ModelBtView,
        'Model B': ModelBView
    }

    model = Any
    model_view = Instance(ModelView)
    model_updated = Event
    model_update_suspended = Bool(False)

    #annotations_container = Instance(AnnotationsContainer)
    annotations_file = File
    annotations_are_defined = Bool(False)
    annotations_updated = Event
    annotations_view = Instance(AnnotationsView)
    annotations_stats_view = Instance(AnnotationsStatisticsView)

    annotations = Property
    def _get_annotations(self):
        return self.annotations_view.annotations_container.annotations

    nclasses = Property
    def _get_nclasses(self):
        return max(self.model.nclasses, self.annotations.max())

    info_string = Str
    log_likelihood = Float

    @on_trait_change('annotations_file')
    def _update_annotations_file(self):
        print 'loading file', self.annotations_file
        anno = AnnotationsContainer.from_file(self.annotations_file)
        self.annotations_view = AnnotationsView(annotations_container = anno,
                                                nclasses = self.model.nclasses)
        self.annotations_stats_view = AnnotationsStatisticsView(
            annotations = self.annotations,
            nclasses = self.nclasses
        )

        self.annotations_are_defined = True
        self.annotations_updated = True

    @on_trait_change('annotations_updated,model_updated')
    def _update_log_likelihood(self):
        if self.annotations_are_defined:
            self.log_likelihood = self.model.log_likelihood(self.annotations)

    @on_trait_change('model,model:theta,model:gamma')
    def _fire_model_updated(self):
        if not self.model_update_suspended:
            self.model_updated = True

    def _annotations_view_default(self):
        anno = AnnotationsContainer.from_array([[0]], name='<undefined>')
        return AnnotationsView(annotations_container = anno,
                               nclasses = self.model.nclasses)

    @on_trait_change('model.nclasses')
    def _update_nclasses(self):
        self.annotations_view.nclasses = self.model.nclasses
        self.annotations_view.annotations_updated = True

    ### Actions ##############################################################

    ### Model creation actions
    # FIXME tooltip begins with "specifies..."
    new_model = Button(label='New model...')
    # TODO: get_info shows docstring
    get_info_on_model = Button(label='Info...')

    ml_estimate = Button(label='ML estimate...',
                         desc=('Maximum Likelihood estimate of model '
                               'parameters'))
    map_estimate = Button(label='MAP estimate...')
    sample_theta_posterior = Button(label='Sample parameters...')
    estimate_labels = Button(label='Estimate labels...')

    def _new_model_fired(self):
        """Create new model."""

        # delegate creation to associated model_view
        model_name = self.model_name
        responsible_view = self._model_name_to_view[model_name]

        # model == None if the user cancelled the action
        model, model_view = responsible_view.create_model_dialog()
        if model is not None:
            self.model = model
            self.model_view = model_view
            self.model_updated = True

    def _ml_estimate_fired(self):
        """Run ML estimation of parameters."""
        print 'ML estimate...'
        self.model_update_suspended = True
        self.model.mle(self.annotations, estimate_gamma=True)
        self.model_update_suspended = False
        # TODO change this into event listener (self.model_updated)
        self._fire_model_updated()
        self.model_view.model_updated = True

    def _map_estimate_fired(self):
        """Run ML estimation of parameters."""
        print 'MAP estimate...'
        self.model_update_suspended = True
        self.model.map(self.annotations, estimate_gamma=True)
        self.model_update_suspended = False
        # TODO change this into event listener (self.model_updated)
        self._fire_model_updated()
        self.model_view.model_updated = True

    def _sample_theta_posterior_fired(self):
        """Sample the posterior of the parameters `theta`."""
        print 'Sample...'
        self.model_update_suspended = True
        nsamples = 100
        samples = self.model.sample_posterior_over_theta(self.annotations,
                                                         nsamples,
                                                         step_optimization_nsamples=3)
        self.model_update_suspended = False
        self.model_view.plot_theta_samples(samples)


    ### Views ################################################################

    def traits_view(self):
        ## Model view

        model_create_group = (
            HGroup(
                Item(name='model_name',show_label=False),
                Item(name='new_model', show_label=False),
                Item(name='get_info_on_model', show_label=False,
                     enabled_when='False'),
                show_border=True
            )
        )

        model_group = (
            VGroup (
                model_create_group,
                Item('model_view', style='custom', show_label=False),
                show_border = True,
                label = 'Model view',
            )
        )

        ## Data view

        data_create_group = VGroup(
            Item('annotations_file', style='simple', label='Annotations file',
                 width=400),
            show_border = True,
        )

        data_info_group = VGroup(
            Item('annotations_view',
                 style='custom',
                 show_label=False,
                 visible_when='annotations_are_defined'),
            Item('annotations_stats_view',
                 style='custom',
                 show_label=False,
                 visible_when='annotations_are_defined')
        )

        data_group = (
            VGroup (
                data_create_group,
                data_info_group,
                show_border = True,
                label = 'Data view',
            )
        )

        ## (Model,Data) view

        model_data_group = (
            VGroup(
                Item('info_string', show_label=False, style='readonly'),
                Item('log_likelihood', label='Log likelihood', style='readonly'),
                HGroup(
                    Item('ml_estimate',
                         enabled_when='annotations_are_defined',
                         show_label=False),
                    Item('map_estimate',
                         enabled_when='annotations_are_defined',
                         show_label=False),
                    Item('sample_theta_posterior',
                         enabled_when='annotations_are_defined',
                         show_label=False),
                    Item('estimate_labels',
                         #enabled_when='annotations_are_defined',
                         show_label=False,
                         enabled_when='False'),
                ),
                show_border = True,
                label = 'Model-data view'
            )
        )


        ## Full view

        full_view = View(
            VGroup(
                HGroup(
                    model_group,
                    data_group
                ),
                model_data_group
            ),
            title='PyAnno - Models of data annotations by multiple curators',
            width = 1200,
            height = 800,
            resizable = True
        )

        return full_view



#### Testing and debugging ####################################################

def main():
    """ Entry point for standalone testing/debugging. """

    model = ModelB.create_initial_state(5, 8)
    model_data_view = ModelDataView(model=model,
                                    model_view=ModelBView(model=model))
    model_data_view.configure_traits(view='traits_view')

    return model, model_data_view


if __name__ == '__main__':
    m, mdv = main()
