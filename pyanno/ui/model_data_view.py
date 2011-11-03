# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

"""View for model and data pair."""
from traits.has_traits import HasTraits, on_trait_change
from traits.trait_types import (Any, File, Instance, Button, Enum, Str, Bool,
                                Float, Event, Int)
from traits.traits import Property
from traitsui.editors.range_editor import RangeEditor
from traitsui.group import HGroup, VGroup
from traitsui.handler import ModelView
from traitsui.item import Item, Label
from traitsui.menu import OKCancelButtons
from traitsui.view import View
from traitsui.message import error
from pyanno import ModelBt, ModelB, ModelA
from pyanno.annotations import AnnotationsContainer
from pyanno.plots.annotations_plot import PosteriorPlot
from pyanno.ui.annotation_stat_view import AnnotationsStatisticsView
from pyanno.ui.annotations_view import AnnotationsView
from pyanno.ui.appbase.long_running_call import LongRunningCall
from pyanno.ui.model_a_view import ModelAView
from pyanno.ui.model_bt_view import ModelBtView
from pyanno.ui.model_b_view import ModelBView

import numpy as np

# TODO remember last setting of parameters
from pyanno.ui.posterior_view import PosteriorView
from pyanno.util import PyannoValueError

import logging
logger = logging.getLogger(__name__)


class ModelDataView(HasTraits):

    #### Information about available models

    model_name = Enum('Model B-with-theta',
                      'Model B',
                      'Model A')

    _model_name_to_class = {
        'Model B-with-theta': ModelBt,
        'Model B': ModelB,
        'Model A': ModelA
    }

    _model_class_to_view = {
        ModelBt: ModelBtView,
        ModelB: ModelBView,
        ModelA: ModelAView
    }

    #### Model-related traits

    # the annotations model
    model = Any

    # Traits UI view of the model
    model_view = Instance(ModelView)

    # fired when the model is updates
    model_updated = Event

    # parameters view should not update when this trait is False
    model_update_suspended = Bool(False)


    #### Annotation-related traits

    # File trait to load a new annotations file
    annotations_file = File

    # True then annotations are loaded correctly
    annotations_are_defined = Bool(False)

    # fired when annotations are updated
    annotations_updated = Event

    # Traits UI view of the annotations
    annotations_view = Instance(AnnotationsView)

    # Traits UI view of the annotations' statistics
    annotations_stats_view = Instance(AnnotationsStatisticsView)

    # shortcut to the annotations
    annotations = Property
    def _get_annotations(self):
        return self.annotations_view.annotations_container.annotations

    # property that combines information from the model and the annotations
    # to give a consistent number of classes
    nclasses = Property
    def _get_nclasses(self):
        return max(self.model.nclasses, self.annotations.max() + 1)

    # info string -- currently not used
    info_string = Str

    # used to display the current log likelihood
    log_likelihood = Float


    def _annotations_view_default(self):
        anno = AnnotationsContainer.from_array([[0]], name='<undefined>')
        return AnnotationsView(annotations_container = anno,
                               nclasses = self.model.nclasses)


    @on_trait_change('annotations_file')
    def _update_annotations_file(self):
        logger.info('Load file {}'.format(self.annotations_file))
        anno = AnnotationsContainer.from_file(self.annotations_file)
        self.set_annotations(anno)


    @on_trait_change('annotations_updated,model_updated')
    def _update_log_likelihood(self):
        if self.annotations_are_defined:
            if not self.model.are_annotations_compatible(self.annotations):
                self.log_likelihood = np.nan
            else:
                self.log_likelihood = self.model.log_likelihood(
                    self.annotations)


    @on_trait_change('model.nclasses')
    def _update_nclasses(self):
        self.annotations_view.nclasses = self.model.nclasses
        self.annotations_view.annotations_updated = True


    @on_trait_change('model,model:theta,model:gamma')
    def _fire_model_updated(self):
        if not self.model_update_suspended:
            self.model_updated = True
            if self.model_view is not None:
                self.model_view.model_updated = True


    ### Control content #######################################################

    def set_model(self, model):
        """Update window with a new model.
        """
        self.model = model
        model_view_class = self._model_class_to_view[model.__class__]
        self.model_view = model_view_class(model=model)
        self.model_update = True


    def set_annotations(self, annotations_container):
        """Update window with a new set of annotations."""
        self.annotations_view = AnnotationsView(
            annotations_container = annotations_container,
            nclasses = self.model.nclasses
        )
        self.annotations_stats_view = AnnotationsStatisticsView(
            annotations = self.annotations,
            nclasses = self.nclasses
        )

        self.annotations_are_defined = True
        self.annotations_updated = True


    def set_from_database_record(self, record):
        """Set main window model and annotations from a database record."""
        self.set_model(record.model)
        self.set_annotations(record.anno_container)


    ### Actions ##############################################################

    #### Model creation actions

    # FIXME tooltip begins with "specifies..."

    # create a new model
    new_model = Button(label='New model...')

    # TODO: get_info shows docstring
    # show informations about the selected model
    get_info_on_model = Button(label='Info...')


    #### Model <-> data computations

    # execute Maximum Likelihood estimation of parameters
    ml_estimate = Button(label='ML estimate...',
                         desc=('Maximum Likelihood estimate of model '
                               'parameters'))

    # execute MAP estimation of parameters
    map_estimate = Button(label='MAP estimate...')

    # draw samples from the posterior over accuracy
    sample_posterior_over_accuracy = Button(label='Sample parameters...')

    # compute posterior over label classes
    estimate_labels = Button(label='Estimate labels...')


    def _new_model_fired(self):
        """Create new model."""

        # delegate creation to associated model_view
        model_name = self.model_name
        model_class = self._model_name_to_class[model_name]
        responsible_view = self._model_class_to_view[model_class]

        # model == None if the user cancelled the action
        model = responsible_view.create_model_dialog()
        if model is not None:
            self.set_model(model)


    def _action_finally(self):
        """Operations that need to be executed both in case of a success and
        a failure of the long-running action.
        """
        self.model_update_suspended = False


    def _action_success(self, result):
        self._action_finally()
        self._fire_model_updated()


    def _action_failure(self, err):
        self._action_finally()

        if isinstance(err, PyannoValueError):
            errmsg = err.args[0]
            if 'Annotations' in errmsg:
                # raised when annotations are incompatible with the model
                error('Error: ' + errmsg)
            else:
                # re-raise exception if it has not been handled
                raise


    def _action_on_model(self, message, method, args=None, kwargs=None,
                         on_success=None, on_failure=None):
        """Call long running method on model.

        While the call is running, a window with a pulse progress bar is
        displayed.

        An error message is displayed if the call raises a PyannoValueError
        (raised when annotations are incompatible with the current model).
        """

        if args is None: args = []
        if kwargs is None: kwargs = {}

        if on_success is None: on_success = self._action_success
        if on_failure is None: on_failure = self._action_failure

        self.model_update_suspended = True

        call = LongRunningCall(
            parent     = None,
            title      = 'Calculating...',
            message    = message,
            callable   = method,
            args       = args,
            kw         = kwargs,
            on_success = on_success,
            on_failure = on_failure,
        )
        call()


    def _ml_estimate_fired(self):
        """Run ML estimation of parameters."""

        message = 'Computing maximum likelihood estimate'
        self._action_on_model(message, self.model.mle, args=[self.annotations])


    def _map_estimate_fired(self):
        """Run ML estimation of parameters."""

        message = 'Computing maximum a posteriori estimate'
        self._action_on_model(message, self.model.map, args=[self.annotations])


    def _sample_posterior_success(self, samples):
        if (samples is not None
            and hasattr(self.model_view, 'plot_theta_samples')):
            self.model_view.plot_theta_samples(samples)

        self._action_finally()


    def _sample_posterior_over_accuracy_fired(self):
        """Sample the posterior of the parameters `theta`."""

        message = 'Sampling from the posterior over accuracy'

        # open dialog asking for number of samples
        params = _SamplingParamsDialog()
        dialog_ui = params.edit_traits(kind='modal')

        if not dialog_ui.result:
            # user pressed "Cancel"
            return

        nsamples = params.nsamples

        self._action_on_model(
            message,
            self.model.sample_posterior_over_accuracy,
            args=[self.annotations, nsamples],
            on_success=self._sample_posterior_success
        )


    def _estimate_labels_success(self, posterior):
        if posterior is not None:
            post_plot = PosteriorPlot(posterior=posterior,
                                      title='Posterior over classes')

            post_view = PosteriorView(posterior_plot=post_plot,
                                      annotations=self.annotations)

            post_view.edit_traits()

        self._action_finally()


    def _estimate_labels_fired(self):
        """Compute the posterior over annotations and show it in a new window"""

        message = 'Computing the posterior over classes'
        self._action_on_model(
            message,
            self.model.infer_labels,
            args=[self.annotations],
            on_success=self._estimate_labels_success
        )


    ### Views ################################################################

    def traits_view(self):
        ## Model view

        model_create_group = (
            VGroup(
                HGroup(
                    Item(name='model_name',show_label=False, width=200),
                    Item(name='new_model', show_label=False, width=100),
                    Item(name='get_info_on_model', show_label=False,
                         enabled_when='False', width=100),
                    #show_border=True
                ),
                label = 'Create new model'
            )
        )

        model_group = (
            VGroup (
                model_create_group,
                VGroup(
                    Item(
                        'model_view',
                        style='custom',
                        show_label=False,
                        width=400
                    ),
                    #show_border = True,
                    label = 'Model view',
                )
            ),
        )

        ## Data view

        data_create_group = VGroup(
            Label('Open annotation file:', width=800),
            Item('annotations_file', style='simple', label='Annotations file',
                 show_label=False,
                 width=400),
            show_border = True,
        )

        data_info_group = VGroup(
            Item('annotations_view',
                 style='custom',
                 show_label=False,
                 visible_when='annotations_are_defined',
                 width=800
            ),
            Item('annotations_stats_view',
                 style='custom',
                 show_label=False,
                 visible_when='annotations_are_defined')
        )

        data_group = (
            VGroup (
                data_create_group,
                data_info_group,
                label = 'Data view',
            ),
        )

        ## (Model,Data) view

        model_data_group = (
            VGroup(
                #Item('info_string', show_label=False, style='readonly'),
                Item('log_likelihood', label='Log likelihood', style='readonly'),
                HGroup(
                    Item('ml_estimate',
                         enabled_when='annotations_are_defined',
                         show_label=False),
                    Item('map_estimate',
                         enabled_when='annotations_are_defined',
                         show_label=False),
                    Item('sample_posterior_over_accuracy',
                         enabled_when='annotations_are_defined',
                         show_label=False),
                    Item('estimate_labels',
                         enabled_when='annotations_are_defined',
                         show_label=False),
                ),
                label = 'Model-data view'
            )
        )


        ## Full view

        full_view = View(
            VGroup(
                HGroup(
                    model_group,
                    data_group,
                ),
                model_data_group
            ),
            title='PyAnno - Models of data annotations by multiple curators',
            width = 1200,
            height = 800,
            resizable = False
        )

        return full_view


class _SamplingParamsDialog(HasTraits):
    nsamples = Int(200)
    traits_view = View(
        Item('nsamples',
             label  = 'Number of samples',
             editor = RangeEditor(mode='xslider',
                                  low=100, high=50000,
                                  is_float=False)
        ),
        buttons = OKCancelButtons
    )


#### Testing and debugging ####################################################

def create_database():
    from tempfile import mktemp
    from pyanno.database import PyannoDatabase

    # database filename
    tmp_filename = mktemp(prefix='tmp_pyanno_db_')
    db = PyannoDatabase(tmp_filename)

    # populate database
    from pyanno.modelA import ModelA
    from pyanno.modelB import ModelB

    model = ModelA.create_initial_state(5)
    annotations = model.generate_annotations(100)
    value = model.log_likelihood(annotations)
    db.store_result('test_id', annotations, model, value)

    modelb = ModelB.create_initial_state(5, 8)
    value = modelb.log_likelihood(annotations)
    db.store_result('test_id', annotations, modelb, value)

    annotations = model.generate_annotations(100)
    value = model.log_likelihood(annotations)
    db.store_result('test_id2', annotations, model, value)

    return db


def main():
    """ Entry point for standalone testing/debugging. """
    from pyanno.ui.database_view import DatabaseView
    from pyanno.ui.model_data_view import ModelDataView

    model = ModelBt.create_initial_state(5)
    model_data_view = ModelDataView(model=model,
                                    model_view=ModelBtView(model=model))

    # create database view
    db = create_database()
    print model_data_view
    database_view = DatabaseView(database=db, model_data_view=model_data_view)
    database_view.edit_traits(view='traits_view', kind='live')

    # open model_data_view
    model_data_view.configure_traits(view='traits_view')

    return model, model_data_view


if __name__ == '__main__':
    m, mdv = main()
