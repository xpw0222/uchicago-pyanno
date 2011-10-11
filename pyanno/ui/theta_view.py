"""TraitsUI view of the Theta parameters, and their samples."""
from chaco.array_plot_data import ArrayPlotData
from chaco.data_range_2d import DataRange2D
from chaco.label_axis import LabelAxis
from chaco.plot import Plot
from chaco.scales.scales import FixedScale
from chaco.scales_tick_generator import ScalesTickGenerator
from enable.component_editor import ComponentEditor
from traits.has_traits import HasTraits, on_trait_change
from traits.trait_numeric import Array
from traits.trait_types import Instance, Bool, Event
from traitsui.item import Item
from traitsui.view import View

import numpy as np

class ThetaView(HasTraits):

    theta = Array(dtype=float, shape=(1, None))
    theta_samples = Array(dtype=float, shape=(None, None))
    theta_samples_valid = Bool(False)
    redraw = Event

    theta_plot = Instance(Plot)
    theta_plot_data = Instance(ArrayPlotData)

    @on_trait_change('theta_plot_data,theta_samples_valid,redraw')
    def _update_plot_data(self):
        print 'udpate'
        if not self.theta_plot_data:
            self.theta_plot_data = ArrayPlotData()

        plot_data = self.theta_plot_data

        for idx, th in enumerate(self.theta[0,:]):
            plot_data.set_data('x%d' % idx, [th])
            plot_data.set_data('y%d' % idx, [idx+1.2])

            # add samples if defined
            if self.theta_samples_valid:
                samples = np.sort(self.theta_samples[:,idx])
                nsamples = samples.shape[0]
                perc5 = samples[int(nsamples*0.05)]
                perc95 = samples[int(nsamples*0.95)]
                avg_pstd = samples.mean() + samples.std()
                avg_mstd = samples.mean() - samples.std()
                plot_data.set_data('xsamples%d' % idx, samples)
                plot_data.set_data('ysamples%d' % idx,
                                   np.zeros((nsamples,)) + idx+1.2)
                plot_data.set_data('min%d' % idx, [perc5])
                plot_data.set_data('max%d' % idx, [perc95])
                plot_data.set_data('barmin%d' % idx, [avg_mstd])
                plot_data.set_data('barmax%d' % idx, [avg_pstd])
                plot_data.set_data('avg%d' % idx, [samples.mean()])
                plot_data.set_data('index%d' % idx, [float(idx)+0.8])

        self.theta_plot = self._theta_plot_default()

    def _theta_plot_default(self):
        print 'theta', self.theta
        if not self.theta_plot_data:
            self._update_plot_data()

        theta_len = self.theta.shape[1]

        theta_plot = Plot(self.theta_plot_data)
        if self.theta_samples_valid:
             for idx in range(theta_len):
                 theta_plot.candle_plot(('index%d' % idx,
                                         'min%d' % idx,
                                         'barmin%d' % idx,
                                         'avg%d' % idx,
                                         'barmax%d' % idx,
                                         'max%d' % idx),
                                          color = "lightgray",
                                          bar_line_color = "black",
                                          stem_color = "blue",
                                          center_color = "red",
                                          center_width = 2)

                 theta_plot.plot(('ysamples%d' % idx, 'xsamples%d' % idx),
                                 type='scatter',
                                 color='red',
                                 marker='plus',
                                 line_width=1,
                                 marker_size=3)
        for idx in range(theta_len):
            theta_plot.plot(('y%d' % idx, 'x%d' % idx),
                            type='scatter',
                            color='black',
                            marker='plus',
                            marker_size=8,
                            line_width=2)

        # adjust axis bounds
        xlow = min(0.6, self.theta.min()-0.05)
        if self.theta_samples_valid:
            xlow = min(xlow, self.theta_samples.min()-0.05)
        range2d = DataRange2D(low=(0., xlow),
                              high=(self.theta.shape[1]+1, 1.))
        theta_plot.range2d = range2d

        # remove horizontal grid and axis
        theta_plot.underlays = [theta_plot.x_grid, theta_plot.y_axis]

        # create new vertical axis
        label_list = [str(i) for i in range(1, theta_len+1)]

        label_axis = LabelAxis(
            theta_plot,
            orientation = 'bottom',
            positions = range(1, theta_len+1),
            labels = label_list,
            label_rotation = 0
        )
        # use a FixedScale tick generator with a resolution of 1
        label_axis.tick_generator = ScalesTickGenerator(scale=FixedScale(1.))

        theta_plot.index_axis = label_axis
        theta_plot.underlays.append(label_axis)

        # title and axis name
        theta_plot.title = 'Accuracy of annotators (theta)'
        # some padding right, on the bottom
        #theta_plot.padding = [0, 15, 0, 25]

        return theta_plot

    traits_view = View(
        Item('theta_plot',
             editor=ComponentEditor(),
             resizable=True,
             show_label=False,
             #height=-100,
            ),
    )

#### Testing and debugging ####################################################

def main():
    """ Entry point for standalone testing/debugging. """

    from pyanno.modelBt import ModelBt
    import numpy as np

    model = ModelBt.create_initial_state(5)
    annotations = model.generate_annotations(model.generate_labels(100))
    theta_samples = model.sample_posterior_over_theta(annotations, 100,
                                                      step_optimization_nsamples=3)

    theta_view = ThetaView(theta=model.theta[None,:],
                           theta_samples=theta_samples)
    theta_view.theta_samples_valid = True
    theta_view.configure_traits(view='traits_view')

    return model, theta_view


if __name__ == '__main__':
    model, theta_view = main()
