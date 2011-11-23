# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes, Mark Dickinson and Martin Chilvers
# License: Modified BSD license (2-clause)


""" A progress dialog that shows a 'pulse' as opposed to progress ;^) """


from pyface.api import GUI, ProgressDialog
from traits.api import Bool, Int


class PulseProgressDialog(ProgressDialog):
    """ A progress dialog that shows a 'pulse' as opposed to progress.

    This is useful when you don't have enough information to let the user
    know the percentage completion etc, but you still want to let them know
    that something is happening!

    By default, the dialog is refreshed every 40 milliseconds (25Hz).
    """

    #### 'PulseProgressDialog' protocol #######################################

    # The delay between pulses (in milliseconds).
    delay = Int(40)

    def pulse(self):
        """ Pulse the progress bar. """

        # The actual value passed to 'update' here isn't used by pyface because
        # 'max' is set to 0 (zero) which makes the progress bar 'pulse'!
        widget = self.progress_bar.control
        widget.Pulse()

        return

    #### 'IProgressDialog' protocol ###########################################

    can_cancel   = False
    max          = 100
    show_percent = False
    show_time    = False

    def open(self):
        """ Open the dialog. """
        self._schedule_pulse()
        super(PulseProgressDialog, self).open()

    def close(self):
        """ Close the dialog. """

        self._closing = True

        # We don't call our superclass' 'close' method immediately as there
        # may be one final call to '_pulse_and_reschedule' already on the
        # event queue. This makes sure that we call 'close' *after* that final
        # update.
        GUI.invoke_after(self.delay, super(PulseProgressDialog, self).close)

    def update(self, value):
        # overwrite because the superclass closes the dialog when the
        # progressbar reaches the end
        if self.progress_bar is not None:
            self.progress_bar.update(value)

    #### Private protocol #####################################################

    # Flag that is set when the dialog is closing.
    _closing = Bool(False)

    def _schedule_pulse(self):
        """ Schedule a pulse for 'self.delay' milliseconds from now. """
        GUI.invoke_after(self.delay, self._pulse_and_reschedule)

    def _pulse_and_reschedule(self):
        """ Pulse the progress bar and reschedule the next one!

        If the progress dialog is closing then we don't do anything.
        """

        if not self._closing:
            self.pulse()
            self._schedule_pulse()


def main():
    """ Entry point for standalone testing/debugging. """

    from pyface.api import GUI

    gui = GUI()

    progress_dialog = PulseProgressDialog(
        title='Test', message='Doing something possibly interesting...'
    )
    progress_dialog.open()

    gui.invoke_after(3000, progress_dialog.close)
    gui.start_event_loop()

    return


if __name__ == '__main__':
    main()
