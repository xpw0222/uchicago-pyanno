# Copyright (c) 2011, Enthought, Ltd.
# Author: Mark Dickinson and Martin Chilvers
# License: Modified BSD license (2-clause)


""" A progress dialog that shows a 'pulse' as opposed to progress ;^) """

from traits.api import Int
from pyanno.ui.appbase.pulse_progress_dialog import PulseProgressDialog

class MacOSXPulseProgressDialog(PulseProgressDialog):
    """ A progress dialog that shows a 'pulse' as opposed to progress.

    This is useful when you don't have enough information to let the user
    know the percentage completion etc, but you still want to let them know
    that something is happening!

    This subclass is specialized for Mac OS X, where a nicer animation is
    possible.

    By default, the dialog pulses every 40 milliseconds (25Hz).
    """

    #### 'PulseProgressDialog' protocol #######################################

    # The delay between pulses (in milliseconds).
    delay = Int(40)

    def pulse(self):
        """ Pulse the progress bar. """

        # The actual value passed to 'update' here isn't used by pyface because
        # 'max' is set to 0 (zero) which makes the progress bar 'pulse'!
        self.update(1)

        return

    #### 'IProgressDialog' protocol ###########################################

    can_cancel   = False
    max          = 0
    show_percent = False
    show_time    = False

def main():
    """ Entry point for standalone testing/debugging. """

    from pyface.api import GUI

    gui = GUI()

    progress_dialog = MacOSXPulseProgressDialog(
        title='Test', message='Doing something possibly interesting...'
    )
    progress_dialog.open()

    gui.invoke_after(3000, progress_dialog.close)
    gui.start_event_loop()

    return


if __name__ == '__main__':
    main()
