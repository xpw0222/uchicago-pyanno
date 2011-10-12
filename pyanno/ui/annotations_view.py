from traits.api import ModelView
from traitsui.api import View


class AnnotationsViewView(ModelView):
    """ Traits UI Model/View for  'AnnotationsView' objects.
    """

    pass

    pass


def main():
    """ Entry point for standalone testing/debugging. """

    from a import AnnotationsView


    model = AnnotationsView.random_model(5, 100)
    model_view = ModelView(model=model)
    model_view.configure_traits(view='debug_view')


if __name__ == '__main__':
    main()
