from torch.utils import data
from zplib.image import colorize
import freeimage

from elegant import process_images
from elegant import worm_spline

class WormDataset(data.Dataset):
    """Adaptor from datamodel classes to Torch Dataset.

    The general idea is that you provide the constructor with a sequence of
    datamodel.Timepoint instances (generally this would be via constructing a
    datamodel.Timepoints instance, but other methods will work), and a callable
    generate_data function that takes a Timepoint and returns whatever Torch is
    intended to work with (either a single image for testing/running a convnet,
    or a image/target pair for training).

    The generate_data function can either be a regular function, or a callable
    class to enable parameters to be set to define the specific behavior of
    the function.

    Example:
    experiment_paths = [...] # list of paths to experiment directories
    experiments = [datamodel.Experiment(path) for path in experiment_paths]
    test, train = datamodel.Timepoints.split_experiments(*experiments, fractions=[0.3, 0.7])

    test_labframe_dataset = WormDataset(test, normalized_bf_image)

    generate_worm_image = GenerateWormFrame(image_size=(1000,200))
    test_wormframe_dataset = WormDataset(test, generate_worm_image)

    """
    def __init__(self, timepoints, generate_data):
        """Parameters:
            timepoints: sequence of datamodel.Timepoint instances
            generate_data: function (or callable class) that takes a Timepoint
                instance and returns whatever PyTorch needs.
        """
        super().__init__()
        self.timepoints = timepoints
        self.generate_data = generate_data

    def __len__(self):
        return len(self.timepoints)

    def __getitem__(self, i):
        timepoint = self.timepoints[i]
        return self.generate_data(timepoint)


def normalized_bf_image(timepoint):
    """Given a timepoint, return a normalized brightfield image."""
    bf = freeimage.read(timepoint.image_path('bf'))
    mode = process_images.get_image_mode(bf, optocoupler=timepoint.position.experiment.metadata['optocoupler'])
    # map image image intensities in range (100, 2*mode) to range (0, 2)
    bf = colorize.scale(bf, min=100, max=2*mode, output_max=2)
    # now shift range to (-1, 1)
    bf -= 1
    return bf

class GenerateWormFrame:
    """Callable class that returns a worm-frame image when called with a Timepoint instance.

    Shape of the worm-frame image can be configured at class initialization.
    """
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, timepoint):
        bf = normalized_bf_image(timepoint)
        annotations = timepoint.annotations
        center_tck, width_tck = annotations['pose']
        reflect = False
        if 'keypoints' in annotations and 'vulva' in annotations['keypoints']:
            x, y = annotations['keypoints']['vulva']
            reflect = y < 0
        image_width, image_height = self.image_shape
        worm_frame = worm_spline.to_worm_frame(bf, center_tck, width_tck,
            sample_distance=image_height//2, standard_length=image_width, reflect_centerline=reflect)
        mask = worm_spline.worm_frame_mask(width_tck, worm_frame.shape)
        worm_frame[mask == 0] = 0
        return worm_frame


