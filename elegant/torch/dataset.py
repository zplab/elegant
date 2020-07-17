from torch.utils import data
from zplib.image import colorize
import freeimage

from elegant import process_images
from elegant import worm_spline

class WormDataset(data.Dataset):
    def __init__(self, timepoints, generate_data):
        super().__init__()
        self.timepoints = timepoints
        self.generate_data = generate_data

    def __len__(self):
        return len(self.timepoints)

    def __getitem__(self, i):
        timepoint = self.timepoints[i]
        return generate_data(timepoint)


def normalized_bf_image(timepoint):
    bf = freeimage.read(timepoint.image_path('bf'))
    mode = process_images.get_image_mode(bf, optocoupler=timepoint.position.experiment.metadata['optocoupler'])
    # map image image intensities in range (100, 2*mode) to range (0, 2)
    bf = colorize.scale(bf, min=100, max=2*mode, output_max=2)
    # now shift range to (-1, 1)
    bf -= 1
    return bf

class GenerateWormFrame:
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


