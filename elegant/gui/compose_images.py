# This code is licensed under the MIT License (see LICENSE file for details)
"""
Useful tools for scaling and colorizing images that look like what you see in ris_widget.

compose_image: takes the scalings from ris_widget, potentially with multiple
colorized layers, and gives you a RGB image that looks just like that (with
optional cropping to a specific region of interest defined by a box you can
draw on the GUI).

generate_images_from_flipbook: same, but does this for every flipbook page --
useful for saving movies

pin_image_mode: set an image's mode (peak of the intensity histogram) to a
specific value. Useful for making brightfield images directly comparable, so
that movies don't flicker and cause epilepsy :)

pin_flipbook_modes: do the above for every brightfield image in the flipbook.
Then you can adjust the min/max/gamma just so, set an ROI, and use
generate_images_from_flipbook to save a movie.

Example:

from ris_widget import ris_widget
import pathlib

p = pathlib.Path('path/to/image_dir')
rw = ris_widget.RisWidget()
bf = sorted(p.glob('* bf.png'))
gfp = [b.parent / (b.name.split(' ')[0] + ' gfp.png') for b in bf]
af = [b.parent / (b.name.split(' ')[0] + ' green_yellow_excitation_autofluorescence.png') for b in bf]
rw.add_image_files_to_flipbook(zip(bf, gfp, af))

from zplib.gui import compose_images
roi = compose_images.add_roi(rw)
compose_images.pin_flipbook_modes(rw)

# adjust tint colors, scalings, ROI, etc. in GUI

# write a single image
i = compose_images.compose_image(rw, roi)
import freeimage
freeimage.write(i, 'test.png')

# write a whole movie
from zplib.image import write_movie
write_movie.write_movie(compose_image.generate_images_from_flipbook(rw, roi, downsample_factor=2.5), 'movie.mp4', framerate=24)
"""

import numpy

from ris_widget.overlay import roi
from zplib.image import colorize
from zplib.image import pyramid

from .. import process_images

def add_roi(ris_widget):
    """Convenience function to add a rectangular ROI selector to the ris_widget.

    Once added, the ROI can be drawn by clicking and releasing to start the ROI
    drawing, then dragging the region to the desired size and clicking again.

    To resize the ROI, click it to select, then drag the handles. To delete the
    ROI, press delete when it is selected. Upon the next click, a deleted ROI
    will start drawing again. To cancel drawing, press escape. To remove the ROI
    entirely, use roi.remove()

    Returns: roi instance.
    """
    return roi.RectROI(ris_widget)

def compose_image(ris_widget, roi=None, downsample_factor=None, fast_downsample=False, layer_images=None):
    """
    Return an RGB, uint8 image representing the ris_widget image displayed.

    This is better than taking a snapshot using the ris_widget tool, because it
    returns a full-resolution image, or one downsampled with precise control.
    In addition, if an ROI is provided, the image will be cropped to that ROI.

    The image scaling (min, max, gamma), color tint, opacity, and blend mode are
    taken from the ris_widget layers. The images can be taken from ris_widget
    as well (default) or provided separately.

    To scale a single image from uint16 to uint8 using the current ris_widget
    min/max/gamma, this function is overkill. For that, one merely needs to run:
        scaled = zplib.image.colorize.scale(image, ris_widget.layer.min,
            ris_widget.layer.max, ris_widget.layer.gamma, output_max=255).astype(numpy.uint8)

    Parameters:
        ris_widget: a ris_widget instance.
        roi: a RectROI instance (e.g. provided by the add_roi() convenience
            function) to provide the crop area. If None, or if no ROI is drawn,
            the full image will be used.
        downsample_factor: if not None, amount to shrink the image by (fold-change)
        fast_downsample: if True and if downsample_factor is an integer, perform
            no smoothing upon downsampling. If False, smooth the image before
            downsampling to avoid aliasing.
        layer_images: if not None, a list of images to scale/tint/crop/composite
            using the ris_widget settings and ROI.
    """
    if downsample_factor is not None and fast_downsample:
        fast_downsample = int(downsample_factor) != downsample_factor
    use_roi = False
    if roi is not None and roi.geometry is not None:
        (x1, y1), (x2, y2) = numpy.round(roi.geometry).astype(int)
        use_roi = True
    images = []
    colors = []
    alphas = []
    modes = []
    if layer_images is None:
        layer_images = [layer.image for layer in ris_widget.layers]
    for image, layer in list(zip(layer_images, ris_widget.layers)):
        if image is not None and layer.visible:
            image = image.data
            if use_roi:
                image = image[x1:x2+1, y1:y2+1]
            if downsample_factor is not None:
                if fast_downsample:
                    image[::int(downsample_factor), ::int(downsample_factor)]
                else:
                    image = pyramid.pyr_down(image, downsample_factor)
            images.append(colorize.scale(image, layer.min, layer.max, layer.gamma, output_max=1))
            colors.append(layer.tint[:3])
            alphas.append(layer.tint[3])
            modes.append(layer.blend_function)
    composited, alpha = colorize.multi_blend(images, colors, alphas, modes)
    if alpha != 1:
        composited = colorize.blend(composited, numpy.zeros_like(composited), alpha)
    return (composited * 255).astype(numpy.uint8)

def generate_images_from_flipbook(ris_widget, roi=None, downsample_factor=None, fast_downsample=False):
    """Yield a series of composited images from the ris_widget flipbook, using
    the current image scalings and ROI. (See compose_image() documentation.)

    This is useful for writing a movie from ris_widget images via the functions
    in zplib.image.write_movie.
    """
    for layer_images in ris_widget.flipbook_pages:
        yield compose_image(ris_widget, roi, downsample_factor, fast_downsample, layer_images)

def pin_flipbook_modes(ris_widget, layer=0, noise_floor=200, new_mode=24000, optocoupler=None):
    """For every image in a given layer in the flipbook, pin its modal value
    as described in pin_image_mode(). Images are modified in-place.

    This is most useful for brightfield images, which may have different
    exposures, but can be made almost perfectly uniform in brightness by
    scaling by the image's histogram mode.

    Parameters:
        ris_widget: a ris_widget instance.
        layer: the layer whose images should be modified (brightfield images
            are generally layer 0).
        noise_floor: the "zero value" for the image (e.g. the camera noise floor)
        new_mode: the value to set the mode of the image to.
        optocoupler: magnification of optocoupler (1 or 0.7) to use in defining
            the vignetted region (ignored for mode calculation); or None to use
            whole image.
    """
    for layer_images in ris_widget.flipbook_pages:
        image = layer_images[layer]
        image.data[:] = process_images.pin_image_mode(image.data, noise_floor, new_mode, optocoupler=optocoupler)
        image.refresh()