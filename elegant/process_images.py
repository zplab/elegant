# This code is licensed under the MIT License (see LICENSE file for details)

import functools

import numpy
import celiagg
import freeimage

from zplib.image import draw
try:
    from ris_widget import histogram
    HAS_RW_HIST = True
except ImportError:
    HAS_RW_HIST = False

from . import worm_spline

VIGNETTE_RADIUS_1X = 0.55

def flatfield_correct(position_root, timepoint, image_type):
    image = freeimage.read(position_root / f'{timepoint} {image_type}.png')
    flatfield = freeimage.read(position_root.parent / 'calibrations' / f'{timepoint} fl_flatfield.tiff')
    return image.astype(numpy.float32) * flatfield

def corrected_worm_frame_image(position_root, timepoint, image_type, center_tck, optocoupler=None):
    if image_type == 'bf':
        image = freeimage.read(position_root / f'{timepoint} {image_type}.png')
        image = pin_image_mode(image, optocoupler=optocoupler)
    else:
        image = flatfield_correct(position_root, timepoint, image_type)
    return worm_spline.to_worm_frame(image, center_tck)

def pin_image_mode(image, noise_floor=200, new_mode=24000,
        mask=None, optocoupler=None, cx=0.5, cy=0.5):
    """Set an image's modal intensity to a specified value.

    This is most useful for brightfield images, which may have different
    exposures, but can be made almost perfectly uniform in brightness by
    scaling by the image's histogram mode.

    Parameters:
        image: an image of uint8 or uint16 dtype
        noise_floor: the "zero value" for the image (e.g. the camera noise floor)
        new_mode: the value to set the mode of the image to.
        mask: optional boolean mask with True were the mode should be calculated
        optocoupler: magnification of optocoupler (as a float), which determines
            the vignette size. Should be 1 or 0.7, or None to not calculate a
            vignette mask.
        cx, cy: center of the vignetted region, as a fraction of image width
            or height, respectively.

    Returns: modified image
    """
    mode = get_image_mode(image, mask, optocoupler, cx, cy)
    fimage = image.astype(numpy.float32)
    fimage -= noise_floor
    fimage *= (new_mode - noise_floor) / (mode - noise_floor)
    fimage += noise_floor
    fimage.clip(0, None, out=fimage)
    return fimage.astype(image.dtype)

@functools.lru_cache(maxsize=10) # cache commonly-used masks
def vignette_mask(optocoupler, shape, cx=0.5, cy=0.5):
    """Return a boolean mask of the un-vignetted area of an image.

    Parameters:
        optocoupler: magnification of optocoupler (as a float), which determines
            the vignette size. Should be 1 or 0.7.
        shape: shape of desired mask.
        cx, cy: center of the vignetted region, as a fraction of image width
            or height, respectively.

    Returns: boolean mask
    """
    r = VIGNETTE_RADIUS_1X * optocoupler
    cx, cy, r = int(cx * shape[0]), int(cy * shape[1]), int(r * shape[0])
    path = celiagg.Path()
    path.ellipse(cx, cy, r, r)
    return draw.draw_mask(shape, path, antialias=False) > 0

def get_image_mode(image, mask=None, optocoupler=None, cx=0.5, cy=0.5):
    """Return the approximate modal value of the image's pixels.

    For uint16 and float images, a binned histogram is computed and the center
    of the heaviest bin is returned. This is more robust than actually taking
    the mode of a uint16 image, and is the only way to realistically calculate
    the mode of a float image. For uint8 images, the modal pixel value is
    returned.

    A mask can optionally be provided; in addition, information about the
    optocoupler used allows computation of a vignette mask. Only non-masked,
    non-vignetted regions will be used in calculating the modal value.

    Parameters:
        image: input image
        mask: optional boolean mask with True were the mode should be calculated
        optocoupler: magnification of optocoupler (as a float), which determines
            the vignette size. Should be 1 or 0.7, or None to not calculate a
            vignette mask.
        cx, cy: center of the vignetted region, as a fraction of image width
            or height, respectively.

    Returns: modal value
    """

    if mask is None and HAS_RW_HIST:
        if optocoupler is None:
            mask_geometry = None
        else:
            r = VIGNETTE_RADIUS_1X * optocoupler
            mask_geometry = (cx, cy, r)
        mode = _image_mode_rw(image, mask_geometry)
    else:
        if optocoupler is not None:
            vmask = vignette_mask(optocoupler, image.shape, cx, cy)
        if mask is None:
            if optocoupler is not None:
                mask = vmask
        else: # mask is not None
            mask = mask > 0
            if optocoupler is not None:
                mask &= vmask
        mode = _image_mode_numpy(image, mask)
    return mode

def _image_mode_rw(image, mask_geometry=None):
    image_min, image_max, hist = histogram.histogram(image, mask_geometry=mask_geometry)
    if image.dtype == numpy.uint8:
        return hist.argmax()
    elif image.dtype == numpy.uint16:
        # 1024-bin histogram has a 64-intensity-level width for a 16-bit image.
        hist_min = 0
        binwidth = 64
    else: # float image
        hist_min = image_min
        binwidth = (image_max - image_min) / 1024
    # return midpoint of bin, so add 0.5 to bin index
    return binwidth * (hist.argmax() + 0.5) + hist_min

def _image_mode_numpy(image, mask=None):
    if mask is None:
        pixels = image.flat
    else:
        pixels = image[mask]
    if image.dtype == numpy.uint8:
        return numpy.bincount(pixels).argmax()
    elif image.dtype == numpy.uint16:
        # make 64-level-wide bins, and return the midpoint of the bin
        return 64 * (numpy.bincount(pixels // 64).argmax() + 0.5)
    else:
        hist, bin_edges = numpy.histogram(pixels, bins=1024)
        maxbin = hist.argmax()
        # return bin midpoint
        return (bin_edges[maxbin] + bin_edges[maxbin+1]) / 2