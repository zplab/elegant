# This code is licensed under the MIT License (see LICENSE file for details)

import numpy

from zplib.scalar_stats import mcd
from zplib.image import colorize

def region_measures(image, mask):
    """Measure several basic region properties of a masked image.

    Parameters:
        image, mask: numpy.ndarrays of same shape.

    Returns: sum, mean, median, percentile_95, percentile_99
        representing the sum, mean, etc. of image pixels in non-zero region
        of the mask.
    """
    if mask.dtype != bool:
        mask = mask > 0
    pixels = image[mask]
    sum = pixels.sum()
    mean = pixels.mean()
    median, percentile_95, percentile_99 = numpy.percentile(pixels, [50, 95, 99])
    return [sum, mean, median, percentile_95, percentile_99]

REGION_FEATURES = 'sum mean median percentile_95 percentile_99'.split()

def subregion_measures(image, mask):
    """Measure region properties of a masked image, defining several sub-regions
    relevant to fluorescence images.

    Three sub-regions will be defined:
        expression: the region of a fluorescence image where there is
            apparent fluorescent protein expression (2.5 standard deviations
            above the background distribution of low-valued pixels in the
            masked region)
        high_expression: the region of apparent intense fluorescence (6 standard
            deviations above background)
        over_99: the region including the top 1% of pixels by intensity.

    Parameters:
        image, mask: numpy.ndarrays of same shape.

    Returns: data, region_masks
        data:
            [sum, mean, median, percentile_95, percentile_99,
            expression_sum, expression_mean, expression_median,
            expression_area_fraction,
            high_expression_sum, high_expression_mean, high_expression_median,
            high_expression_area_fraction,
            over_99_sum, over_99_mean, over_99_median]
            where the 'area_fraction' measurements represent the fraction of the
            mask area encompassed by the expression and high_expression areas.
        region_masks: [expression_mask, high_expression_mask, over_99_mask]
            where each is a mask defining the sub-region of the mask.
    """
    mask_measures = region_measures(image, mask)
    mean = mask_measures[1]
    pixels = image[mask]
    area = pixels.sum()
    low_mean, low_std = mcd.robust_mean_std(pixels[pixels < mean], 0.5)
    expression_mask = (image > (low_mean + 2.5*low_std)) & mask
    expression_measures = region_measures(image, expression_mask)[:3]
    expression_area_fraction = expression_mask.sum() / area
    expression_measures.append(expression_area_fraction)
    high_expression_mask = (image > (low_mean + 6*low_std)) & mask
    high_expression_measures = region_measures(image, expression_mask)[:3]
    high_expression_area_fraction = high_expression_mask.sum() / area
    high_expression_measures.append(high_expression_area_fraction)
    percentile_99 = mask_measures[-1]
    over_99_mask = (image > percentile_99) & mask
    over_99_measures = region_measures(image, expression_mask)[:3]
    data = mask_measures + expression_measures + high_expression_measures + over_99_measures
    region_masks = [expression_mask, high_expression_mask, over_99_mask]
    return data, region_masks


_EXPRESSION_FEATURES = ['expression_'+f for f in REGION_FEATURES[:3] + ['fraction']]
_HIGH_EXPRESSION_FEATURES = ['high_'+f for f in _EXPRESSION_FEATURES]
_OVER99_FEATURES = ['over_99_'+f for f in REGION_FEATURES[:3]]
SUBREGION_FEATURES = REGION_FEATURES + _EXPRESSION_FEATURES + _HIGH_EXPRESSION_FEATURES + _OVER99_FEATURES

def colorize_masks(mask, region_masks):
    """Given a mask and a set of region_masks as returned by subregion_measures,
    return a color image where the background is black, the mask is white, and
    each nested sub-region is a different color.
    """
    labels = numpy.array(mask, dtype=bool) # make copy, make sure masked area has value = 1
    for i, region_mask in enumerate(masks):
        labels[region_mask] = i + 2 # region mask values are 2 and up
    color = colorize.colorize_label_image(labels)
    color[labels == 1] = 255 # set the area where the original mask was not obsured to white, instead of the default color
    return color
