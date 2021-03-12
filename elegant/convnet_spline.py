# This code is licensed under the MIT License (see LICENSE file for details)

import numpy
from scipy import ndimage
from skimage import morphology
from skimage import graph

from zplib.image import canny
from zplib.curve import interpolate
from . import worm_spline


def find_centerline(ap_coords, dv_coords, mask, dv_sigma=2, worm_width=60,
        low_threshold=0.25, high_threshold=0.6,
        width_step=1, center_smoothing=0.1, width_smoothing=0.0001):
    """Convert convnet outputs into useful spline poses and related images.

    Parameters:
        ap_coords: floating-point image of anterior-posterior worm coordinates with
            posterior high-valued and anterior low. Should NOT be masked.
        dv_coords: floating-point image of dorsal-ventral worm coordinates with
            midline values high and edge values low. Should NOT be masked.
        mask: segmented mask (approximate is ok, as ridgeline may extend out of
            the masked region if the dv_coords image supports it).
        dv_sigma: gaussian blur applied to smooth the dv_coords image
        worm_width: approximate mean width of worms at their widest at this magnification.
            60 is a reasonable value for adult worms at 5x magnification.
        low_threshold: lowest dv-coordinate value that could possibly be part of the
            centerline.
        high_threshold: lowest dv-coordinate value that would almost certainly be
            part of the centerline.
        width_step: pixel step size for fitting width profile to mask image.
            (see documentation for fit_splines().)
        center_smoothing: average distance the center_tck spline is allowed
            to deviate from the input path coordinates.(see documentation for fit_splines().)
        width_smoothing: average distance the width_tck spline is allowed to deviate
            from the input dv coordinate values. (see documentation for fit_splines().)

    Returns: costs, centerline, center_path, pose
        costs: cost image (only evaluated near the masked pixels)
        centerline: image containing centerline pixels of the worm
        center_path: (n, 2)-shape array of coordinates along the centerline of a worm
        pose: (center_tck, width_tck) pose tuple
    """
    orig_shape = mask.shape
    slices, mask, (ap_coords, dv_coords) = _trim_images(mask, [ap_coords, dv_coords])
    centerlines = find_centerline_pixels(dv_coords, mask, dv_sigma, worm_width, low_threshold, high_threshold)[2]
    costs = get_costs(dv_coords)
    centerline, is_loop = connect_centerline(centerlines, costs)
    if is_loop:
        center_path = find_centerline_from_loop(centerline, ap_coords)
    else:
        center_path = worm_spline.longest_path(centerline)
        if numpy.hypot(*(center_path[0] - center_path[-1])) < 5: # basically a loop
            path, cost = graph.route_through_array(costs, center_path[0], center_path[-1])
            centerline[tuple(numpy.transpose(path))] = 1
            center_path = find_centerline_from_loop(centerline, ap_coords)
    center_path = orient_path_ascending(center_path, ap_coords)
    width_range = worm_width/1.5, worm_width*1.5
    pose = fit_splines(center_path, mask, dv_coords, width_range, width_step, center_smoothing, width_smoothing)
    costs, centerline = _untrim_images([costs, centerline], orig_shape, slices)
    # add back the trim offset to the centerline tck and coordinates
    offset = slices[0].start, slices[1].start
    c = pose[0][1] # the c value of the center_tck
    c += offset
    center_path += offset
    return costs, centerline, center_path, pose

### Helper functions are below

def _trim_images(mask, images, margin=30):
    """Crop a mask and related images to the masked region, plus some margin"""
    slices = ndimage.find_objects(mask)
    assert len(slices) == 1
    slicex, slicey = slices[0]
    # add a margin around the slice, but make sure that the low end doesn't go negative
    # no need to worry about too-large slices at the top end though; that works transparently
    slicex = slice(max(0, slicex.start - margin), slicex.stop + margin)
    slicey = slice(max(0, slicey.start - margin), slicey.stop + margin)
    slices = slicex, slicey
    mask = mask[slices]
    images = [image[slices] for image in images]
    return slices, mask, images

def _untrim_images(images, orig_shape, slices):
    """Place trimmed images back into a zero-value image of the original shape"""
    for image in images:
        new_image = numpy.zeros(orig_shape, image.dtype)
        new_image[slices] = image
        yield new_image


FULLY_CONNECTED = numpy.ones((3,3), dtype=bool)

def find_centerline_pixels(dv_coords, mask, sigma=1, worm_width=50, low_threshold=0.25, high_threshold=0.6):
    """ Find pixels along the centerline of the worm given a DV-coordinate image.

    Parameters:
        dv_coords: floating-point image of dorsal-ventral worm coordinates with
            midline values high and edge values low. Should NOT be masked.
        mask: segmented mask (approximate is ok, as ridgeline may extend out of
            the masked region if the dv_coords image supports it.
        sigma: gaussian blur applied to smooth the dv_coords image
        worm_width: approximate width of worms at their widest at this magnification.
            50 is a reasonable value for adult worms at 5x magnification.
        low_threshold: lowest dv-coordinate value that could possibly be part of the
            centerline.
        high_threshold: lowest dv-coordinate value that would almost certainly be
            part of the centerline.

    Returns: all_ridges, extended_ridges, neighboring_ridges
        all_ridges: all local ridglines in the image, regardless of mask. Mainly
            useful for diagnostics.
        extended_ridges: all low-confidence-or-better ridgelines extending from
            high-confidence regions within the mask. Strictest set of ridges.
        neighboring_ridges: all low-confidence-or-better ridgelines extending from
            locations within three pixels of the extended_ridges. Most inclusive
            set of ridges.
    """

    # A horizontal ridge of dv_coords will have a positive y gradient on the low-y side of the ridge
    # and a negative gradient on the other, with a zero-gradient patch right along the center of the
    # ridgeline:
    # y^
    #  |
    #  |-------- horizontal ridgeline of high-value dv-coordinates on a low-value background
    #  |
    #  --------->x
    #
    # Gradient in y direction:
    # y^
    #  |--------
    #  |00000000
    #  |++++++++
    #  --------->x
    #
    # A vertical ridge will have a positive x gradient on the low-x side and negative on the other.
    # A diagonal ridge will be a combination of the above, and the orientation of the diagonal
    # can be determined by whether the low-x and low-y sides of the ridge are the same.
    # I.e. if the ridge is +45 degrees, the low-x and low-y sides are opposite:
    #                    x-gradient:       y-gradient:
    #  y^               y^                y^
    #   |  / ridge       |++0-             |--0+
    #   | /              |+0--             |-0++
    #   |/               |0---             |0+++
    #   ------>x         ------>x          ------>x
    # In the above the low-x side of the diagonal ridgeline is above the diagonal and the
    # low-y side is below. So in this case, the x and y gradients on each side of the ridge
    # are of opposite sign.
    # However, if the ridge is -45 degrees, the low-x and low-y sides are both below
    # the ridge, and so the gradients have the same sign.
    #                    x-gradient:       y-gradient:
    #   ------>x         ------>x          ------>x
    #   |\               |0---             |0---
    #   | \              |+0--             |+0--
    #   |  \ ridge       |++0-             |++0-
    #  yv               yv                yv
    #
    # To find the ridgeline pixels (0 gradient surrounded by high + and - gradients on
    # either side), we could look for maxima of the second derivative,
    # but that is noise-prone in practice. Instead, we will take a local average of
    # the absolute value of the x- and y-gradients nearby each pixel. The 0-valued ridge
    # pixels will have high averages. This, however, discards the orientation information.
    # To recover the degree to which the y-gradient has an + or - diagonal angle,
    # we compare the local average of the product of the x and y gradients to
    # the local average of the product of the absolute values of the gradients.
    # This number ranges from -1 indicating that the x- and y-gradients are on
    # opposite sides of the ridgeline, to +1 indicating that they're on the same sides.
    # We then weight the local magnitude in the y direction by this number, to produce
    # an (x, y) vector that captures the direction normal to the ridge.
    #
    # Given these normals, we can then locate the exact center pixels of the ridge
    # by choosing pixels from the original dv_coordinate image that are local maxima
    # along the ridge normal. Even if the ridgeline itself is slowly climbing or
    # descending, the saddle points along the ridge will be maximal normal to the
    # ridge. This technique comes from the Canny edge-finding method.
    #
    # Finally, we also use Canny hysteresis thresholding to extend an initial seed
    # of high-confidence ridge pixels into potentially lower confidence regions:
    # If a low-confidence region of ridge is connected to a high-confidence region
    # (or in this implementation, within a few pixels therof), extend the ridge
    # into it.
    smooth_dv = ndimage.gaussian_filter(dv_coords, sigma)
    ridge = smooth_dv**3
    dx, dy = numpy.gradient(ridge)
    adx, ady = numpy.abs(dx), numpy.abs(dy)
    # average over a region about the worm's width, so choose a smoothing
    # gaussian with a standard deviation about 1/4 of the width, and truncate
    # at two standard deivations (1/2th of the width) on either side of the ridge
    gradient_sigma = worm_width / 4
    local_magnitude_x = ndimage.gaussian_filter(adx, gradient_sigma, truncate=2)
    local_magnitude_y = ndimage.gaussian_filter(ady, gradient_sigma, truncate=2)
    orientation = ndimage.gaussian_filter(dx*dy, gradient_sigma, truncate=2)
    max_orientation = ndimage.gaussian_filter(adx*ady, gradient_sigma, truncate=2)
    orientation /= max_orientation
    all_ridges = canny.canny_local_maxima(ridge, [local_magnitude_x, local_magnitude_y*orientation])
    # Hysteresis threshold: high_mask below is regions of the local_maxima image
    # that are within the overall mask region and also have a dv_coord above
    # the high-confidence value. Then we extend those regions into any connected
    # low-confidence regions (the low_mask regions), ignoring low-confidence
    # regions not contiguous with a high-confidence region.
    high_mask = mask & all_ridges & (smooth_dv >= high_threshold)
    low_mask = all_ridges & (smooth_dv >= low_threshold)
    extended_ridges = ndimage.binary_propagation(high_mask, mask=low_mask, structure=FULLY_CONNECTED)
    # Now add regions of the maxima that are just a couple pixels away from the
    # extended regions too.
    # nearby_pixels below is regions of the low_mask that are both within the overall
    # mask and near to regions in extended_ridges. Then expand out such pixels to
    # all connected pixels within low_mask
    nearby_pixels = mask & low_mask & ndimage.binary_dilation(extended_ridges, iterations=3, structure=FULLY_CONNECTED)
    neighboring_ridges = ndimage.binary_propagation(nearby_pixels, mask=low_mask, structure=FULLY_CONNECTED)
    neighboring_ridges = morphology.thin(neighboring_ridges)
    return all_ridges, extended_ridges, neighboring_ridges

def connect_centerline(centerlines, dv_costs):
    """Connect a set of centerline pixel segments into a single spanning tree.

    Parameters:
        centerlines: a mask image of centerline pixels, as returned by find_centerline_pixels()
        dv_costs: a traversal-cost matrix based on the dv coordinages, as returned by get_costs()

    Returns: connected_centerline, is_loop
        connected_centerline: mask image with a single connected component: either
            a large loop (containing over 85% of the centerline pixels), or a
            spanning tree of the non-loop components where all of the endpoints
            are mutually reachable via the centerline pixels, or the lowest-cost
            path between components.
        is_loop: True if a single large loop or lariat (with 0 or 1 endpoints),
            was returned; if False if a spanning tree with two or more endpoints
            was returned.
    """
    # retain only pixels that have at least one nonzero neighbor (i.e. remove stray singletons)
    centerlines = centerlines & ndimage.maximum_filter(centerlines, footprint=[[1,1,1], [1,0,1], [1,1,1]])
    costs = numpy.array(dv_costs)
    costs[centerlines] = 0
    endpoints = worm_spline.get_endpoints(centerlines)
    ep_indices = numpy.transpose(endpoints.nonzero())
    # check for any large loops/lariats, and if found, just return that for special handling
    # NB: no handling of trying to connect segments to loops; too complex
    labels, num_segments = ndimage.label(centerlines, structure=FULLY_CONNECTED)
    centerline_px = centerlines.sum()
    for i in range(1, num_segments+1):
        segment_mask = labels == i
        segment_endpoints = segment_mask & endpoints
        if segment_endpoints.sum() < 2:
            # the segment contains a loop
            if segment_mask.sum() > 0.85 * centerline_px:
                return segment_mask, True
    if num_segments == 1:
        # it's not a loop/lariat because that would be caught above, and there's no reason
        # to try to connect it because there's just one, so return it now
        return centerlines, False
    # find minimum-cost paths from arbitrary endpoint to all others
    spanning_tree = numpy.zeros_like(centerlines)
    mcp = graph.MCP_Geometric(costs)
    if len(ep_indices) > 1:
        mcp.find_costs(starts=[ep_indices[0]], ends=ep_indices[1:])
        for end in ep_indices[1:]:
            path = mcp.traceback(end)
            spanning_tree[tuple(numpy.transpose(path))] = 1
    # The spanning tree could actually have become a loop or lariat structure
    # so check how many endpoints we have
    endpoints = worm_spline.get_endpoints(spanning_tree)
    return spanning_tree, endpoints.sum() < 2

def get_costs(dv_coords):
    """Convert a dv_coord image (not masked) to a costs array suitable for
    pathfinding through the array with the Minimum Cost Path methods from skimage."""
    # Strategy: look for minima of the second derivatives of a well-filtered
    # dv coordinate image. These are regions along the crest of the centerline.
    dx, dy = numpy.gradient(ndimage.gaussian_filter(dv_coords, 3))
    dxx = numpy.gradient(dx, axis=0)
    dyy = numpy.gradient(dy, axis=1)
    # Combine dxx and dyy to get total centerline "magnitude" -- but first
    # zero out any regions where the derivatives are positive since those aren't
    # the centerline areas.
    dxx[dxx>0] = 0
    dyy[dyy>0] = 0
    costs = numpy.hypot(dxx, dyy)
    # Now invert the "centerline magnitude" so that more centerline-y means lower-valued
    # and raise to the third power to make the valley steeper
    costs = costs.max() - costs
    return costs**3

def orient_path_ascending(path, ap_coords):
    """Orient the path coordinates such that the overall direction is ascending
    according to the ap_coords image."""
    coords = ap_coords[tuple(path.T)]
    if numpy.sign(coords[1:] - coords[:-1]).sum() < 0: # path is mostly descending
        path = path[::-1]
    return path

def find_centerline_from_loop(loop, ap_coords):
    """Given a loop or lariat structure, do as good a job as possible in finding
    a plausible start/end point for the centerline, then return path coordinates
    along that line."""
    # Basic strategy: assume that the worm nose/tail comes together at some point
    # on the loop. Assume that this point is midway between the lowest and highest
    # value of the ap_coords array along that loop. Then break the loop at that
    # point, and find the shortest path from the nose to tail.
    # This doesn't deal wonderfully with lariat structures, but that's a really hard
    # problem to solve correctly unless the ap coordinate image is really reliable
    # which it currently is not...
    masked_coords = ap_coords.copy()
    masked_coords[~loop] = numpy.nan
    low = numpy.unravel_index(numpy.nanargmin(masked_coords), masked_coords.shape)
    high = numpy.unravel_index(numpy.nanargmax(masked_coords), masked_coords.shape)
    costs = numpy.array(loop, dtype=numpy.float32)
    costs[~loop] = numpy.inf
    path, cost = graph.route_through_array(costs, low, high, geometric=False)
    midpoint = len(path)//2
    costs[tuple(numpy.transpose(path[midpoint-1:midpoint+2]))] = numpy.inf # break the loop
    path, cost = graph.route_through_array(costs, path[midpoint-2], path[midpoint+2], geometric=False)
    return numpy.array(path)


def fit_splines(center_path, mask, dv_coords, width_range=(50,100), width_step=1,
        center_smoothing=0.1, width_smoothing=0.0001):
    """Find a (center_tck, width_tck) pose from the centerline path.

    The main challenge is that the dv coordinates are relative to that worm, so an
    absolute scale must be determined. This is done by trying various absolute
    scale factors to find the pixel width of the worm as a function of the dv
    coordinate value along the centerline, and choosing that which best recapitulates
    the mask image provided.

    Parameters:
        center_path: (n, 2)-shape array of coordinates along the centerline of a worm
        mask: approximate mask of worm outline
        dv_coords: dv coordinate image
        width_range: range of possible scale factors to try for the worm (though note that the
            width_tck values are in terms of half-widths from the centerline to the edge)
        width_step: step size for evaluating possible worm widths.
        center_smoothing: average distance the center_tck spline is allowed
            to deviate from the input path coordinates.
        width_smoothing: average distance the width_tck spline is allowed to deviate
            from the input dv coordinate values.

    Returns: (center_tck, width_tck) pose tuple
    """
    center_smoothing *= len(center_path)
    center_tck = interpolate.fit_spline(center_path, smoothing=center_smoothing, force_endpoints=False)
    width_profile = dv_coords[tuple(center_path.T)] / 6 # scale widths in [0, 0.5] range
    # This range is so that when the width_profile is multiplied by the total worm width, the
    # resulting width_tck will produce the expected half-width distances from centerline to edge
    x = numpy.linspace(0, 1, len(width_profile))
    width_smoothing *= len(width_profile)
    width_profile_tck = interpolate.fit_nonparametric_spline(x, width_profile, smoothing=width_smoothing)
    # do coarse / fine search for best width multiplier
    width_tck, max_width = _fit_widths_to_mask(center_tck, width_profile_tck, width_range, width_step*4, mask)
    width_range = max_width - 8*width_step, max_width + 8*width_step
    width_tck, max_width = _fit_widths_to_mask(center_tck, width_profile_tck, width_range, width_step, mask)
    return center_tck, width_tck

def _fit_widths_to_mask(center_tck, width_profile_tck, width_range, width_step, mask):
    """Search a range of width multipliers to find the one that generates a mask with the
    best IOU value with respect to the original mask"""
    low, high = width_range
    num_steps = int((high - low) / width_step + 1)
    max_widths = numpy.linspace(low, high, num_steps)
    t, c, k = width_profile_tck
    width_tcks = [(t, c*max_width, k) for max_width in max_widths]
    test_masks = [worm_spline.worm_coords_lab_frame_mask(mask.shape, center_tck, width_tck) for width_tck in width_tcks]
    best_i = numpy.argmax([iou(mask, test_mask) for test_mask in test_masks])
    return width_tcks[best_i], max_widths[best_i]

def iou(m1, m2):
    """Return intersection over union score for similarity of two masks"""
    return (m1 & m2).sum() / (m1 | m2).sum()
