import numpy
import scipy.interpolate
from scipy import ndimage
from scipy import spatial

from skimage import morphology
from skimage import graph

import celiagg
from zplib.curve import spline_geometry
from zplib.curve import interpolate


def pose_from_mask(mask):
    """Calculate worm pose splines from mask image.

    Parameter:
        mask: a binary mask image with a single object

    Returns: center_tck, width_tck
        splines defining the (x, y) position of the centerline and the distance
        from the centerline to the edge (called "widths" but more like "half-
        widths" more accurately) as a function of position along the centerline.
        If there was no object in the mask, returns None for each.
    """
    mask = mask > 0
    # crop image for faster medial axis transform
    sx, sy = ndimage.find_objects(mask)[0]
    cropped = mask[sx, sy]
    centerline, widths = _get_centerline(cropped)
    if len(centerline) == 0:
        return None, None
    center_tck, width_tck = _get_splines(centerline, widths)
    # adjust x, y coords to account for the cropping
    c = center_tck[1]
    c += sx.start, sy.start
    return center_tck, width_tck


def _get_centerline(mask):
    # Strategy: use the medial axis transform to get a skeleton of the mask,
    # then find the endpoints with a binary hit-or-miss operation. Next,
    # use MCP to trace from each endpoint to each other, in order to find the
    # most-distant pair, which will be the centerline. Using the distance matrix
    # provided by the medial axis transform, compute how "wide" the worm is at
    # each centerline point ("width" is actually a half-width...)
    skeleton, distances = morphology.medial_axis(mask, return_distance=True)
    # 1-values in structure1 are where we require a True in the input if the transform is to be True
    # 1-values in structure2 are where we requre a False in the input if the transform is to be True
    # so: structure1 requires a point in the middle of a 3x3 neighborhoood be True
    # and structure2 requires that it be an endpoint, i.e. only have one axial neighbor
    structure1 = numpy.array([[0,0,0],[0,1,0],[0,0,0]])
    structure2 = numpy.array([[1,0,1],[1,0,1],[1,1,1]])
    # return the union of all places that structure2 matches, when rotated to all four orientations
    endpoints = _rotated_hit_or_miss(skeleton, structure1, structure2)
    # now modify structure2 to signal when there is exactly one diagonal neighbor
    structure2 = numpy.array([[1,1,1],[1,0,1],[1,1,0]])
    endpoints |= _rotated_hit_or_miss(skeleton, structure1, structure2)
    ep_indices = numpy.transpose(endpoints.nonzero())

    skeleton = skeleton.astype(float)
    skeleton[skeleton == 0] = numpy.inf
    mcp = graph.MCP(skeleton)
    longest_traceback = []
    #compute costs for every endpoint pair
    for i, endpoint in enumerate(ep_indices[:-1]):
        remaining_indices = ep_indices[i+1:]
        costs = mcp.find_costs([endpoint], remaining_indices)[0]
        path_lengths = costs[list(remaining_indices.T)]
        most_distant = remaining_indices[path_lengths.argmax()]
        traceback = mcp.traceback(most_distant)
        if len(traceback) > len(longest_traceback):
            longest_traceback = traceback

    centerline = numpy.asarray(longest_traceback)
    widths = distances[list(centerline.T)]
    return centerline, widths


def _rotated_hit_or_miss(mask, structure1, structure2):
    # perform a binary hit-or-miss with structure2 rotated in all four orientations,
    # or-ing together the output.
    output = ndimage.binary_hit_or_miss(mask, structure1, structure2)
    for i in range(3):
        structure2 = structure2[::-1].T # rotate 90 degrees
        output |= ndimage.binary_hit_or_miss(mask, structure1, structure2)
    return output


def _get_splines(centerline, widths):
    # Strategy: extrapolate the first/last few pixels to get the full length of
    # the worm, since medial axis skeleton doesn't go to the edge of the mask
    # To extrapolate, make linear fits from the first and last few points, and
    # then extrapolate out a distance equal to the "width" at that point (which
    # is the distance to the closest edge).

    # create splines for the first points and extrapolate to presumptive mask edge
    begin_tck = interpolate.fit_spline(centerline[:10], smoothing=4, order=1)
    dist_to_edge = widths[0]
    t = numpy.linspace(-dist_to_edge, 0, int(round(dist_to_edge)), endpoint=False)
    begin = interpolate.spline_evaluate(begin_tck, t)
    begin_widths = numpy.linspace(0, dist_to_edge, int(round(dist_to_edge)), endpoint=False)

    # create splines for the last points and extrapolate to presumptive mask edge
    end_tck = interpolate.fit_spline(centerline[-10:], smoothing=4, order=1)
    dist_to_edge = widths[-1]
    t_max = end_tck[0][-1]
    t = numpy.linspace(t_max + dist_to_edge, t_max, int(round(dist_to_edge)), endpoint=False)[::-1]
    end = interpolate.spline_evaluate(end_tck, t)
    end_widths = numpy.linspace(dist_to_edge - 1, 0, int(round(dist_to_edge)))

    center_tck = interpolate.fit_spline(numpy.concatenate([begin, centerline, end]),
        smoothing=0.2*len(centerline))

    new_widths = numpy.concatenate([begin_widths, widths, end_widths])
    x = numpy.linspace(0, 1, len(y))
    width_tck = interpolate.fit_nonparametric_spline(x, new_widths, smoothing=0.2*len(centerline))
    return center_tck, width_tck


def  to_worm_frame(images, center_tck, width_tck=None, width_margin=20, sample_distance=None,
        standard_length=None, standard_width=None, zoom=1, order=3, dtype=None, **kwargs):
    """Transform images from the lab reference frame to the worm reference frame.

    The width of the output image is defined by the center_tck, which defines
    the length of the worm and thus the width of the image. The height of the
    image can be specified either directly by the sample_distance parameter,
    or can be computed from a width_tck that defines the location of the sides
    of the worm (a fixed width_margin is added so that the image extends a bit
    past the worm).

    The size and shape of the output worm can be standardized to a "unit worm"
    by use of the "standard_length" and "standard_width" parameters; see below.

    Parameters:
        images: single numpy array, or list/tuple/3d array of multiple images to
            be transformed.
        center_tck: centerline spline defining the pose of the worm in the lab
            frame.
        width_tck: width spline defining the distance from centerline to worm
            edges. Optional; uses are as follows: (1) if sample_distance is not
            specified, a width_tck must be specified in order to calculate the
            output image height; (2) if standard_width is specified, a width_tck
            must also be specified to define the transform from this worm's
            width profile to the standardized width profile.
        width_margin: if sample_distance is not specified, width_margin is used
            to define the distance (in image pixels) that the output image will
            extend past the edge of the worm (at its widest). If a zoom is
            specified, note that the margin pixels will be zoomed too.
        sample_distance: number of pixels to sample in each direction
            perpendicular to the centerline. The height of the output image is
            int(round(2 * sample_distance * zoom)).
        standard_length: if not specified, the width of the output image is
            int(round(arc_length)*zoom), where arc_length is the path integral
            along center_tck (i.e. the length from beginning to end). If
            standard_length is specified, then the width of the output image is
            int(round(standard_length*zoom)). The full length of the worm will
            be compressed or expanded as necessary to bring it to the specified
            standard_length.
        standard_width: a width spline specifying the "standardized" width
            profile for the output image. If specified, the actual width profile
            must also be provided as width_tck. In this case, the output image
            will be compressed/expanded perpendicular to the centerline as needed
            to make the actual widths conform to the standard width profile.
        zoom: zoom factor, can be any real number > 0.
        order: image interpolation order (0 = nearest neighbor, 1 = linear,
            3 = cubic). Cubic is best, but slowest.
        dtype: if None, use dtype of input images for output. Otherwise, use
            the specified dtype.
        kwargs: additional keyword arguments to pass to ndimage.map_coordinates.

    Returns: single image or list of images (depending on whether the input is a
        single image or list/tuple/3d array).
    """
    assert width_tck is not None or sample_distance is not None
    if standard_width is not None:
        assert width_tck is not None

    if standard_length is None:
        length = spline_geometry.arc_length(center_tck)
    else:
        length = standard_length
    length = int(round(length * zoom))

    if sample_distance is None:
        wtck = standard_width if standard_width is not None else width_tck
        sample_distance = interpolate.spline_interpolate(wtck, num_points=length).max() + width_margin
    width = int(round(2 * sample_distance * zoom))

    # basic plan:
    # get the centerline, then construct perpendiculars to it
    # then define positions along each perpendicular at which to sample the input images.
    # This is the "offset_directions" variable.
    points = interpolate.spline_interpolate(center_tck, length) # shape = (length, 2)
    points = points.T[..., numpy.newaxis] # now points.shape = (2, length, 1)

    perpendiculars = spline_geometry.perpendiculars(center_tck, length).T # shape = (2, length)
    offsets = numpy.linspace(-sample_distance, sample_distance, width) # distances along each perpendicular across the width of the sample swath
    offset_directions = numpy.multiply.outer(perpendiculars, offsets) # shape = (2, length, width)

    # if we are given a width profile to warp to, do so by adjusting the offset_directions
    # value to be longer or less-long than normal based on whether the width at any
    # position is wider or narrower (respectively) than the standard width.
    if standard_width is not None:
        src_widths = interpolate.spline_interpolate(width_tck, num_points=length)
        dst_widths = interpolate.spline_interpolate(standard_width, num_points=length)
        zero_width = dst_widths == 0
        dst_widths[zero_width] = 1 # don't want to divide by zero below
        width_ratios = src_widths / dst_widths # shape = (length,)
        width_ratios[zero_width] = 0 # this will enforce dest width of zero at these points
        # ratios are width_tck / standard_tck. If the worm is wider than the standard width
        # we need to compress it, meaning go farther out for each sample
        offset_directions *= width_ratios[:, numpy.newaxis]
        # Note: shape of width_ratios[:, numpy.newaxis] is (length, 1), which numpy promotes to (1, length, 1)
        # which is compatible with offset_directions's (2, length, width): we multiply both the offsets' x and y
        # coords by the ratio to scale the offset vectors.

    # now the positions in the image to sample from is just the x,y points of the centerline
    # plus the offset directions.
    # points.shape = (2, length, 1), and offset_directions.shape = (2, length, width)
    # where the first axis is x vs. y, the second axis is along the length of the worm
    # and the third is along the width. The output has shape (2, length, width),
    # which fully defines the positions at which to sample the image.
    sample_coordinates = points + offset_directions

    unpack_list = False
    if isinstance(images, numpy.ndarray):
        if images.ndim == 3:
            images = list(images)
        else:
            unpack_list = True
            images = [images]
    worm_frame = [ndimage.map_coordinates(image, sample_coordinates, order=order,
        output=dtype, **kwargs) for image in images]
    if unpack_list:
        worm_frame = worm_frame[0]
    return worm_frame


def longitudinal_warp_spline(t_in, t_out, center_tck, width_tck=None):
    """Transform a worm spline by longitudinally compressing/expanding it.

    Given the positions of a set of landmarks along the length of the worm, and
    a matching set of positions where the landmarks "ought" to be, return a
    splines that are compressed/expanded so that the landmarks appear in the
    correct location.

    Parameters:
        t_in: list / array of positions in the range (0, 1) exclusive, defining
            the relative position of input landmarks. For example, if the vulva
            was measured to be exactly halfway between head and tail, its
            landmark position would be 0.5. Landmarks for head and tail at 0 and
            1 are automatically added: do not include! List must be in sorted
            order and monotonic.
            To convert from a pixel position along a straightened image into a
            value in (0, 1), simply divide the position by the arc length of the
            spline, which can be calculated by:
                zplib.curve.spline_geometry.arc_length(center_tck)
        t_out: list / array matching t_in, defining the positions of those
            landmarks in the output spline.
        center_tck: input centerline spline. Note: the spline must be close to
            the "natural parameterization" for this to work properly. That is,
            the parameter value must be approximately equal to the distance
            along the spline at that parameter. Splines produced from the
            pose annotator have this property; otherwise please use
            zplib.interpolate.reparameterize_spline first.
        width_tck: optional: input width spline. If provided, the widths will
            be warped as well. (Warping the widths is necessary if a standard
            width profile is to be used with to_worm_frame().)

    Returns: center_tck, width_tck
        center_tck: warped centerline tck. Note that the parameterization is
            *not* natural! The spline "accelerates" and "decelerates" along
            the parameter values to squeeze and stretch the worm (respectively)
            along its profile. Running zplib.interpolate.reparameterize_spline
            will undo this.
        width_tck: if input width_tck was specified, a warped width profile;
            otherwise None.
    """
    t_max = center_tck[0][-1]
    num_knots = t_max // 10 # have one control point every ~10 pixels
    t, c, k = interpolate.insert_control_points(center_tck, num_knots)
    t_in = numpy.concatenate([[0], t_in, [1]])
    t_out = numpy.concatenate([[0], t_out, [1]])
    monotonic_interpolator = scipy.interpolate.PchipInterpolator(t_in, t_out)
    new_t = monotonic_interpolator(t/t_max) * t_max
    center_tck = new_t, c, k
    if width_tck is not None:
        t, c, k = interpolate.insert_control_points(width_tck, num_knots // 3)
        new_t = monotonic_interpolator(t)
        width_tck = new_t, c, k
    return center_tck, width_tck


def to_lab_frame(images, lab_image_shape, center_tck, width_tck,
        standard_width=None, worm_zoom=1, order=3, dtype=None, cval=0, **kwargs):
    """Transform images from the worm reference frame to the lab reference frame.

    This is the inverse transform from to_worm_frame. Regions outside of the
    worm mask in the lab frame of reference will be set equal to the 'cval'
    parameter.

    Parameters:
        images: single numpy array, or list/tuple/3d array of multiple images to
            be transformed.
        lab_image_shape: shape of lab-frame image.
        center_tck: spline defining the pose of the worm in the lab frame.
        width_tck: spline defining the distance from centerline to worm edges.
        standard_width: a width spline specifying the "standardized" width
            profile used to generate the worm-frame image(s), if any.
        worm_zoom: zoom factor used to generate the worm-frame image(s).
        order: image interpolation order (0 = nearest neighbor, 1 = linear,
            3 = cubic). Cubic is best, but slowest.
        dtype: if None, use dtype of input images for output. Otherwise, use
            the specified dtype.
        cval: value with which the lab-frame image will be filled outside of the
            worm are. (numpy.nan with dtype=float is a potentially useful
            combination.)
        kwargs: additional keyword arguments to pass to ndimage.map_coordinates.

    Returns: single image or list of images (depending on whether the input is a
        single image or list/tuple/3d array).
    """
    unpack_list = False
    if isinstance(images, numpy.ndarray):
        if images.ndim == 3:
            images = list(images)
        else:
            unpack_list = True
            images = [images]
    shape = images[0].shape
    for image in images[1:]:
        assert image.shape == shape

    oversample = 4 # oversample a bit to get subpixel precision in coordinate locations
    points = interpolate.spline_interpolate(center_tck, num_points=oversample*shape[0]) # shape (n, 2)
    kd = spatial.cKDTree(points)
    mask = lab_frame_mask(center_tck, width_tck, lab_image_shape) > 0
    mask_indices = numpy.transpose(mask.nonzero()) # shape (m, 2) where m is number of in-mask pixels
    distances, indices = kd.query(mask_indices)
    # indices is the index into the centerline points array of the closest centerline point
    # for each nonzero mask pixel.
    # distances is the distance from each nonzero mask pixel to that centerline point
    worm_frame_x = indices / oversample

    nearest_points = points[indices]
    offsets = mask_indices - nearest_points
    perps = spline_geometry.perpendiculars(center_tck, num_points=len(points))
    matching_perps = perps[indices]
    # sign of dot product between perpendiculars and offset vectors (which should
    # basically be either parallel or antiparallel) tells which side of centerline we're on:
    # if sign is negative, we're on the right side of the worm, which has y-values smaller
    # than the centerline in the worm-frame image (i.e. is above the centerline)
    side = numpy.sign((matching_perps * offsets).sum(axis=1))

    if standard_width is not None:
        src_widths = interpolate.spline_interpolate(width_tck, num_points=len(points))
        dst_widths = interpolate.spline_interpolate(standard_width, num_points=len(points))
        zero_width = src_widths == 0
        src_widths[zero_width] = 1 # don't want to divide by zero below
        width_ratios = dst_widths / src_widths # shape = (length,)
        width_ratios[zero_width] = 0 # this will enforce src width of zero at these points
        # ratios are standard_width / width_tck. Where the worm is wider than the standard
        # profile, need to sample less far into the (standardized) image than the widths
        # would have you believe, so reduce the distances accordingly
        distances *= numpy.interp(indices, numpy.arange(len(points)), width_ratios)
    worm_frame_y = shape[1]/2 + side*distances*worm_zoom # shape[1]/2 is the position of the centerline
    sample_coordinates = numpy.array([worm_frame_x, worm_frame_y])

    lab_frame = []
    dtype = kwargs.get('output')
    for image in images:
        lab_frame_image = numpy.empty(lab_image_shape, dtype=image.dtype if dtype is None else dtype)
        lab_frame_image.fill(cval)
        lab_frame_image[mask] = ndimage.map_coordinates(image, sample_coordinates,
            order=order, cval=cval, output=dtype, **kwargs)
        lab_frame.append(lab_frame_image)
    if unpack_list:
        lab_frame = lab_frame[0]
    return lab_frame


def lab_frame_mask(center_tck, width_tck, image_shape, num_spline_points=None, antialias=False):
    """Use a centerline and width spline to draw a worm mask image in the lab frame of reference.

    Parameters:
        center_tck, width_tck: centerline and width splines defining worm pose.
        image_shape: shape of the output mask
        num_spline_points: number of points to evaluate the worm outline along
            (more points = smoother mask). By default, ~1 point/pixel will be
            used, which is more than enough.
        antialias: if False, return a mask with only values 0 and 255. If True,
            edges will be smoothed for better appearance. This is slightly slower,
            and unnecessary when just using the mask to select pixels of interest.

    Returns: mask image with dtype=numpy.uint8 in range [0, 255]. To obtain a
        True/False-valued mask from a uint8 mask (regardless of antialiasing):
            bool_mask = uint8_mask > 255
    """
    path = celiagg.Path()
    path.lines(spline_geometry.outline(center_tck, width_tck, num_points=num_spline_points)[-1])
    return _celiagg_draw_mask(image_shape, path, antialias)


def worm_frame_mask(width_tck, image_shape, num_spline_points=None, antialias=False, zoom=1):
    """Use a centerline and width spline to draw a worm mask image in the worm frame of reference.

    Parameters:
        width_tck: width splines defining worm outline
        image_shape: shape of the output mask
        num_spline_points: number of points to evaluate the worm outline along
            (more points = smoother mask). By default, ~1 point/pixel will be
            used, which is more than enough.
        antialias: if False, return a mask with only values 0 and 255. If True,
            edges will be smoothed for better appearance. This is slightly slower,
            and unnecessary when just using the mask to select pixels of interest.
        zoom: zoom-value to use (for matching output of to_worm_frame with zooming.)

    Returns: mask image with dtype=numpy.uint8 in range [0, 255]. To obtain a
        True/False-valued mask from a uint8 mask (regardless of antialiasing):
            bool_mask = uint8_mask > 255
    """
    worm_length = image_shape[0]
    if num_spline_points is None:
        num_spline_points = worm_length
    widths = interpolate.spline_interpolate(width_tck, num_points=num_spline_points)
    widths *= zoom
    x_vals = numpy.linspace(0, worm_length, num_spline_points)
    centerline_y = image_shape[1] / 2
    top = numpy.transpose([x_vals, centerline_y - widths])
    bottom = numpy.transpose([x_vals, centerline_y + widths])[::-1]
    path = celiagg.Path()
    path.lines(numpy.concatenate([top, bottom]))
    return _celiagg_draw_mask(image_shape, path, antialias)


def _celiagg_draw_mask(image_shape, path, antialias):
    image = numpy.zeros(image_shape, dtype=numpy.uint8, order='F')
    # NB celiagg uses (h, w) C-order convention for image shapes
    canvas = celiagg.CanvasG8(image.T)
    state = celiagg.GraphicsState(drawing_mode=celiagg.DrawingMode.DrawFill, anti_aliased=antialias)
    fill = celiagg.SolidPaint(1,1,1)
    transform = celiagg.Transform()
    canvas.draw_shape(path, transform, state, fill=fill)
    return image
