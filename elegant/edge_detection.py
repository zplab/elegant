# This code is licensed under the MIT License (see LICENSE file for details)

import numpy
from scipy import ndimage
from skimage import graph

from zplib.image import pyramid
from zplib.image import colorize
from zplib.curve import interpolate

from . import worm_spline
from . import process_images

# dictionary of the parameters found by optimization
OBJ_PARAMS = {
    10: dict(image_gamma=1, downscale=2, gradient_sigma=1, sigmoid_midpoint=55,
        sigmoid_growth_rate=1280, edge_weight=20, roughness_penalty=3, post_smoothing=2),
    5: dict(image_gamma=0.72, downscale=2, gradient_sigma=1, sigmoid_midpoint=60,
        sigmoid_growth_rate=512, edge_weight=40, roughness_penalty=1, post_smoothing=2)
}

def detect_edges(image, center_tck, width_tck, avg_width_tck, objective=5, optocoupler=1):
    """Trace the edges of a worm and return a new center_tck and width_tck.

    Parameters:
        image: ndarray of the brightfield image
        center_tck: spline defining the pose of the worm.
        width_tck: spline defining the distance from centerline to worm edges.
        avg_width_tck: width spline for an average worm (of the given age, etc.)
        objective: objective magnification (to look up correct parameters with)
        optocoupler: optocoupler magnification (to correctly calculate the image
            vignette)

    Returns: (new_center_tck, new_width_tck)
        new_center_tck: new centerline spline
        new_width_tck: new width spline
    """
    cost_image, new_center_tck, new_width_tck = _detect_edges(image, optocoupler,
        center_tck, width_tck, avg_width_tck, **OBJ_PARAMS[objective])
    return new_center_tck, new_width_tck

def _detect_edges(image, optocoupler, center_tck, width_tck, avg_width_tck,
        image_gamma, downscale, gradient_sigma, sigmoid_midpoint,
        sigmoid_growth_rate, edge_weight, roughness_penalty, post_smoothing):
    """Trace the edges of a worm and return a new center_tck and width_tck.

    Parameters:
        image: ndarray of the brightfield image
        optocoupler: optocoupler magnification (to correctly calculate the image
            vignette)
        center_tck: spline defining the pose of the worm.
        width_tck: spline defining the distance from centerline to worm edges.
        avg_width_tck: width spline for an average worm (of the given age, etc.)
        image_gamma: gamma value for intensity transform to highlight worm edges
        downscale: factor by which to downsample the image
        gradient_sigma: sigma for gaussian gradient to find worm edges
        sigmoid_midpoint: midpoint of edge_highlighting sigmoid function for
            gradient values, expressed as a percentile of the gradient value
            over the whole image.
        sigmoid_growth_rate: steepness of the sigmoid function.
        edge_weight: how much to weight image edge strength vs. distance from
            the average widths in the cost function.
        roughness_penalty: how much to penalize diagonal steps vs. straight
            steps in the edge tracing (to penalize jagged edges)
        post_smoothing: spline smoothing factor for re-fit centerline.

    Returns: (cost_image, new_center_tck, new_width_tck)
        cost_image: image defining the cost function for edge tracing
        new_center_tck: new centerline spline
        new_width_tck: new width spline
    """
    cost_image = get_cost_image(image, optocoupler, image_gamma, center_tck,
        width_tck, avg_width_tck, downscale, gradient_sigma, sigmoid_midpoint,
        sigmoid_growth_rate, edge_weight)

    # trace edges to calculate new centerline and widths
    center_coordinates, widths = edge_coordinates(cost_image, roughness_penalty)
    new_center_coordinates = worm_spline.coordinates_to_lab_frame(center_coordinates, cost_image.shape, center_tck, zoom=1/downscale)
    #generate new splines
    new_center_tck = interpolate.fit_spline(new_center_coordinates, smoothing=post_smoothing*len(new_center_coordinates))
    x = numpy.linspace(0, 1, len(widths))
    # don't forget to expand widths to account for downsampling
    new_width_tck = interpolate.fit_nonparametric_spline(x, widths*downscale, smoothing=len(widths))
    return cost_image, new_center_tck, new_width_tck

def get_cost_image(image, optocoupler, image_gamma, center_tck, width_tck, avg_width_tck,
        downscale, gradient_sigma, sigmoid_midpoint, sigmoid_growth_rate, edge_weight):
    """Trace the edges of a worm and return a new center_tck and width_tck.

    Parameters:
        image: ndarray of the brightfield image
        optocoupler: optocoupler magnification (to correctly calculate the image
            vignette)
        center_tck: spline defining the pose of the worm.
        width_tck: spline defining the distance from centerline to worm edges.
        avg_width_tck: width spline for an average worm (of the given age, etc.)
        image_gamma: gamma value for intensity transform to highlight worm edges
        downscale: factor by which to downsample the image
        gradient_sigma: sigma for gaussian gradient to find worm edges
        sigmoid_midpoint: midpoint of edge_highlighting sigmoid function for
            gradient values, expressed as a percentile of the gradient value
            over the whole image.
        sigmoid_growth_rate: steepness of the sigmoid function.
        edge_weight: how much to weight image edge strength vs. distance from
            the average widths in the cost function.

    Returns: image defining the cost function for edge tracing
    """
    # normalize, warp, and downsample image
    image = process_images.pin_image_mode(image, optocoupler=optocoupler)
    image = colorize.scale(image, min=600, max=26000, gamma=image_gamma, output_max=1)
    warped_image = worm_spline.to_worm_frame(image, center_tck, width_tck)
    small_warped = pyramid.pyr_down(warped_image, downscale=downscale)

    # calculate the edge costs
    gradient = ndimage.gaussian_gradient_magnitude(small_warped, gradient_sigma)
    gradient = sigmoid(gradient, numpy.percentile(gradient, sigmoid_midpoint), sigmoid_growth_rate)
    gradient = gradient.max() - abs(gradient)

    # penalize finding edges away from the average width along the worm
    average_widths = (interpolate.spline_interpolate(avg_width_tck, small_warped.shape[0])) / downscale
    distance_from_average = abs(numpy.subtract.outer(average_widths, numpy.arange(0, small_warped.shape[1])))
    return edge_weight * gradient + distance_from_average

def sigmoid(x, x0, k):
    """Sigmoid function (logistic): https://en.wikipedia.org/wiki/Logistic_function
    Returns result in range [0, 1]
    """
    return 1 / (1 + numpy.exp(-k * (x - x0)))

def edge_coordinates(cost_image, roughness_penalty):
    """Trace through a cost image to find the lowest-cost worm edges.

    It is assumed that the centerline of the worm is approximately along the
    center of the cost image.

    Parameters:
        cost_image: image defining the cost function for tracing edges.
        roughness_penalty: how much to penalize diagonal steps vs. straight
            steps in the edge tracing (to penalize jagged edges).

    Returns: (center_coordinates, new_widths)
        center_coordinates: shape (cost_image.shape[0], 2) array of coordinates
            defining the new centerline in the cost image.
        new_widths: array (of length cost_image.shape[0]) defining the widths
            at each point along the centerline.
    """
    # break up the image into top and bottom parts to find the widths on each side
    centerline_index = (cost_image.shape[1] - 1) / 2 # may fall at a non-integer pixel
    # the below makes sure that odd-height images get the centerline included on
    # both the top and bottom half
    top_image = cost_image[:, int(numpy.floor(centerline_index))::-1] # flip upside-down so centerline is at top
    bottom_image = cost_image[:, int(numpy.ceil(centerline_index)):]

    x, top_widths = _trace_costs(top_image, roughness_penalty)
    x, bottom_widths = _trace_costs(bottom_image, roughness_penalty)

    # if centerline was halfway between pixels, then we need to add 0.5 to the resulting widths
    if int(centerline_index) != centerline_index:
        top_widths += 0.5
        bottom_widths += 0.5

    # offset centerline by difference between bottom and top...
    new_centerline = centerline_index + (bottom_widths - top_widths) / 2
    # ... which means equal widths on each side
    new_widths = (top_widths + bottom_widths) / 2
    center_coordinates = numpy.transpose([x, new_centerline])
    return center_coordinates, new_widths

def _trace_costs(cost_image, roughness_penalty):
    """Trace through a cost image to find the lowest-cost worm edge.

    The input image should correspond to HALF of a worm, with the presumptive
    centerline running along the top of the image (y=0).

    Parameters:
        cost_image: image defining the cost function for tracing edges.
        roughness_penalty: how much to penalize diagonal steps vs. straight
            steps in the edge tracing (to penalize jagged edges).

    Returns: x and y positions of the identfied edges with shape (2, cost_image.shape[0])
    """
    # set start and end points for the traceback
    # we have multiple start sites in case the head does not start at zero width
    starts = [(0,i) for i in range(6)]
    ends = [(cost_image.shape[0]-1, 0)] # assume the tail ends at zero width though

    # begin edge detection
    offsets = [(1,-1), (1,0), (1,1)] # allow straight forward or up/down diagonal moves
    mcp = _SmoothMCP(cost_image, roughness_penalty, offsets=offsets)
    mcp.find_costs(starts, ends)
    route = mcp.traceback(ends[0])
    return numpy.transpose(route).astype(float)

class _SmoothMCP(graph.MCP_Flexible):
    """Custom MCP class to weight different route possibilities.
    Penalize diagonal steps to make resulting path smoother
    """
    def __init__(self, costs, roughness_penalty, offsets):
        graph.MCP_Flexible.__init__(self, costs, offsets=offsets)
        self.roughness_penalty = roughness_penalty

    def travel_cost(self, old_cost, new_cost, offset_length):
        # Make longer (i.e. more diagonal) steps cost more
        return self.roughness_penalty*(new_cost + abs(offset_length))