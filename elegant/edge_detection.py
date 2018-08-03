import numpy as np
import pickle

from scipy import ndimage

from zplib import pca
from zplib.image import pyramid
from zplib.image import colorize
from zplib.curve import interpolate
import celiagg
from ris_widget import histogram_mask

from skimage import graph
from elegant import worm_spline

def edge_detection(image, center_tck, width_tck, avg_width_tck, objective=5):
    """Main function to detect the edges of the worm. Returns a new center_tck
    and width_tck in the lab frame of reference

    Parameters:
        image: ndarray of the brightfield image
        center_tck: centerline spline defining the pose of the worm in the lab
            frame.
        width_tck: width spline defining the distance from centerline to worm
            edges.
        avg_width_tck: width spline defining the average distance from the centerline
            to the worm edges (This is taken from the pca things we did earlier)

    Returns:
        new_center_tck: centerline spline defining the new pose of the worm in the lab
            frame.
        new_width_tck: width spline defining the distance from centerline to worm
            edges.
    """
    #normalize image
    image = scale_image(image, objective=objective)
    warped_image = worm_spline.to_worm_frame(image, center_tck, width_tck=width_tck)
    center_coordinates, width_coordinates = edge_coordinates(warped_image, avg_width_tck)
    new_center_tck, new_width_tck = new_tcks(center_coordinates, width_coordinates, warped_image.shape, center_tck)

    return(new_center_tck, new_width_tck)

def new_tcks(center_coordinates, width_coordinates, worm_image_shape, center_tck):
    """From the coordinates in the worm frame generate new center and width tcks 
    in the lab frame.

    Parameters:
        center_coordinates: shape (num_coords, 2) list of coordinates that define the centerline
            in the worm frame of reference
        width_coordinates: shape (num_coords, 2) list of coordinates that define the widths
            in the worm frame of reference
        worm_image_shape: shape of worm image in which the coordinates are defined
        center_tck: centerline spline defining the pose of the worm in the lab
            frame.
    """
    #get new coordinates for the centerline in the lab frame
    new_center_coordinates = worm_spline.coordinates_to_lab_frame(center_coordinates, worm_image_shape, center_tck)
    #generate new splines
    _,widths = width_coordinates.T
    new_center_tck = interpolate.fit_spline(new_center_coordinates, smoothing=1*len(new_center_coordinates))
    x = np.linspace(0,1, len(widths))
    new_width_tck = interpolate.fit_nonparametric_spline(x, widths, smoothing=1*len(new_center_coordinates))

    return(new_center_tck, new_width_tck)


def edge_coordinates(image, avg_width_tck, objective=5):
    """From an image of a straightened worm, find the edges of the worm.
    It is assumed that the centerline of the worm is in the center of the image
    (NOTE: this is the way that elegant generates the straightened worm panel in the gui)

    Parameters:
        image: ndarray of the straightened worm image
        avg_width_tck: width spline defining the average distance from the centerline
            to the worm edges (This is taken from the pca things we did earlier)
        mag: string indicating whether the images are from 10x corrals or 5x corrals. This is to
            ensure the right parameters are used in edge finding 

    Returns:
        center_coordinates: shape (num_coords, 2) list of coordinates that define the centerline
            in the worm frame of reference
        width_coordinates: shape (num_coords, 2) list of coordinates that define the widths
            in the worm frame of reference
    """
    #dictionary of the parameters from the optimizer
    params = {10: dict(ggm_sigma=1, sig_per=56.5, sig_growth_rate=5, alpha=2, mcp_alpha=3), 5: dict(ggm_sigma=1, sig_per=61, sig_growth_rate=2, alpha=1, mcp_alpha=1)}
    #break up the image into top and bottom parts to find the widths on each side
    top_image = np.flip(image[:,:int(image.shape[1]/2)], axis =1)
    bottom_image = image[:,int(image.shape[1]/2):]

    xt, top_widths = find_edges(top_image, avg_width_tck, **params[objective])
    xb, bottom_widths = find_edges(bottom_image, avg_width_tck, **params[objective])

    top_widths = top_widths.astype(np.float)
    bottom_widths = bottom_widths.astype(np.float)

    #need to account for the top and bottom splitting sincethe centerline is exactly 
    #in the middle of the pixels, we need to add 0.5 to the widths
    if (image.shape[1]/2) % 2 == 0:
        top_widths += 0.5
        bottom_widths += 0.5

    #NOTE: using the midpoint as the zero axis, we negate the bottom widths (hence, it is
    #negative here) ie. it becomes (top_widths + (-bottom_widths))
    new_center = (bottom_widths-top_widths)/2 #find the midpoint between the two points
    #put the pixels in the same frame of reference as straightened worm
    new_center += int(image.shape[1]/2)
    new_widths = (top_widths+bottom_widths)/2#calculate the average widths

    center_coordinates = np.transpose([xt,new_center])
    width_coordinates = np.transpose([xt,new_widths])

    return (center_coordinates, width_coordinates)


def find_edges(image, avg_width_tck, ggm_sigma=1, sig_per=61, sig_growth_rate=2, alpha=1, mcp_alpha=1):
    """Find the edges of one side of the worm and return the x,y positions of the new widths
    NOTE: This function assumes that the image is only half of the worm (ie. from the centerline
    to the edges of the worm)

    Parameters:
        image: ndarray of the straightened worm image (typically either top or bottom half)
        avg_width_tck: width spline defining the average distance from the centerline
            to the worm edges (This is taken from the pca things we did earlier)
        ggm_sigma, sig_per, sig_growth_rate, alpha, mcp_alpha: hyperparameters for 
            the edge-detection scheme
    
    Returns:
        route: tuple of x,y positions of the identfied edges
    """

    #down sample the image
    image_down = pyramid.pyr_down(image, downscale=2)

    #get the gradient
    gradient = ndimage.filters.gaussian_gradient_magnitude(image_down, ggm_sigma)
    top_ten = np.percentile(gradient,  sig_per)
    gradient = sigmoid(gradient, gradient.min(), top_ten, gradient.max(), sig_growth_rate)
    gradient = gradient.max()-abs(gradient)

    #penalize finding edges near the centerline or outside of the avg_width_tck
    #since the typical worm is fatter than the centerline and not huge
    #Need to divide by 2 because of the downsampling
    pen_widths = (interpolate.spline_interpolate(avg_width_tck, image_down.shape[0]))
    distance_matrix = abs(np.subtract.outer(pen_widths, np.arange(0, image_down.shape[1])))
    penalty = alpha*(distance_matrix)
    new_costs = gradient+penalty
    
    #set start and end points for the traceback
    start = (0, int(pen_widths[0]))
    end = (len(pen_widths)-1, int(pen_widths[-1]))

    #begin edge detection
    offsets= [(1,-1),(1,0),(1,1)]
    mcp = Smooth_MCP(new_costs, mcp_alpha, offsets=offsets)
    mcp.find_costs([start], [end])
    route = mcp.traceback(end)

    x,y = np.transpose(route)
    return (x*2, y*2) #multiply by 2 to account for downsampling

def sigmoid(gradient, min, mid, max, growth_rate):
    '''Apply the sigmoid function to the gradient.

    Parameters:
        gradient: array of the gradient of the image
        min: lower asymptote of the sigmoid function
        mid: midpoint of the sigmoid function (ie. the point that is halfway between
            the lower and upper asymptotes)
        max: upper asymptote
        growth_rate: growth rate of the sigmoid function

    Returns:
        result from the sigmoid function
    '''
    return min+((max-min)/(1+np.exp(-(growth_rate)*(gradient-mid))))

def circle_mask(cx, cy, r, shape):
    """Helper functions for the 10x vignettes
    """
    cx, cy, r = int(cx * shape[0]), int(cy * shape[1]), int(r * shape[0])
    path = celiagg.Path()
    path.ellipse(cx, cy, r, r)
    return worm_spline._celiagg_draw_mask(shape, path, antialias=False)

def tenX_mask(img_shape):
    """Helper function to generate the 10x vignette
    """
    return circle_mask(*histogram_mask.HistogramMask.DEFAULT_MASKS[0.7], shape=img_shape)

def scale_image(image, objective=5):
    """Scale images based on the mode

    Parameters:
        image: ndarray of the image
        mag: string indicating whether the images are from 10x corrals
            or not. This is because the 10x images have different vignetting
            and parameters to make the images look better

    Returns:
        bf8: uint8 ndarray of the normalized image
    """
    if objective==10:
        mask = tenX_mask(image.shape).astype(bool)
        pixels = image[mask]
        gamma = 1
    else:
        pixels = image.flat
        gamma = 0.72

    mode = np.bincount(pixels)[1:].argmax()+1
    bf = image.astype(np.float32)
    bf -= 200
    bf *= (24000-200) / (mode-200)
    bf8 = colorize.scale(bf, min=600, max=26000, gamma=gamma, output_max=255)
            
    return bf8

class Smooth_MCP(graph.MCP_Flexible):
    """Custom MCP class to weight different route possibilities.
    Penalize sharp changes to make the end widths a little smoother
    """
    def __init__(self, costs, alpha, offsets=None):
        graph.MCP_Flexible.__init__(self, costs, offsets=offsets)
        self.alpha = alpha

    def travel_cost(self, old_cost, new_cost, offset_length):
        """Override method to smooth out the traceback
        """

        return self.alpha*(new_cost + abs(offset_length)) 