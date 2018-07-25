import numpy as np
import pickle

from scipy import ndimage

from zplib import pca
from zplib.image import pyramid
from zplib.image import colorize
from zplib.curve import interpolate

from skimage import graph
from elegant import worm_spline

#TODO: Make this part not hard-coded

"""mean, pcs, norm_pcs, variances, total_variance,positions, norm_positions = pickle.load(open('/mnt/lugia_array/data/human_masks/warped_splines/pca_stuff.p', 'rb'))
avg_width_positions = pca.pca_reconstruct([0,0,0], pcs[:3], mean)
avg_width_tck = interpolate.fit_nonparametric_spline(np.linspace(0,1, len(avg_width_positions)), avg_width_positions)"""

def edge_detection(image, center_tck, width_tck, avg_width_tck):
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
    image = scale_image(image)
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


def edge_coordinates(image, avg_width_tck):
    """From an image of a straightened worm, find the edges of the worm.
    It is assumed that the centerline of the worm is in the center of the image
    (NOTE: this is the way that elegant generates the straightened worm panel in the gui)

    Parameters:
        image: ndarray of the straightened worm image
        avg_width_tck: width spline defining the average distance from the centerline
            to the worm edges (This is taken from the pca things we did earlier)

    Returns:
        center_coordinates: shape (num_coords, 2) list of coordinates that define the centerline
            in the worm frame of reference
        width_coordinates: shape (num_coords, 2) list of coordinates that define the widths
            in the worm fram of reference
    """
    #break up the image into top and bottom parts to find the widths on each side
    top_image = np.flip(image[:,:int(image.shape[1]/2)], axis =1)
    bottom_image = image[:,int(image.shape[1]/2):]
    xt, top_widths = find_edges(top_image, avg_width_tck)
    xb, bottom_widths = find_edges(bottom_image, avg_width_tck)

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
    gradient = ndimage.filters.gaussian_gradient_magnitude(image_down, 1)
    top_ten = np.percentile(gradient,  61)
    gradient = sigmoid(gradient, gradient.min(), top_ten, gradient.max(), 2)
    gradient = gradient.max()-abs(gradient)

    #penalize finding edges near the centerline or outside of the avg_width_tck
    #since the typical worm is fatter than the centerline and not huge
    #Need to divide by 2 because of the downsampling
    pen_widths = (interpolate.spline_interpolate(avg_width_tck, image_down.shape[0]))
    #pen_widths = pen_widths/2
    distance_matrix = abs(np.subtract.outer(pen_widths, np.arange(0, image_down.shape[1])))
    #distance_matrix = np.flip(abs(np.subtract.outer(pen_widths, np.arange(0, image_down.shape[1]))), 1)
    penalty = alpha*(distance_matrix)
    new_costs = gradient+penalty
    
    #set start and end points for the traceback
    start = (0, int(pen_widths[0]))
    end = (len(pen_widths)-1, int(pen_widths[-1]))
    #start = (0, int((image_down.shape[1]-1)-pen_widths[0]))
    #end = (len(pen_widths)-1, int((image_down.shape[1]-1)-pen_widths[-1]))
    #start = (0,0)
    #end = (len(pen_widths)-1, 0)

    #begin edge detection
    offsets= [(1,-1),(1,0),(1,1)]
    mcp = Smooth_MCP(new_costs, mcp_alpha, offsets=offsets)
    mcp.find_costs([start], [end])
    route = mcp.traceback(end)

    x,y = np.transpose(route)
    #print("x: ", len(x), "y:", len(y))
    #print(route)
    #y = image_down.shape[1] - y
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
    '''
    return min+((max-min)/(1+np.exp(-(growth_rate)*(gradient-mid))))

def scale_image(image):
    mode = np.bincount(image.flat)[1:].argmax()+1
    bf = image.astype(np.float32)
    bf -= 200
    bf *= (24000-200) / (mode-200)
    bf8 = colorize.scale(bf, min=600, max=26000, gamma=0.72, output_max=255).astype(np.float32)
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