import celiagg
import numpy
from zplib.curve import spline_geometry
from zplib.curve import interpolate

def image_frame_mask(center_tck, width_tck, image_shape, num_spline_points=None, antialias=False):
    path = celiagg.Path()
    path.lines(spline_geometry.outline(center_tck, width_tck, num_points=num_spline_points)[-1])
    return _celiagg_draw_mask(image_shape, path, antialias)

def worm_frame_mask(width_tck, image_shape, num_spline_points=None, antialias=False):
    worm_length = image_shape[0]
    if num_spline_points is None:
        num_spline_points = worm_length
    widths = interpolate.spline_interpolate(width_tck, num_points=num_spline_points)
    x_vals = numpy.linspace(0, worm_length, num_spline_points)
    centerline_y = image_shape[1] / 2
    top = numpy.transpose([x_vals, centerline_y - widths])
    bottom = numpy.transpose([x_vals, centerline_y + widths])[::-1]
    path = celiagg.Path()
    path.lines(top)
    path.lines(bottom)
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
