import freeimage
import pathlib
import pickle
import pkg_resources
from PyQt5 import Qt
import numpy
from functools import partial

from ris_widget import ris_widget
from ris_widget import shared_resources

from elegant.gui import pose_annotation
from elegant import worm_widths
from elegant import worm_spline

from zplib.image import colorize

def save_annotations(rw):
    """Save the pose annotations as pickle files into the parent directory.
    A pickle file is created for each page in the flipbook with the name of the first image in the 
    flipbook_page list as the base for the pickle file name.
    """
    for fp in rw.flipbook_pages:
        annotations = fp.annotations
        path = pathlib.Path(fp[0].name)
        save_path = path.parent/(path.name.split(".")[0]+".pickle") 
        pickle.dump(annotations, open(save_path, 'wb'))

def load_metadata(rw):
    """Loads annotation metadata from a pickle file for each flipbook page.
    NOTE: the metadata is only loaded if there is no current pose annotations.
    """
    for fp in rw.flipbook_pages:
        center_tck, width_tck = fp.annotations['pose']
        if center_tck is None and width_tck is None:
            path = pathlib.Path(fp[0].name)
            img_name = path.name.split('.')[0]
            glob = path.parent.glob(img_name+('.pickle'))
            
            for ann_path in glob:
                annotations = pickle.load(open(ann_path, 'rb'))
                fp.annotations = annotations

    rw.annotator.update_fields()

def save_warped_images(rw):
    """Save the warped images into the same parent directory as the images in the flipbook.
    If there is a width tck, the saved images have a transparent channel with the worm mask.
    The warped images will be saved with the same base name as the first image in the flipbook_page list
    with "-straightening.png" afterwards.

    Example:
        image name: 'test.png'
        straightened image name: 'test-straightened.png'
    """
    for fp in rw.flipbook_pages:
        lab_frame_image = fp[0].data
        center_tck, width_tck = fp.annotations['pose']
        if center_tck is not None:
            warp = worm_spline.to_worm_frame(lab_frame_image, center_tck, width_tck)
            warp = colorize.scale(warp).astype(numpy.uint8)
            if width_tck is not None:
                worm_frame_mask = worm_spline.worm_frame_mask(width_tck, warp.shape)
                image = numpy.dstack((warp, warp, warp, worm_frame_mask))
            else:
                image = numpy.dstack((warp, warp, warp))
            path = pathlib.Path(fp[0].name)
            save_path = path.parent/(path.name.split(".")[0]+"-straightening.png")
            freeimage.write(image, save_path)

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="zplab image viewer")
    parser.add_argument('images', nargs="*", metavar='image', help='image files to open')
    
    args = parser.parse_args(argv)
    
    rw = ris_widget.RisWidget()
    width_estimator = worm_widths.WidthEstimator.from_default_widths(pixels_per_micron=1/1.3)
    pa = pose_annotation.PoseAnnotation(rw, width_estimator=width_estimator)
    rw.add_annotator([pa])

    load_data = Qt.QPushButton('Load')
    load_data.clicked.connect(partial(load_metadata,rw))
    save_data = Qt.QPushButton('Poses')
    save_data.clicked.connect(partial(save_annotations,rw))
    save_warps = Qt.QPushButton('Images')
    save_warps.clicked.connect(partial(save_warped_images,rw))
    pa._add_row(pa.widget.layout(), Qt.QLabel('Data:'),load_data, save_data, save_warps)    
     
    rw.add_image_files_to_flipbook(args.images)
    shared_resources._QAPPLICATION.exec()

if __name__ == '__main__':
    main()