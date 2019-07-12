import freeimage
import pathlib
import pickle
from PyQt5 import Qt
import numpy
from functools import partial

from ris_widget import ris_widget
from ris_widget import shared_resources

from elegant.gui import pose_annotation
from elegant import worm_widths
from elegant import worm_spline

from zplib.image import colorize

class GeneralPoseAnnotator():
    def __init__(self, rw):
        self.ris_widget = rw

    def save_annotations(self):
        """Save the pose annotations as pickle files into the parent directory.
        A pickle file is created for each page in the flipbook with the name of the first image in the 
        flipbook_page list as the base for the pickle file name.
        """
        for fp in self.ris_widget.flipbook_pages:
            annotations = getattr(fp, 'annotations', None)
            if annotations is not None:
                pose = annotations.get('pose')
                if pose is not None or pose is not (None, None):     
                    path = pathlib.Path(fp[0].name)
                    save_path = path.with_suffix('.pickle')
                    with save_path.open('wb') as f:
                        pickle.dump(annotations, f)

    def load_metadata(self):
        """Loads annotation metadata from a pickle file for each flipbook page.
        NOTE: the metadata is only loaded if there is no current pose annotations.
        """
        for fp in self.ris_widget.flipbook_pages:
            annotations = getattr(fp, 'annotations', None)
            if annotations is not None:    
                path = pathlib.Path(fp[0].name)
                img_name = path.name.split('.')[0]
                
                for ann_path in path.parent.glob(img_name+('.pickle')):
                    annotations = pickle.load(open(ann_path, 'rb'))
                    fp.annotations = annotations

        self.ris_widget.annotator.update_fields()

    def save_warped_images(self):
        """Save the warped images into the same parent directory as the images in the flipbook.
        If there is a width tck, the saved images have a transparent channel with the worm mask.
        The warped images will be saved with the same base name as the first image in the flipbook_page list
        with "-straightening.png" afterwards.

        Example:
            image name: 'test.png'
            straightened image name: 'test-straightened.png'
        """
        for fp in self.ris_widget.flipbook_pages:
            lab_frame_image = fp[0].data
            annotations = getattr(fp, 'annotations', None)
            if annotations is not None:
                pose = annotations.get('pose')
                if pose is not None or pose is not (None, None):
                    center_tck, width_tck = pose
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

    gp = GeneralPoseAnnotator(rw)
    load_data = Qt.QPushButton('Load')
    load_data.clicked.connect(gp.load_metadata)
    save_data = Qt.QPushButton('Poses')
    save_data.clicked.connect(gp.save_annotations)
    save_warps = Qt.QPushButton('Images')
    save_warps.clicked.connect(gp.save_warped_images)
    pa._add_row(pa.widget.layout(), Qt.QLabel('Data:'),load_data, save_data, save_warps)    
     
    rw.add_image_files_to_flipbook(args.images)
    shared_resources._QAPPLICATION.exec()

if __name__ == '__main__':
    main()