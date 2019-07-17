import freeimage
import pathlib
import pickle
from PyQt5 import Qt
import numpy

from ris_widget import ris_widget
from elegant.gui import pose_annotation
from elegant import worm_widths
from elegant import worm_spline

class GeneralPoseAnnotator:
    def __init__(self, rw, pixels_per_micron=1/1.3):
        self.ris_widget = rw
        width_estimator = worm_widths.WidthEstimator.from_default_widths(pixels_per_micron=pixels_per_micron)
        self.pose_annotator = pose_annotation.PoseAnnotation(self.ris_widget, width_estimator=width_estimator)
        self.ris_widget.add_annotator([self.pose_annotator])
        
        load_data = Qt.QPushButton('Load')
        load_data.clicked.connect(self.load_annotations)
        save_data = Qt.QPushButton('Save')
        save_data.clicked.connect(self.save_annotations)
        self.pose_annotator._add_row(self.pose_annotator.widget.layout(), Qt.QLabel('Data:'),load_data, save_data)

    def save_annotations(self):
        """Save the pose annotations as pickle files into the parent directory.
        A pickle file is created for each page in the flipbook with the name of the first image in the 
        flipbook_page list as the base for the pickle file name.
        """
        for fp in self.ris_widget.flipbook_pages:
            annotations = getattr(fp, 'annotations', None)
            if annotations is not None:
                pose = annotations.get('pose')
                if pose is not None and pose is not (None, None):
                    path = pathlib.Path(fp[0].name)
                    lab_frame_image = fp[0].data
                    center_tck, width_tck = pose
                    #warp and save the warped images
                    if center_tck is not None:
                        warp = worm_spline.to_worm_frame(lab_frame_image, center_tck, width_tck)
                        warp_save_path = path.parent/(path.stem+"-straight.png")
                        freeimage.write(warp, warp_save_path)

                        #if the widths are drawn, then create a mask that allows the user to make an alpha channel later
                        if width_tck is not None:
                            worm_frame_mask = worm_spline.worm_frame_mask(width_tck, warp.shape)
                            img_save_path = path.parent/(path.stem+"-mask.png")
                            freeimage.write(worm_frame_mask, img_save_path)

                    #save annotations into a pickle file
                    save_path = path.with_suffix('.pickle')
                    with save_path.open('wb') as f:
                        pickle.dump(annotations, f)

    def load_annotations(self):
        """Loads annotation metadata from a pickle file for each flipbook page.
        NOTE: If there is already a pose annotation, the metadata will not be loaded.
        """
        for fp in self.ris_widget.flipbook_pages:
            annotations = getattr(fp, 'annotations', {})
            pose = annotations.get('pose')
            if pose in (None, (None, None)):
                path = pathlib.Path(fp[0].name)
                annotation_path = path.with_suffix('.pickle')
                #if there are no annotation pickle files do nothing
                if annotation_path.exists():
                    with annotation_path.open('rb') as f:
                        annotations = pickle.load(f)
                    fp.annotations = annotations

        self.ris_widget.annotator.update_fields()

def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="zplab image viewer")
    parser.add_argument('pixels_per_micron', nargs='?', default=1/1.3, type=float, metavar='pixels_per_micron', help='conversion factor for objective used')
    parser.add_argument('images', nargs="*", metavar='image', help='image files to open')

    args = parser.parse_args(argv)
    rw = ris_widget.RisWidget()
    gp = GeneralPoseAnnotator(rw, pixels_per_micron=args.pixels_per_micron)
     
    rw.add_image_files_to_flipbook(args.images)
    rw.run()

if __name__ == '__main__':
    main()