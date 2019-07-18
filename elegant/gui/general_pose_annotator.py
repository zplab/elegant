import freeimage
import pathlib
import pickle
from PyQt5 import Qt

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
        self.pose_annotator._add_row(self.pose_annotator.widget.layout(), Qt.QLabel('Warps:'), load_data, save_data)

    def save_annotations(self):
        """Save the pose annotations as pickle files into the parent directory.
        A pickle file is created for each page in the flipbook with the name of the first image in the
        flipbook_page list as the base for the pickle file name.
        """
        for fp in self.ris_widget.flipbook_pages:
            if len(fp) == 0:
                # skip empty flipbook pages
                continue
            annotations = getattr(fp, 'annotations', {})
            pose = annotations.get('pose', (None, None))
            if pose is not None:
                center_tck, width_tck = pose
                if center_tck is not None:
                    path = pathlib.Path(fp[0].name)
                    with path.with_suffix('.pickle').open('wb') as f:
                        pickle.dump(dict(pose=pose), f)

                    # warp and save images from all flipbook pages
                    for lab_frame in fp:
                        lab_frame_image = lab_frame.data
                        path = pathlib.Path(lab_frame.name)
                        warp = worm_spline.to_worm_frame(lab_frame_image, center_tck, width_tck)
                        warp_save_path = path.parent / (path.stem + '-straight.png')
                        freeimage.write(warp, warp_save_path)

                        # If the widths are drawn, then create a mask that allows the user to make an alpha channel later.
                        # We create one mask for each flipbook page, in case the images were saved in different places.
                        # If we wind up redundantly writing the same mask a few times, so be it.
                        if width_tck is not None:
                            mask = worm_spline.worm_frame_mask(width_tck, warp.shape)
                            mask_save_path = path.parent / (path.stem + '-mask.png')
                            freeimage.write(mask, mask_save_path)


    def load_annotations(self):
        """Loads annotation metadata from a pickle file for each flipbook page.
        NOTE: If there is already a pose annotation, the metadata will not be loaded.
        """
        for fp in self.ris_widget.flipbook_pages:
            annotations = getattr(fp, 'annotations', {})
            pose = annotations.get('pose')
            if pose in (None, (None, None)):
                annotation_path = pathlib.Path(fp[0].name).with_suffix('.pickle')
                #if there are no annotation pickle files do nothing
                if annotation_path.exists():
                    with annotation_path.open('rb') as f:
                        annotations = pickle.load(f)
                    fp.annotations = annotations
        self.ris_widget.annotator.update_fields()

def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description='straighten worm images')
    parser.add_argument('pixels_per_micron', nargs='?', default=1/1.3, type=float, metavar='pixels_per_micron', help='conversion factor for objective used')
    parser.add_argument('images', nargs="*", metavar='image', help='image files to open')

    args = parser.parse_args(argv)
    rw = ris_widget.RisWidget()
    gp = GeneralPoseAnnotator(rw, pixels_per_micron=args.pixels_per_micron)

    rw.add_image_files_to_flipbook(args.images)
    rw.run()

if __name__ == '__main__':
    main()