import os.path as op
import time
import json
import glob

import numpy as np
import tifffile
import cv2
import matplotlib.pyplot as plt
import skimage.io
from skimage.draw import polygon

from utils.visualization import plot_rois, ChartSelector
from utils.misc import rgb2luminance, NumpyEncoder


class DTSRoIExtractor(object):
    """
    A utility to allow user to manually annotate CDP pattern in the image(s)
    and extract RoIs for further CDP calculation
    """
    def __init__(self, cfg):
        self.cfg = cfg

        if cfg.output_bit_depth <= 8:
            self.dtype = np.uint8
        elif cfg.output_bit_depth <= 16:
            self.dtype = np.uint16
        else:
            self.dtype = np.uint32

        # initial chart size and chart center coordinates in xy format, both relative to image size
        self.chart_size = 0.2
        self.chart_centers = np.array([[0.15, 0.2], [0.5, 0.2], [0.85, 0.2],
                                       [0.15, 0.8], [0.5, 0.8], [0.85, 0.8]])

        # corners of a reference chart, in xy format and [lt, rt, lb, rb] order
        self.ref_chart_corners = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
        cx, cy = np.meshgrid(np.linspace(0, 1, 13, dtype=np.float32)[1:-1:2],
                             np.linspace(0, 1, 13, dtype=np.float32)[1:-1:2])
        # add an extra dimension to match cv2.perspectiveTransform input format
        self.ref_roi_centers = np.stack([cy.ravel(), cx.ravel()]).T.reshape(-1, 1, 2)

        # determine if the image should be flipped
        self.flip_x, self.flip_y = False, False

    def extract_images(self, image_dir, ignore_exist=False, ignore_variance_check=False):
        """
        Extract RoIs from images with dts patterns
        :param image_dir: directory of images
        :param ignore_exist: force to ignore the existing RoIs files in the directory
            and re-run the chart selection procedure, otherwise, the program will
            automatically find if there are historical RoIs files in the directory.
        :param ignore_variance_check: by default this function will check the variance
            among images before running RoIs extraction, set this option to True to
            bypass this procedure. A large variance among set of images generally
            means that there was fluctuation during capturing, which degrades the
            accuracy of CDP calculation.
        :return: dict(patch_id: roi)
        """
        roi_boxes, rois = None, None
        image_paths = glob.glob(op.join(image_dir, '*.{}'.format(self.cfg.format)))

        # check if there exists a historical RoIs info file
        exists = False
        roi_boxes = self.load_rois(image_dir)
        if not ignore_exist and (roi_boxes is not None):
            exists = True

        images = []
        for image_path in image_paths:
            images.append(self.imread(image_path))

        if not ignore_variance_check:
            self.check_images_variance(images)

        for image in images:
            rois_cur_image, roi_boxes = self.extract_image(image, roi_boxes)
            if rois is None:
                rois = rois_cur_image
                continue

            for patch_id, roi in rois.items():
                rois[patch_id] = np.concatenate(
                    [rois[patch_id], rois_cur_image[patch_id]], axis=0
                )

        if not exists:
            self.save_rois(roi_boxes, image_dir)
            plot_rois(image, roi_boxes)

        return rois

    def extract_image(self, image, roi_boxes=None):
        """
        Extract RoIs from one single image
        :param image:
        :param roi_boxes: if None, user will be asked to manually annotate corners of 6
            CDP charts in the image, otherwise, the selected RoI coordinates from the
            previous frame will be reused
        """
        if roi_boxes is None:
            charts_corners = self.annotate_charts(image)
            charts_corners = self.sort_charts(image, charts_corners)
            roi_boxes = self.charts_to_rois(charts_corners.astype(np.float32))

        rois = self.extract_rois(image, roi_boxes)
        return rois, roi_boxes

    def imread(self, file_path):
        if self.cfg.format == 'tiff':
            image = tifffile.imread(file_path).astype(self.dtype)
        elif self.cfg.format in ('jpg', 'png', 'bmp'):
            image = skimage.io.imread(file_path).astype(self.dtype)
        else:
            raise IOError('currently {} format is not supported.'.format(self.cfg.format))

        if image.ndim == 2:  # (H, W) case
            image = np.stack((image, ) * 3, axis=-1)
        elif image.shape[2] == 1:  # (H, W, 1) case
            image = np.tile(image, (1, 1, 3))

        return image  # (H, W, 3)

    def annotate_charts(self, image):
        """
        Ask user to manually select CDP charts by dragging chart corners
        :param image:
        :return: np.ndarray(24, 2), coordinates of 24 corners in xy format
        """
        height, width, _ = image.shape

        fig, ax = plt.subplots()
        plt.get_current_fig_manager().window.state('zoomed')
        ax.imshow(image.astype(np.float32) / image.max())
        ax.set_title('Drag red corners to locate CDP charts. Close to continue.')
        chart_selectors = []
        for i in range(len(self.chart_centers)):
            chart_selectors.append(
                ChartSelector(ax,
                              chart_center=(self.chart_centers[i, 0] * width, self.chart_centers[i, 1] * height),
                              chart_size=(self.chart_size * height, self.chart_size * height))
            )

        plt.show()
        corners = []
        for i in range(len(chart_selectors)):
            corners.append(chart_selectors[i].vertices)
            chart_selectors[i].disconnect()

        return np.array(corners).reshape(-1, 2)

    def sort_charts(self, image, charts_corners):
        """
        Determine if the input image should be flipped and
        sort charts by their luminance in descending order.
        """
        image = image / image.max()  # normalize to avoid overflow
        assert charts_corners.shape[0] == 24, 'not enough annotated charts.'
        chart_mean_values = []
        for chart_id in range(6):
            chart_corners = charts_corners[4 * chart_id: 4 * (chart_id + 1), :]
            mask = np.zeros_like(image)
            r, c = polygon(chart_corners[:, 1], chart_corners[:, 0])
            mask[r, c, :] = 1
            chart_mean_values.append((image * mask).sum() / mask.sum())

        chart_mean_values = np.reshape(chart_mean_values, (2, 3))
        if np.mean(chart_mean_values[:, 0]) < np.mean(chart_mean_values[:, 2]):
            self.flip_x = True
            print('Image will be horizontally flipped.')
        if np.mean(chart_mean_values[0, :]) < np.mean(chart_mean_values[1, :]):
            self.flip_y = True
            print('Image will be vertically flipped.')

        charts_corners = np.reshape(charts_corners, (2, 3, 4, 2))
        if self.flip_x:
            charts_corners = charts_corners[:, ::-1, ...]
        if self.flip_y:
            charts_corners = charts_corners[::-1, ...]

        charts_corners = np.reshape(charts_corners, (-1, 2))

        for chart_id in range(6):
            charts_corners[4 * chart_id: 4 * (chart_id + 1), :] = self.sort_corners(
                charts_corners[4 * chart_id: 4 * (chart_id + 1), :]
            )

        return charts_corners

    def charts_to_rois(self, charts_corners):
        """
        Calculate box coordinates of all patches for all charts
        :param charts_corners: np.ndarray(24, 2), coordinates of 24 corners in xy format
        :return: Dict(patch_id: roi_box in xyxy format)
        """
        roi_boxes = []
        for chart_id in range(6):
            chart_corners = charts_corners[4 * chart_id: 4 * (chart_id + 1), :].reshape(2, 2, 2)
            if self.flip_x:
                chart_corners = chart_corners[:, ::-1, :]
            if self.flip_y:
                chart_corners = chart_corners[::-1, ...]

            # use perspective transform to get patch centers' coordinates
            mat = cv2.getPerspectiveTransform(self.ref_chart_corners, chart_corners.reshape(-1, 2))
            roi_centers = cv2.perspectiveTransform(self.ref_roi_centers, mat).squeeze()
            roi_centers_3d = np.reshape(roi_centers, (6, 6, 2), order='f')
            patch_distance_x = np.min(np.abs(roi_centers_3d[:, 1:, 0] - roi_centers_3d[:, :-1, 0]))
            patch_distance_y = np.min(np.abs(roi_centers_3d[1:, :, 1] - roi_centers_3d[:-1, :, 1]))
            roi_radius_x = self.cfg.roi_to_patch_ratio * 0.5 * patch_distance_x
            roi_radius_y = self.cfg.roi_to_patch_ratio * 0.5 * patch_distance_y
            roi_boxes.append(np.hstack([roi_centers[:, :1] - roi_radius_x,
                                        roi_centers[:, 1:] - roi_radius_y,
                                        roi_centers[:, :1] + roi_radius_x,
                                        roi_centers[:, 1:] + roi_radius_y]))

        return {i: box for i, box in enumerate(np.concatenate(roi_boxes, axis=0))}

    @staticmethod
    def extract_rois(image, roi_boxes):
        """
        Extract pixel values from one image given RoI coordinates
        :param image:
        :param roi_boxes: dict(patch_id: np.ndarray(4,)), box is in xyxy format
        :return: dict(patch_id: roi)
        """
        rois = {}
        for patch_id, box in roi_boxes.items():
            box = box.astype(np.uint16)
            roi = rgb2luminance(image[box[1]: box[3], box[0]: box[2], :])
            rois[patch_id] = np.expand_dims(roi, axis=0)  # add a 'batch' dim

        return rois

    @staticmethod
    def save_rois(roi_boxes, save_dir):
        save_path = op.join(save_dir, 'rois_info_{}.json'.format(time.strftime('%Y-%m-%d-%H-%M')))
        with open(save_path, 'w') as fp:
            json.dump({'roi_boxes': roi_boxes}, fp, indent=4, cls=NumpyEncoder)

        print('Saved RoIs info to {}.'.format(save_path))

    @staticmethod
    def load_rois(load_dir):
        load_paths = [f for f in glob.glob(op.join(load_dir, '*.json')) if 'rois_info_' in f]
        if len(load_paths) == 0:
            return None

        with open(max(load_paths)) as fp:  # find the latest file if there are more than one
            rois = json.load(fp)
        roi_boxes = {int(i): np.array(b) for i, b in rois['roi_boxes'].items()}

        print('Loaded historical RoIs info from {}.'.format(max(load_paths)))
        return roi_boxes

    @staticmethod
    def sort_corners(corners):
        """
        Sort corners of one chart to [lt, rt, lb, rb] order
        :param corners: np.ndarray(4, 2), each row in xy format
        :return: np.ndarray(4, 2) in sorted order
        """
        center = np.mean(corners, axis=0)
        angles = np.arctan2(center[1] - corners[:, 1], corners[:, 0] - center[0])
        idx_lb, idx_rb, idx_rt, idx_lt = np.argsort(angles)
        return corners[(idx_lt, idx_rt, idx_lb, idx_rb), :]

    @staticmethod
    def check_images_variance(images):
        # todo
        pass
