import os
import time
import glob
import json

import tqdm
import numpy as np
import cv2
import skimage.io
import tifffile
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from utils.config import load_config
from utils.misc import rgb2luminance, NumpyEncoder


class DTSRoIExtractor(object):
    def __init__(self, cfg, chart_ids=(0, 1, 2, 3, 4, 5), ignore_exist=False):
        """
        :param cfg:
        :param chart_ids: a 0-indexed list to indicate which charts are included in 'rois',
        for example, use chart_ids = [0, 1, 2, 3] to exclude two darkest charts.
        :param ignore_exist: force to override existing rois from historical experiments
        """
        self.cfg = cfg
        self.chart_ids = chart_ids
        self.ignore_exist = ignore_exist
        self.num_charts = 0
        self.max_value = float(2 ** cfg.output_bit_depth - 1)

        self.dtype = np.uint8 if cfg.output_bit_depth <= 8 else \
            np.uint16 if cfg.output_bit_depth <= 16 else np.uint32

        # reference square w.r.t. which the perspective transformation is calculated
        self.corners_ref = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)  # [lt, rt, lb, rb] in [x, y]
        cx, cy = np.meshgrid(np.linspace(0, 1, 13, dtype=np.float32)[1:-1:2],
                             np.linspace(0, 1, 13, dtype=np.float32)[1:-1:2])
        self.roi_centers_ref = np.stack([cx.flatten(order='f'), cy.flatten(order='f')], axis=1).reshape((-1, 1, 2))

    def extract_images(self, image_dir):
        roi_boxes, rois = None, None
        metas = []
        image_paths = glob.glob(os.path.join(image_dir, '*.{}'.format(self.cfg.format)))

        # check if there is historical RoIs info file
        rois_info = None if self.ignore_exist else self.load_history(image_dir)
        use_history = (rois_info is not None)
        roi_boxes = rois_info['roi_boxes'] if use_history else None

        for image_path in tqdm.tqdm(image_paths):
            image = self.imread(image_path)
            metas.append({'image_id': os.path.basename(image_path)})
            rois_cur_image, roi_boxes = self.extract_image(image, roi_boxes)
            if rois is None:
                rois = rois_cur_image
            else:
                for patch_id, roi in rois.items():
                    rois[patch_id] = np.concatenate(
                        [rois[patch_id], rois_cur_image[patch_id]], axis=0
                    ) if roi is not None else None

        if not use_history:
            self.save_info(roi_boxes, metas, image_dir)

        return rois

    def extract_image(self, image, roi_boxes=None):
        if roi_boxes is None:
            chart_corners = self.select_charts(image)
            roi_boxes = self.corners2rois(chart_corners)
            roi_boxes = self.sort_rois(image, roi_boxes)
            self.plot_rois(image, roi_boxes)

        rois = self.extract_rois(image, roi_boxes)
        return rois, roi_boxes

    @staticmethod
    def extract_rois(image, roi_boxes):
        """
        extract pixel values from image given roi coordinates.
        :param image:
        :param roi_boxes: dict(patch_id: np.ndarray(4,) or None)
        :return: dict(patch_id: roi or None)
        """
        rois = {}
        for patch_id, box in roi_boxes.items():
            if box is not None:
                box = box.astype(np.uint16)
                roi = rgb2luminance(image[box[1]: box[3], box[0]: box[2], :])
                rois[patch_id] = np.expand_dims(roi, axis=0)  # add a 'batch' dim
            else:
                rois[patch_id] = None
        return rois

    def imread(self, file_path):
        if self.cfg.format == 'tiff':
            image = tifffile.imread(file_path).astype(self.dtype)
        elif self.cfg.format in ('jpg', 'png', 'bmp'):
            image = skimage.io.imread(file_path).astype(self.dtype)
        else:
            raise IOError('{} format is not supported.'.format(self.cfg.format))

        if image.ndim == 2:  # H*W case
            image = np.stack((image, ) * 3, axis=-1)
        elif image.shape[2] == 1:  # H*W*1 case
            image = np.tile(image, (1, 1, 3))

        return image  # np.ndarray(H, W, 3)

    def select_charts(self, image):
        """
        Allow user to manually select corners of each chart
        :param image:
        :return: corner coordinates of charts: np.ndarray(n * 4, 2),
        where n is the number of charts and each row is in [x, y] format
        """
        chart_corners = []
        # (x, y) format
        fig = plt.figure('Right click corners of all target charts.')
        plt.imshow(image.astype(np.float32) / self.max_value)

        def onclick(event):
            if event.button == 3:  # right click
                chart_corners.append([int(event.xdata), int(event.ydata)])
                c = plt.Circle((event.xdata, event.ydata), 5, color='r')
                plt.gcf().gca().add_artist(c)
                fig.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.draw()

        chart_corners = np.array(chart_corners)
        self.num_charts = chart_corners.shape[0] // 4
        assert chart_corners.shape[1] == 2 and chart_corners.shape[0] % 4 == 0
        assert self.num_charts >= len(self.chart_ids), \
            'specified {} charts, but only {} are annotated.'.format(
                len(self.chart_ids), self.num_charts
            )

        return chart_corners

    def corners2rois(self, corners):
        """
        calculate bbox coordinates of patch rois
        :param corners: corner coordinates of charts: np.ndarray(n * 4, 2)
        :return: np.ndarray(n * 36, 4), where n is the number of charts and
        each row is in [x1, y1, x2, y2] format
        """
        roi_boxes = []
        for i in range(self.num_charts):
            corners_cur_chart = corners[4 * i:4 * i + 4, :]
            corners_cur_chart = correct_corners_order(corners_cur_chart)
            # use perspective transform to get coordinates of patch centers
            mat = cv2.getPerspectiveTransform(self.corners_ref, corners_cur_chart.astype(np.float32))
            roi_centers_cur_chart = cv2.perspectiveTransform(self.roi_centers_ref, mat).squeeze()  # 36 * 2, [x, y]
            roi_centers_3d = np.reshape(roi_centers_cur_chart, (6, 6, 2), order='f')
            patch_distance_x = np.min(np.abs(roi_centers_3d[:, 1:, 0] - roi_centers_3d[:, :-1, 0]))
            patch_distance_y = np.min(np.abs(roi_centers_3d[1:, :, 1] - roi_centers_3d[:-1, :, 1]))
            roi_radius_x = self.cfg.roi_to_patch_ratio * 0.5 * patch_distance_x
            roi_radius_y = self.cfg.roi_to_patch_ratio * 0.5 * patch_distance_y
            roi_cur_chart = np.hstack([roi_centers_cur_chart[:, 0:1] - roi_radius_x,
                                       roi_centers_cur_chart[:, 1:] - roi_radius_y,
                                       roi_centers_cur_chart[:, 0:1] + roi_radius_x,
                                       roi_centers_cur_chart[:, 1:] + roi_radius_y])
            roi_boxes.append(roi_cur_chart)

        return np.vstack(roi_boxes)

    def sort_rois(self, image, roi_boxes):
        """
        sort charts and patches in descending order by their average pixel intensities
        :param image:
        :param roi_boxes: np.ndarray(36 * n, 4), each row is in [x1, y1, x2, y2] format
        :return: sorted boxes: dict(patch_id: np.ndarray(4,) or None)
        """
        roi_means = []
        for box in roi_boxes.astype(np.uint16):
            roi = image[box[1]: box[3], box[0]: box[2], :]
            roi_means.append(np.mean(roi))

        roi_means = np.reshape(roi_means, (6, 6, -1), order='f')  # (6, 6, n) tensor
        # (6, 6) intensity matrix of intermediate luminance chart
        mid_chart_roi_means = roi_means[:, :, self.num_charts // 2]
        hflip = True if mid_chart_roi_means[:, 0].mean() < mid_chart_roi_means[:, -1].mean() else False
        vflip = True if mid_chart_roi_means[0, :].mean() < mid_chart_roi_means[-1, :].mean() else False
        roi_boxes = np.reshape(roi_boxes, (6, 6, -1, 4), order='f')
        if hflip:
            roi_boxes = roi_boxes[:, ::-1, :, :]
        if vflip:
            roi_boxes = roi_boxes[::-1, :, :, :]
        # sort charts in descending order by their luminance levels
        charts_indices = np.argsort(-np.mean(roi_means, axis=(0, 1)))
        roi_boxes = roi_boxes[:, :, charts_indices, :]
        roi_boxes = np.reshape(roi_boxes, (-1, 4), order='f')
        roi_boxes_sorted = {}
        box_id = 0
        for i in range(216):
            if (i // 36) in self.chart_ids:
                roi_boxes_sorted[i] = roi_boxes[box_id, :]
                box_id += 1
            else:
                roi_boxes_sorted[i] = None
        return roi_boxes_sorted

    def plot_rois(self, image, roi_boxes):
        """
        :param image:
        :param roi_boxes: dict(patch_id: np.ndarray(4,) or None)
        :return:
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(image.astype(np.float32) / self.max_value)
        for patch_id, box in roi_boxes.items():
            if box is None:
                continue
            ax.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                   fc='None', ec='r', lw=3))
            plt.text((box[0] + box[2]) // 2, (box[1] + box[3]) // 2, str(patch_id),
                     horizontalalignment='center', verticalalignment='center'
                     )
        plt.draw()

    def save_info(self, roi_boxes, metas, image_dir):
        info = {'chart_ids': self.chart_ids,
                'roi_boxes': roi_boxes,
                'metas': metas}
        timestamp = time.strftime('%Y-%m-%d-%H-%M')
        save_path = os.path.join(image_dir, 'rois_info_{}.json'.format(timestamp))
        with open(save_path, 'w') as fp:
            json.dump(info, fp, indent=4, cls=NumpyEncoder)

        print('Saved RoIs info to {}.'.format(save_path))

    def load_history(self, image_dir):
        rois_files = glob.glob(os.path.join(image_dir, '*.json'))
        rois_files = [f for f in rois_files if 'rois_info_' in f]
        if len(rois_files) == 0:
            return None
        with open(max(rois_files)) as fp:  # find the latest file if there are more than one
            rois_info = json.load(fp)

        assert rois_info['chart_ids'] == list(self.chart_ids), \
            'Chart IDs changed. Delete historical files or use ignore_exist=True to override them.'
        rois_info['roi_boxes'] = {
            int(i): None if b is None else np.array(b) for i, b in rois_info['roi_boxes'].items()
        }

        print('Loaded historical RoI info from {}.'.format(max(rois_files)))

        return rois_info


def correct_corners_order(corners):
    """
    reorder corners of square to [lt, rt, lb, rb] order
    :param corners: np.array(4, 2), each row in [x, y] format
    :return:
    """
    center = np.mean(corners, axis=0)
    angles = np.arctan2(center[1] - corners[:, 1], corners[:, 0] - center[0]) * 180 / np.pi
    idx_lb, idx_rb, idx_rt, idx_lt = np.argsort(angles)
    return corners[[idx_lt, idx_rt, idx_lb, idx_rb], :]

    
def main():
    cfg = load_config('utils/configurations/simulated_camera_24bit.cfg')
    extractor = DTSRoIExtractor(cfg)
    image_dir = r'/simulation/images'
    rois = extractor.extract_images(image_dir)


if __name__ == '__main__':
    main()
