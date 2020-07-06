import tqdm
import numpy as np
from scipy.interpolate import interp1d
from utils.contrast_metrics import calc_contrast
from utils.visualization import (plot_luminance,
                                 visualize_cdp_2d,
                                 visualize_cdp_3d)

EPSILON = 1E-9


class CDPCalculator(object):
    def __init__(self,
                 dts_luminance,
                 rois,
                 method='michelson',
                 chart_ids=(0, 1, 2, 3, 4, 5),
                 num_randomized_pixels=200000):
        """
        :param dts_luminance: dict(patch_id: float)
        :param rois: dict(patch_id: np.ndarray(h, w, 3) or none),
        where h and w are height and width of each image roi.
        :param cfg:
        :param chart_ids: a 0-indexed list to indicate which charts are included in 'rois',
        for example, use chart_ids = [0, 1, 2, 3] to exclude two darkest charts.
                                     """
        self.dts_luminance = dts_luminance
        self.rois = rois
        self.method = method
        self.chart_ids = chart_ids
        self.paired_luminance = []
        self.paired_contrasts = []
        self.rois_brighter = []
        self.rois_darker = []
        self.interp = None
        self.num_randomized_pixels = num_randomized_pixels
        self.init()

    def calculate(self,
                  target_contrasts=(0.06, 0.1, 0.35),
                  confidence=0.5,
                  contrast_tol=0.1,
                  fig_dir=''):
        """
        Calculate CDPs w.r.t. luminance given target contrast value(s)
        :param target_contrasts: float or list, target contrast(s) for which the cdp is calculated.
        if given a empty list, CDPs will be calculated for all input contrast and luminance combinations.
        :param confidence: the confidence interval for the measurement of the cdp
        :param contrast_tol: the contrast range for which the cdp is calculated
        :param fig_dir: directory to save the visualization results
        :return:
        """
        if isinstance(target_contrasts, (float, int)):
            target_contrasts = [target_contrasts]

        if not target_contrasts:
            target_contrasts = self.paired_contrasts
            plot_mode = '3d'
        else:
            target_contrasts = np.array(target_contrasts)
            plot_mode = '2d'

        input_contrasts, input_luminance, cdps = [], [], []
        print('Calculating CDPs for {} input contrasts...'.format(len(target_contrasts)))
        for idx, c in enumerate(tqdm.tqdm(target_contrasts)):
            if plot_mode == '2d':
                # find all input contrasts within a small range centering at the target contrast
                indices = np.where(
                    (self.paired_contrasts >= (1 - contrast_tol) * c) *
                    (self.paired_contrasts <= (1 + contrast_tol) * c)
                )[0]
                if len(indices) == 0:
                    raise ValueError('RoI pair can not be found for the target contrast {}.'.format(c))
            else:
                indices = [idx]

            for i in indices:
                input_contrasts.append(self.paired_contrasts[i] if plot_mode == '3d' else c)
                input_luminance.append(self.paired_luminance[i])
                roi_contrasts = calc_contrast(self.inverse_transform(self.rois_brighter[i]),
                                              self.inverse_transform(self.rois_darker[i]),
                                              method=self.method,
                                              mode='pairwise')
                p = sum(roi_contrasts < (1 + confidence) * c) - sum(roi_contrasts < (1 - confidence) * c)
                cdps.append(p / roi_contrasts.size)

        input_contrasts = np.array(input_contrasts)
        input_luminance = np.array(input_luminance)
        cdps = np.array(cdps)

        if plot_mode == '3d':
            visualize_cdp_3d(input_contrasts, input_luminance, cdps, fig_dir)
        else:
            visualize_cdp_2d(input_contrasts, input_luminance, cdps, fig_dir)

        return [{'input_contrast': c,
                 'input_luminance': l,
                 'cdp': cdp} for c, l, cdp in zip(input_contrasts, input_luminance, cdps)]

    def init(self):
        num_images = 0
        # discard charts that not been annotated
        for patch_id, roi in self.rois.items():
            if roi is None:
                self.dts_luminance[patch_id] = None
            else:
                num_images = roi.shape[0]

        print('{} luminance levels and {} images are used.'.format(
            sum(bool(_) for _ in self.dts_luminance.values()), num_images)
        )

        # interpolation
        dts_luminance = np.array([v for v in self.dts_luminance.values() if v is not None])
        rois = [v for v in self.rois.values() if v is not None]
        mean_patch_values = [np.mean(r) for r in rois]
        self.interp = interp1d(mean_patch_values, dts_luminance,
                               kind='quadratic', fill_value='extrapolate')
        plot_luminance(mean_patch_values, dts_luminance)

        # sort the RoIs by luminance in descending order
        indices = np.argsort(-dts_luminance)
        dts_luminance = dts_luminance[indices]
        rois = [rois[i] for i in indices]

        # calculate all possible input contrasts
        indices_b, indices_d = np.triu_indices(len(indices), 1)
        self.paired_luminance = (dts_luminance[indices_b] + dts_luminance[indices_d]) / 2.
        self.paired_contrasts = calc_contrast(dts_luminance[indices_b],
                                              dts_luminance[indices_d],
                                              method=self.method)

        # generate all possible contrasts from RoI pairs
        print('Generating {} RoI pairs...'.format(indices_b.size))
        num_roi_pixels = int(np.sqrt(self.num_randomized_pixels))
        self.rois_brighter = tuple(
            np.random.choice(rois[i].ravel(), num_roi_pixels, replace=False) for i in indices_b
        )
        self.rois_darker = tuple(
            np.random.choice(rois[i].ravel(), num_roi_pixels, replace=False) for i in indices_d
        )

    def inverse_transform(self, pixel_values):
        return np.clip(self.interp(pixel_values), 0, None)
