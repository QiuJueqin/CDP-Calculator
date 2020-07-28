import tqdm
import numpy as np
from scipy.interpolate import interp1d
from utils.contrast_metrics import calc_contrast
from utils.visualization import (plot_luminance,
                                 visualize_cdp_2d,
                                 visualize_cdp_3d)

EPSILON = 1E-9


class CDPCalculator(object):
    def __init__(self, dts_luminance, rois, method='michelson', num_samples=50000, fig_dir=''):
        """
        Contrast Detection Probability Calculator. See
        `https://www.imatest.com/docs/cdp/` and
        `https://www.image-engineering.de/library/conference-papers/1045-contrast-
        detection-probability-implementation-and-use-cases` for more details
        :param dts_luminance: dict(patch_id: float)
        :param rois: dict(patch_id: np.ndarray(h, w, 3) or none), where h and w are
            height and width of each RoI
        :param method: contrast metric: 'michelson' | 'weber'
        :param num_samples: number of randomly-sampled pixel pairs for generating
            distribution from which CDP is calculated. Larger number results in more
            accurate calculation. (>50000 is recommended)
        :param fig_dir: directory to save the figures. If empty, figures will be
            interactively shown on the screen
        """
        self.dts_luminance = dts_luminance
        self.rois = rois
        self.method = method
        self.num_randomized_pixels = num_samples
        self.fig_dir = fig_dir

        self.paired_luminance = []
        self.paired_contrasts = []
        self.brighter_rois = []
        self.darker_rois = []
        self.interp = None

        self.init()

    def calculate(self, target_contrasts=(0.06, 0.1, 0.2, 0.3), confidence=0.5, contrast_tol=0.1):
        """
        Calculate CDPs w.r.t. luminance given target contrast value(s)
        :param target_contrasts: float or list, target contrast(s) for which the CDP
            is calculated
        :param confidence: the confidence interval for the calculation of CDP
        :param contrast_tol: contrast tolerance. All RoI pairs in a range of contrasts
            (target_contrast ± tolerance) will be employed to calculate CDP
        """
        if isinstance(target_contrasts, (float, int)):
            target_contrasts = [target_contrasts]
        target_contrasts = np.asarray(target_contrasts)

        input_contrasts, input_luminance, cdps = [], [], []
        for c in target_contrasts:
            print('Calculating CDPs for input contrast {:.0f}%:'.format(100*c), end=' ')

            # find all input contrasts within a small range centering at the target contrast
            indices = np.where(
                (self.paired_contrasts >= (1 - contrast_tol) * c) *
                (self.paired_contrasts <= (1 + contrast_tol) * c)
            )[0]

            if len(indices) == 0:
                raise ValueError('No patch-pair can be found for the target contrast {}. '
                                 'Try using a larger contrast_tol value.'.format(c))
            else:
                print('found {} patch-pairs'.format(len(indices)))

            input_contrasts.append([c] * indices.size)
            input_luminance.append(self.paired_luminance[indices])
            for i in tqdm.tqdm(indices):
                roi_contrasts = calc_contrast(self.brighter_rois[i],
                                              self.darker_rois[i],
                                              method=self.method,
                                              mode='pairwise')
                p = sum(roi_contrasts < (1 + confidence) * c) - sum(roi_contrasts < (1 - confidence) * c)
                cdps.append(p / roi_contrasts.size)

        input_contrasts = np.hstack(input_contrasts)
        input_luminance = np.hstack(input_luminance)
        cdps = np.array(cdps)

        visualize_cdp_2d(input_contrasts, input_luminance, cdps, self.fig_dir)

        result = [{'input_contrast': c, 'input_luminance': l, 'cdp': cdp} for c, l, cdp in zip(
            input_contrasts, input_luminance, cdps
        )]
        return result

    def calculate_all(self, confidence=0.5):
        """
        Calculate CDPs for all possible RoI pairs. Each RoI pair, e.g. B(righter) and
            D(arker), will generate one input contrast value: B/D-1 or (B-D)/(B+D),
            one luminance value: (B+D)/2, and one CDP value. For a test chart that
            contains N patches with different luminance levels, there exist (N*(N-1))/2
            RoI pairs
        :param confidence: the confidence interval for the calculation of CDP
        """
        print('Calculating CDPs for {} input contrast levels...'.format(len(self.paired_contrasts)))

        cdps = []
        for i, c in enumerate(tqdm.tqdm(self.paired_contrasts)):
            roi_contrasts = calc_contrast(self.brighter_rois[i],
                                          self.darker_rois[i],
                                          method=self.method,
                                          mode='pairwise')
            p = sum(roi_contrasts < (1 + confidence) * c) - sum(roi_contrasts < (1 - confidence) * c)
            cdps.append(p / roi_contrasts.size)

        cdps = np.array(cdps)

        visualize_cdp_3d(self.paired_contrasts, self.paired_luminance, cdps, self.fig_dir)

        result = [{'input_contrast': c, 'input_luminance': l, 'cdp': cdp} for c, l, cdp in zip(
            self.paired_contrasts, self.paired_luminance, cdps
        )]
        return result

    def init(self):
        """ Prepare transforming pixel values back to luminance domain,
            and generate all possible RoI pairs """
        num_images = rois[0].shape[0]
        print('{} luminance levels and {} images are used'.format(
            len(self.dts_luminance), num_images)
        )

        # prepare interpolation
        dts_luminance = np.array([v for v in self.dts_luminance.values() if v is not None])
        rois = [v for v in self.rois.values() if v is not None]
        mean_patch_values = np.array([np.mean(r) for r in rois])

        unique_mean_patch_values, unique_indices = np.unique(mean_patch_values, return_index=True)
        unique_dts_luminance = dts_luminance[unique_indices]
        self.interp = interp1d(unique_mean_patch_values, unique_dts_luminance,
                               kind='quadratic', fill_value='extrapolate')
        plot_luminance(mean_patch_values, dts_luminance, self.fig_dir)

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

        # generate all possible RoI pairs and transform pixel values back to luminance domain
        print('Generating {} RoI pairs. Keep patient'.format(indices_b.size))
        num_roi_pixels = int(np.sqrt(self.num_randomized_pixels))
        self.brighter_rois = tuple(self.inverse_transform(
            np.random.choice(rois[i].ravel(), num_roi_pixels, replace=False)
            ) for i in indices_b)
        self.darker_rois = tuple(self.inverse_transform(
            np.random.choice(rois[i].ravel(), num_roi_pixels, replace=False)
            ) for i in indices_d)

    def inverse_transform(self, pixel_values):
        """ Transform pixel values back to luminance domain """
        return np.clip(self.interp(pixel_values), 0, None)
