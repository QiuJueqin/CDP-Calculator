import numpy as np
import matplotlib.pyplot as plt

from utils.config import load_config


class SimulatedCamera(object):
    def __init__(self, cfg, noise_free=False):
        """
        :param cfg: a namespace that stores configs
        :param noise_free: set to True to make it a perfect noise-free camera
        """
        self.cfg = cfg
        self.noise_free = noise_free

        self.luminous_efficacy = 1000
        self.max_digital_value = 2 ** cfg.linear_bit_depth - 1
        self.linear_dtype = np.uint16 if cfg.linear_bit_depth <= 16 else np.uint32
        self.output_dtype = np.uint8 if cfg.output_bit_depth <= 8 else np.uint16
        self.log_base = (cfg.linear_bit_depth * np.log(2)) / (2 ** cfg.output_bit_depth - 1)

        self.base_iso = None
        self.base_gain = None
        self.iso_calibration()
        
    def capture(self, luminance_map, f_num=8, exposure_time=0.01, iso=100):
        """
        Simulated imaging process that converts luminance in the focal plane
        to the output raw digital values in linear domain
        :param luminance_map: H*W luminance array in cd/m^2
        :param f_num: F number of the lens
        :param exposure_time: exposure time in second, aka shutter speed or
        integration time
        :param iso: ISO speed
        :return: H*W image in linear domain
        """
        illuminance_map = self.optical_process(luminance_map, f_num)
        electrons_map = self.photoelectric_process(illuminance_map, exposure_time, noise_free=self.noise_free)

        raw_image = np.clip(
            (iso / self.base_iso) * self.base_gain * electrons_map, 0, self.max_digital_value
        ).astype(self.linear_dtype)

        return raw_image

    def capture_hdr(self, luminance_map, f_num=8, exposure_time=0.01, iso=100, frames=3, ev_step=1.):
        """
        Capture several frames with different exposure time and blend into an
        HDR image. Here a simplified blending strategy is employed that the pixel
        value in the blended HDR image is chosen and normalized from the frame
        that has the highest non-saturated value, with no interpolation performed.
        :param luminance_map: same as self.capture
        :param f_num: same as self.capture
        :param exposure_time: the SHORTEST exposure time within bracket exposures.
        For example, if wish to set the shutter time for 3 frames in a burst to
        1s, 0.1s, and 0.01s respectively, then set exposure_time=0.01
        :param iso: same as self.capture
        :param frames: number of frames in the burst
        :param ev_step: EV step between neighboring exposures, usually 1/3, 1/2,
        1, 2, etc. For example, if exposure_time=0.1, frames=5, and ev_step=1/2,
        the shutter speeds in the burst will be 0.4s, 0.28s, 0.2s, 0.14s and 0.1s
        :return: H*W HDR image in linear domain
        """
        brackets = np.power(2., np.linspace((frames - 1) * ev_step, 0, frames))  # longest frame first
        burst = []

        for i in range(frames):
            exp_cur_frame = exposure_time * brackets[i]
            print('Taking {}/{} frame. Shutter speed: {:.0f}ms'.format(i + 1, frames, 1000 * exp_cur_frame))
            burst.append(self.capture(luminance_map, f_num=f_num, exposure_time=exp_cur_frame, iso=iso))

        print('Blending {} frames into an HDR image.'.format(frames), end=' ')
        burst = np.stack(burst, axis=2)  # np.ndarray(H, W, frames)
        non_saturation_indices = (burst <= 0.99 * self.max_digital_value)
        # if all frames are saturated, choose the one with fastest shutter speed (last frame)
        non_saturation_indices[:, :, -1] = True
        first_non_saturated_frame = non_saturation_indices.argmax(axis=2)

        burst = np.clip(
            burst / brackets.reshape((1, 1, -1)), 0, self.max_digital_value
        ).astype(self.linear_dtype)  # normalization

        hdr_image = np.take_along_axis(burst, first_non_saturated_frame[:, :, None], axis=2).squeeze(2)
        print('Done.')

        return hdr_image

    def tone_mapping(self, linear_image):
        nonlinear_image = (np.log(linear_image + 1) / self.log_base).astype(self.output_dtype)
        return nonlinear_image

    def iso_calibration(self):
        """
        Calibrate camera's base ISO speed and gain using saturation-based
        calculations as per ISO 12232-2019 and OnSemi ISO Measurement document:
        http://www.onsemi.com/pub/Collateral/TND6115-D.PDF
        """
        arbitrary_f_num = 8
        arbitrary_exposure_time = 0.01
        arbitrary_luminance = 100.

        num_electrons = self.photoelectric_process(
            self.optical_process(arbitrary_luminance, arbitrary_f_num), arbitrary_exposure_time, noise_free=True
        )
        saturated_luminance = (0.1698 * self.cfg.full_well_capacity / num_electrons) * arbitrary_luminance / 0.18
        self.base_iso = 15.4 * (arbitrary_f_num ** 2) / (0.18 * saturated_luminance * arbitrary_exposure_time)

        saturated_num_electrons = self.photoelectric_process(
            self.optical_process(saturated_luminance, arbitrary_f_num), arbitrary_exposure_time, noise_free=True
        )
        self.base_gain = self.max_digital_value / saturated_num_electrons

    # ========== Simulated Optical-Electronic-Digital Processes ==========

    def optical_process(self, luminance_map, f_num):
        """
        Optical process that transforms luminance (in cd/m^2) in the
        focal plane to illuminance (in lux) in the sensor plane
        """
        illuminance_map = luminance_map * np.pi * self.cfg.optical_transmission / (4 * f_num ** 2)

        return np.asarray(illuminance_map)

    def photoelectric_process(self, illuminance_map, exposure_time, noise_free=False):
        """
        Photoelectric process that converts photons (illuminance
        times integration time) to the emitted electrons
        """
        photons_map = (illuminance_map * exposure_time *
                       5.05E24 * self.cfg.wavelength * self.cfg.pixel_area) / self.luminous_efficacy
        photons_map = np.round(photons_map)

        if not noise_free:
            # normal approximation to Poisson distribution when lambda > 1000
            poisson_mask = photons_map < 1000
            photons_map = poisson_mask * np.random.poisson(poisson_mask * photons_map) + \
                (1 - poisson_mask) * np.random.normal(photons_map, np.sqrt(photons_map))

        electrons_map = photons_map * self.cfg.quantum_efficacy

        if not noise_free:
            poisson_mask = electrons_map < 1000
            electrons_map = poisson_mask * np.random.poisson(poisson_mask * electrons_map) + \
                (1 - poisson_mask) * np.random.normal(electrons_map, np.sqrt(electrons_map))

        return np.clip(electrons_map, 0, self.cfg.full_well_capacity)


def unit_test(cfg, inputs):
    for k, v in inputs.items():
        if isinstance(v, np.ndarray):
            variable = k
            break
    for k, v in inputs.items():
        if k != variable:
            inputs[k] = [v] * len(inputs[variable])

    camera = SimulatedCamera(cfg)

    raw_responses = []
    for lum, f, exp, iso in zip(inputs['luminance'],
                                inputs['f_num'],
                                inputs['exposure_time'],
                                inputs['iso']):
        raw_responses.append(
            camera.capture(lum, f, exp, iso).squeeze()
        )
    fig, ax = plt.subplots()
    ax.scatter(inputs[variable], raw_responses)
    ax.set_xlabel(variable), ax.set_ylabel('raw response')
    ax.grid()


def test():
    cfg = load_config('../utils/configurations/simulated_camera_24bit.cfg')

    # luminance test
    inputs = {'luminance': np.linspace(0.01, 10000, 100),  # 0.01 to 10000 cd/m^2
              'f_num': 8, 'exposure_time': 0.01, 'iso': 100}
    unit_test(cfg, inputs)

    # F-number test
    inputs = {'f_num': np.logspace(1, 4, 10, base=2),  # 2 to 16 with 0.3 stop
              'luminance': 1000, 'exposure_time': 0.01, 'iso': 100}
    unit_test(cfg, inputs)

    # exposure time test
    inputs = {'exposure_time': np.linspace(0.001, 0.5, 100),  # 0.001 to 0.5 second
              'luminance': 1000, 'f_num': 16, 'iso': 100}
    unit_test(cfg, inputs)

    # ISO speed test
    inputs = {'iso': 100 * np.logspace(0, 4, 13, base=2),  # ISO100 to ISO1600
              'luminance': 1000, 'f_num': 8, 'exposure_time': 0.005}
    unit_test(cfg, inputs)

    plt.show()


if __name__ == '__main__':
    test()
