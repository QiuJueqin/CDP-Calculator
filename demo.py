import os

import tifffile

from simulation.simulated_camera import SimulatedCamera
from simulation.simulated_patterns import (generate_dts_luminance,
                                           generate_dts_luminance_map)

from cdp_calculator import CDPCalculator
from roi_extractor import DTSRoIExtractor
from utils.config import load_config


def main():
    max_luminance = 50000
    num_images = 4
    simulated_images_dir = './simulation/simulated_images'
    os.makedirs(simulated_images_dir, exist_ok=True)

    dts_luminance = generate_dts_luminance(max_luminance=max_luminance)
    dts_luminance_map = generate_dts_luminance_map(dts_luminance)

    cfg = load_config('./utils/configurations/simulated_camera_24bit.cfg')
    camera = SimulatedCamera(cfg)
    for i in range(num_images):
        print('Generating {}/{} HDR image'.format(i + 1, num_images))
        image = camera.capture_hdr(dts_luminance_map, f_num=8, exposure_time=0.001, iso=125, frames=3, ev_step=4)
        image = camera.tone_mapping(image)
        save_path = os.path.join(simulated_images_dir, 'simulated_image_{}.tiff'.format(i))
        tifffile.imwrite(save_path, image)
        print('')

    extractor = DTSRoIExtractor(cfg)
    rois = extractor.extract_images(simulated_images_dir)

    calculator = CDPCalculator(dts_luminance, rois, fig_dir=simulated_images_dir)
    calculator.calculate()


if __name__ == '__main__':
    main()
