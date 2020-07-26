"""
A demo that visualizes the CDP pattern, simulates the imaging process,
calculates the CDPs with respect to target contrast values, and visualize
the final CDP results.
"""

import os

import tifffile

from simulation.simulated_camera import SimulatedCamera
from simulation.simulated_patterns import (generate_dts_luminance,
                                           generate_dts_luminance_map)

from cdp_calculator import CDPCalculator
from dynamic_test_stand.roi_extractor import DTSRoIExtractor
from utils.misc import load_config


MAX_LUMINANCE = 50000  # luminance of the light source, in cd/m^2
NUM_IMAGES = 4  # capturing several frames will produce more accurate results

# load the configuration of the simulated camera
cfg = load_config('./utils/configurations/simulated_camera_24bit.json')

simulated_images_dir = './simulation/simulated_images'  # directory to save the simulated images
os.makedirs(simulated_images_dir, exist_ok=True)

# generate simulated luminance map of CDP pattern in DTS device
dts_luminance = generate_dts_luminance(max_luminance=MAX_LUMINANCE)
dts_luminance_map = generate_dts_luminance_map(dts_luminance)

# a simulated camera that captures the CDP pattern
camera = SimulatedCamera(cfg)
for i in range(NUM_IMAGES):
    print('Generating {}/{} HDR image'.format(i + 1, NUM_IMAGES))
    image = camera.capture_hdr(dts_luminance_map, f_num=8, exposure_time=0.001, iso=125, frames=3, ev_step=4)
    image = camera.tone_mapping(image)
    save_path = os.path.join(simulated_images_dir, 'simulated_image_{}.tiff'.format(i))
    tifffile.imwrite(save_path, image)
    print('')

# a utility to allow user to annotate CDP pattern in the image
extractor = DTSRoIExtractor(cfg)
rois = extractor.extract_images(simulated_images_dir)

# In the demo we use a small num_samples value for the sake of speed. However a larger
# number will produce more accurate CDP calculation. (>50000 is recommended)
calculator = CDPCalculator(dts_luminance, rois, num_samples=10000, fig_dir=simulated_images_dir)
calculator.calculate()
calculator.calculate_all()
