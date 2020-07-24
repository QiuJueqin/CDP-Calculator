import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def generate_dts_luminance(max_luminance=40000, max_reflectance=0.998, decay_factor=0.94):
    """
    :param max_luminance: luminance of the light source, in cd/m^2
    :param max_reflectance: reflectance of the brightest patch
    :param decay_factor: reflectance decay factor between neighboring patches
    :return: Dict(patch_id: reflectance), patch_id starts from the brightest chart, column first
    """
    dts_reflectance = [max_reflectance]
    for patch_id in range(216):
        factor = np.random.normal(decay_factor, 0.01)  # add small perturbation
        dts_reflectance.append(factor * dts_reflectance[-1])

    patch_ids_per_chart = [[0, 2, 5, 9, 14, 20],
                           [1, 4, 8, 13, 19, 25],
                           [3, 7, 12, 18, 24, 29],
                           [6, 11, 17, 23, 28, 32],
                           [10, 16, 22, 27, 31, 34],
                           [15, 21, 26, 30, 33, 35]]

    dts_luminance = {}
    for patch_id in range(216):
        chart_id = patch_id // 36
        row = (patch_id % 36) % 6
        col = (patch_id % 36) // 6
        num_decays = chart_id * 36 + patch_ids_per_chart[row][col]
        dts_luminance[patch_id] = max_luminance * dts_reflectance[num_decays]

    return dts_luminance


def generate_dts_luminance_map(dts_luminance, ambient_luminance=1E-4, reflectance_std=0.001, image_size=(1280, 1920)):
    """
    Generate luminance map with DTS chart patterns
    :param dts_luminance: Dict(patch_id: luminance in cd/m^2)
    :param ambient_luminance: luminance of background in cd/m^2
    :param reflectance_std: perturbation caused by the non-uniformity of reflectance
    :param image_size: simulated map size
    :return:
    """
    dts_luminance_map = ambient_luminance * np.ones(image_size, dtype=np.float32)
    patch_size = min(image_size) // 30
    chart_size = patch_size * 6
    chart_centers = [[0.15, 0.2],
                     [0.5, 0.2],
                     [0.85, 0.2],
                     [0.15, 0.8],
                     [0.5, 0.8],
                     [0.85, 0.8]]  # coordinates of chart centers w.r.t. image, in [x, y] format

    for chart_id in range(6):
        chart = np.zeros((chart_size, chart_size), dtype=np.float32)
        chart_x0 = int(image_size[1] * chart_centers[chart_id][0] - chart_size // 2)
        chart_y0 = int(image_size[0] * chart_centers[chart_id][1] - chart_size // 2)
        for patch_id in range(36):
            luminance_cur_patch = dts_luminance[chart_id * 36 + patch_id]
            luminance_map_cur_patch = luminance_cur_patch * np.ones((patch_size, patch_size))

            # small perturbation caused by the non-uniformity of reflectance
            reflectance_noise = np.random.normal(0, reflectance_std, size=luminance_map_cur_patch.shape)
            luminance_map_cur_patch = luminance_cur_patch * np.clip(
                luminance_map_cur_patch / luminance_cur_patch + reflectance_noise, 0, None

            )
            row, col = patch_id % 6, patch_id // 6
            patch_x0 = col * patch_size
            patch_y0 = row * patch_size
            chart[patch_y0:patch_y0+patch_size, patch_x0:patch_x0+patch_size] = luminance_map_cur_patch
        dts_luminance_map[chart_y0: chart_y0+chart_size, chart_x0:chart_x0+chart_size] = chart

    return dts_luminance_map


def generate_checkerboard_luminance_map(bright_checker_luminance, dark_checker_luminance,
                                        reflectance_std=0.001, checker_size=128):
    """
    Generate luminance map with checkerboard patterns, as used in
    Detection Probabilities: Performance Prediction for Sensors of Autonomous Vehicles, Marc Geese
    and Implementierung von CDP, Lukas Ebbert
    :param bright_checker_luminance: luminance of brighter checker in cd/m^2
    :param dark_checker_luminance: luminance of darker checker in cd/m^2
    :param reflectance_std: perturbation caused by the non-uniformity of reflectance
    :param checker_size: size of each checker in px
    :return:
    """
    num_checkers = 20
    bright_checker = bright_checker_luminance * np.ones((checker_size, checker_size))
    dark_checker = dark_checker_luminance * np.ones((checker_size, checker_size))
    quad = np.block([
        [dark_checker, bright_checker],
        [bright_checker, dark_checker]
    ])
    checkerboard_luminance_map = np.tile(quad, (num_checkers // 2, num_checkers // 2))

    reflectance_noise = np.random.normal(0, reflectance_std, size=checkerboard_luminance_map.shape)
    checkerboard_luminance_map = checkerboard_luminance_map * np.clip(
        checkerboard_luminance_map / bright_checker_luminance + reflectance_noise, 0, None
    )

    return checkerboard_luminance_map


def visualize(image, vmax=None, vmin=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(image, cmap='gray', norm=Normalize(vmin=vmax, vmax=vmin, clip=True))
    fig.colorbar(im, ax=ax)
    plt.show(block=False)


def test():
    max_luminance = 40000

    dts_luminance_map = generate_dts_luminance_map(
        generate_dts_luminance(max_luminance=max_luminance)
    )
    visualize(dts_luminance_map)

    checkerboard_luminance_map = generate_checkerboard_luminance_map(max_luminance, max_luminance / 2)
    visualize(checkerboard_luminance_map)


if __name__ == '__main__':
    test()
