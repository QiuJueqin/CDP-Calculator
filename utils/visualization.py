import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle


def plot_rois(image, roi_boxes):
    """
    :param image:
    :param roi_boxes: dict(patch_id: np.ndarray(4,) or None)
    :return:
    """
    fig, ax = plt.subplots()
    fig.canvas.set_window_title('My Window Title')
    ax.imshow(image.astype(np.float32) / image.max())
    for patch_id, box in roi_boxes.items():
        if box is None:
            continue
        ax.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                               fc='None', ec='r', lw=3))
        plt.text((box[0] + box[2]) // 2, (box[1] + box[3]) // 2, str(patch_id),
                 horizontalalignment='center', verticalalignment='center'
                 )
    plt.show()


def plot_luminance(pixel_values, luminance, save_dir):
    fig, ax = plt.subplots()
    fig.canvas.set_window_title('Close figure to continue')
    ax.grid(which='major', linestyle='-', linewidth=0.5)
    ax.grid(which='minor', linestyle='--', linewidth=0.5)
    ax.scatter(luminance, pixel_values, s=12)
    ax.set_xscale('log')
    plt.xlabel('Luminance ($cd/m^2$)'), plt.ylabel('Pixel Value')
    if save_dir:
        save_path = os.path.join(save_dir, 'luminance_vs_pixel_values.pdf')
        plt.savefig(save_path, format='pdf')
        plt.close(fig)
    else:
        plt.show()


def visualize_cdp_2d(input_contrast, input_luminance, cdps, save_dir):
    # a figure for all input contrasts
    fig0, ax0 = plt.subplots(figsize=(8, 6), dpi=150)
    fig0.canvas.set_window_title('Close figure to continue')
    ax0.grid(which='major', linestyle='-', linewidth=0.5)
    ax0.grid(which='minor', linestyle='--', linewidth=0.5)
    for c in np.unique(input_contrast):
        idx = input_contrast == c
        ax0.scatter(input_luminance[idx], cdps[idx], label='C = {:.0f}%'.format(100*c))

        # a individual figure for current specific input contrast
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        fig.canvas.set_window_title('Close figure to continue')
        ax.grid(which='major', linestyle='-', linewidth=0.5)
        ax.grid(which='minor', linestyle='--', linewidth=0.5)
        ax.set_title('Contrast {:.0f}%'.format(100 * c))
        ax.scatter(input_luminance[idx], cdps[idx])
        ax.set_xscale('log')
        ax.set_xlabel('Luminance ($cd/m^2$)'), ax.set_ylabel('CDP')
        if save_dir:
            save_path = os.path.join(save_dir, 'cdp_contrast_{:.0f}%.pdf'.format(100*c))
            plt.savefig(save_path, format='pdf')
            plt.close(fig)

    ax0.set_xscale('log')
    ax0.set_xlabel('Luminance ($cd/m^2$)'), ax0.set_ylabel('CDP')
    fig0.legend(loc='upper left')
    if save_dir:
        contrasts_str = '_'.join(['{:.0f}%'.format(100*c) for c in sorted(np.unique(input_contrast))])
        save_path = os.path.join(save_dir, 'cdp_contrasts_{}.pdf'.format(contrasts_str))
        plt.savefig(save_path, format='pdf')
        plt.close(fig0)
        print('Saved figures to {}'.format(save_dir))
    else:
        plt.show()


def visualize_cdp_3d(input_contrast, input_luminance, cdps, save_dir):
    cmap = cm.rainbow(np.linspace(0, 1, 256))[:, :3]
    colors = cmap[(255 * np.array(cdps)).astype(np.uint16), :]

    fig = plt.figure(figsize=(8, 6), dpi=150)
    fig.canvas.set_window_title('Close figure to continue')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(input_contrast, np.log10(input_luminance), cdps,
               s=3, c=colors, edgecolors='face', alpha=1)
    ax.set_xlabel('Input Contrast'), ax.set_ylabel('Log Luminance (cd/m$^2$)'), ax.set_zlabel('CDP')
    ax.set_xlim(0, 1), ax.set_zlim(0, 1)
    ax.view_init(45, -40), plt.draw()
    ax.invert_xaxis()
    if save_dir:
        save_path = os.path.join(save_dir, 'cdp_all.pdf')
        plt.savefig(save_path, format='pdf')
        plt.close(fig)

    fig = plt.figure(figsize=(8, 6), dpi=150)
    fig.canvas.set_window_title('Close figure to continue')
    ax = fig.add_subplot(111)
    ax.grid(which='major', linestyle='-', linewidth=0.5)
    ax.grid(which='minor', linestyle='--', linewidth=0.5)
    ax.scatter(input_luminance, input_contrast, s=2, c=colors, edgecolors='face', alpha=1)
    ax.set_xscale('log')
    ax.set_xlabel('Luminance (cd/m$^2$)'), ax.set_ylabel('Contrast')
    ax.set_ylim(0, 1)
    if save_dir:
        save_path = os.path.join(save_dir, 'cdp_all_2d.pdf')
        plt.savefig(save_path, format='pdf')
        plt.close(fig)
        print('Saved figures to {}'.format(save_dir))
    else:
        plt.show()
