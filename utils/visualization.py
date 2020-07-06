import os
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_luminance(measurements, luminance):
    fig, ax = plt.subplots()
    ax.grid(which='major', linestyle='-', linewidth=0.5)
    ax.grid(which='minor', linestyle='--', linewidth=0.5)
    ax.scatter(measurements, luminance, s=12)
    ax.set_yscale('log')
    plt.xlabel('measurement')
    plt.ylabel('luminance ($cd/m^2$)')
    plt.show()


def visualize_cdp_2d(input_contrast, input_luminance, cdps, save_dir):
    # a figure for all input contrasts
    fig0, ax0 = plt.subplots(figsize=(8, 6), dpi=150)
    for c in np.unique(input_contrast):
        idx = input_contrast == c
        ax0.scatter(input_luminance[idx], cdps[idx], label='c = {:.0f}%'.format(100 * c))

        # a individual figure for current specific input contrast
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        ax.set_title('contrast {:.0f}%'.format(100 * c))
        ax.scatter(input_luminance[idx], cdps[idx])
        ax.set_xscale('log')
        ax.set_xlabel('luminance ($cd/m^2$)'), ax0.set_ylabel('cdp')
        save_path = os.path.join(save_dir if save_dir else tempfile.gettempdir(),
                                 'cdp_contrast_{:.0f}%.pdf'.format(100*c))
        plt.savefig(save_path, format='pdf')
        plt.close(fig)

    ax0.set_xscale('log')
    ax0.set_xlabel('luminance ($cd/m^2$)'), ax0.set_ylabel('cdp')
    fig0.legend()
    save_path = os.path.join(save_dir if save_dir else tempfile.gettempdir(), 'cdp_vs_contrasts.pdf')
    plt.savefig(save_path, format='pdf')
    plt.close(fig0)
    print('saved figures to {}'.format(save_dir))


def visualize_cdp_3d(input_contrast, input_luminance, cdps, save_dir):
    cmap = cm.rainbow(np.linspace(0, 1, 256))[:, :3]
    colors = cmap[(255 * np.array(cdps)).astype(np.uint16), :]

    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(input_contrast, np.log10(input_luminance), cdps,
               s=3, c=colors, edgecolors='face', alpha=1)
    ax.set_xlabel('Input Contrast'), ax.set_ylabel('Log Luminance (cd/m$^2$)'), ax.set_zlabel('CDP')
    ax.set_xlim(0, 1), ax.set_zlim(0, 1)
    ax.view_init(45, -40), plt.draw()
    ax.invert_xaxis()

    save_path = os.path.join(save_dir if save_dir else tempfile.gettempdir(), 'cdp_vs_contrasts_and_luminance.pdf')
    plt.savefig(save_path, format='pdf')
    plt.close(fig)
    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(111)
    ax.grid(which='major', linestyle='-', linewidth=0.5)
    ax.grid(which='minor', linestyle='--', linewidth=0.5)
    ax.scatter(input_luminance, input_contrast, s=2, c=colors, edgecolors='face', alpha=1)
    ax.set_xscale('log')
    ax.set_xlabel('luminance (cd/m$^2$)'), ax.set_ylabel('contrast')
    ax.set_ylim(0, 1)
    save_path = os.path.join(save_dir if save_dir else tempfile.gettempdir(), 'cdp_vs_contrasts_and_luminance_2d.pdf')
    plt.savefig(save_path, format='pdf')
    plt.close(fig)
    print('saved figures to {}'.format(save_dir))
