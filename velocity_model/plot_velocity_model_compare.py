import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import matplotlib.cm as cm

def build_step_profile(values, depth_top, depth_bottom):
    x_step = []
    y_step = []
    for i in range(len(values)):
        x_step.extend([values[i], values[i]])
        y_step.extend([depth_top[i], depth_bottom[i]])
    return x_step, y_step

def plot_profile(ax, values, depth_top, depth_bottom, color, xlabel, label=None):
    x_step, y_step = build_step_profile(values, depth_top, depth_bottom)
    ax.plot(x_step, y_step, color=color, label=label)
    ax.set_ylabel("Depth (km)")
    ax.set_xlabel(xlabel)
    ax.grid(True)

def load_model_data(filename):
    data = np.loadtxt(filename, skiprows=1)
    depth_top = data[:, 0]
    depth_bottom = np.append(depth_top[1:], depth_top[-1] + 1.0)  # extend last layer
    return {
        "depth_top": depth_top,
        "depth_bottom": depth_bottom,
        "p_vel": data[:, 1],
        "s_vel": data[:, 2],
        "dens": data[:, 3],
    }

def main():
    parser = argparse.ArgumentParser(description="Plot one or more velocity models.")
    parser.add_argument("files", nargs='+', help="List of velocity model files")
    args = parser.parse_args()

    # Prepare plot
    fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharey=True)

    # Color map for multiple models
    colors = cm.get_cmap('tab10', len(args.files))

    for idx, fn in enumerate(args.files):
        model = load_model_data(fn)
        label = os.path.splitext(os.path.basename(fn))[0]
        color = colors(idx)

        plot_profile(axs[0], model['p_vel'], model['depth_top'], model['depth_bottom'], color, "P-wave velocity (km/s)", label)
        plot_profile(axs[1], model['s_vel'], model['depth_top'], model['depth_bottom'], color, "S-wave velocity (km/s)")
        plot_profile(axs[2], model['dens'], model['depth_top'], model['depth_bottom'], color, "density (g/cm3)")

    axs[0].legend()
    for ax in axs:
        ax.invert_yaxis()

    fig.suptitle("Velocity Model Profile")
    plt.tight_layout()

    # Output filename
    if len(args.files) == 1:
        fnout = os.path.splitext(args.files[0])[0] + ".png"
    else:
        fnout = "velocity_models_comparison.png"

    plt.savefig(fnout)
    print(f"done generating {fnout}")
    # plt.show()

if __name__ == "__main__":
    main()

