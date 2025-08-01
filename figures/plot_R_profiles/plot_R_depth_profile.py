import numpy as np
import argparse
import matplotlib.pyplot as plt  # Added missing import
import easi  # Assumes 'easi' is properly installed and the YAML file is valid
import os
import matplotlib

ps = 12
matplotlib.rcParams.update({"font.size": ps})
plt.rcParams["font.family"] = "sans"
matplotlib.rc("xtick", labelsize=ps)
matplotlib.rc("ytick", labelsize=ps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show a profile of shear stress")
    parser.add_argument("fault_yaml_files", nargs="+", help="List of EASI YAML files")
    args = parser.parse_args()

    fault_yaml_files = args.fault_yaml_files
    n_files = len(fault_yaml_files)

    n = 550 * 4  # Number of samples

    # Generate coordinates for evaluation
    cell_centers = np.zeros((n, 3))
    cell_centers[:, 1] = np.linspace(-450e3, 100e3, n)  # y from 0 to 200 km
    y = cell_centers[:, 1] / 1e3
    cell_centers[:, 2] = -4e3  # depth = 6 km

    regions = np.full((n,), 3)

    # Create subplots: one row, multiple columns
    fig, axes = plt.subplots(n_files, 1, figsize=(10, 4 * n_files), sharex=True)

    if n_files == 1:
        axes = [axes]  # Ensure axes is iterable if only one subplot

    for ax, fault_yaml_file in zip(axes, fault_yaml_files):
        print(ax, fault_yaml_file)
        # Evaluate model
        out = easi.evaluate_model(
            cell_centers,
            regions,
            ["mu_s", "mu_d", "cohesion", "d_c", "T_s", "T_d", "T_n"],
            fault_yaml_file,
        )

        # Compute stress quantities
        out["tau"] = np.sqrt(out["T_s"] ** 2 + out["T_d"] ** 2) / 1e6  # MPa
        out["pos_pn0"] = np.maximum(0, -out["T_n"]) / 1e6  # MPa

        # Plot
        ax.plot(y, out["mu_d"] * out["pos_pn0"], label="Dynamic strength")
        ax.plot(y, out["tau"], label="Shear stress")
        ax.plot(y, out["mu_s"] * out["pos_pn0"], label="Static strength")

        # Format plot
        name = os.path.splitext(os.path.basename(fault_yaml_file))[0]
        sigma_n = name.split("sn")[-1]
        ax.set_title(rf"$\sigma_n$ = {sigma_n} MPa")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")
        ax.set_ylabel("Fault stress and strength (MPa)")

    axes[-1].set_xlabel("Distance (km)")

    # Shared y-label

    # Common legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for legend
    fn = "R_profile_row.pdf"
    plt.savefig(fn, dpi=200, bbox_inches="tight")
    full_path = os.path.abspath(fn)
    print(f"full path: {full_path}")
