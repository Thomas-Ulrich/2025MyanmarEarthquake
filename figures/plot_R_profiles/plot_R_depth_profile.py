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
    parser.add_argument(
        "--depths",
        nargs="+",
        help="depth at which profile is made",
        default=[4e3],
        type=float,
    )
    args = parser.parse_args()

    fault_yaml_files = args.fault_yaml_files
    depths = args.depths

    n_files = len(fault_yaml_files)
    n_depths = len(depths)
    fig, axes = plt.subplots(
        n_files, n_depths, figsize=(7 * n_depths, 3 * n_files), sharex=True
    )
    # Ensure axes is always a 2D array: shape (n_files, n_depths)
    if n_files == 1 and n_depths == 1:
        axes = np.array([[axes]])
    elif n_files == 1:
        axes = axes[np.newaxis, :]
    elif n_depths == 1:
        axes = axes[:, np.newaxis]

    for i, fault_yaml_file in enumerate(fault_yaml_files):
        for j, depth in enumerate(depths):
            ax = axes[i, j]

            n = 550 * 4
            cell_centers = np.zeros((n, 3))
            cell_centers[:, 1] = np.linspace(-450e3, 100e3, n)
            y = cell_centers[:, 1] / 1e3  # km
            cell_centers[:, 2] = -depth
            regions = np.full((n,), 3)

            out = easi.evaluate_model(
                cell_centers,
                regions,
                ["mu_s", "mu_d", "cohesion", "d_c", "T_s", "T_d", "T_n"],
                fault_yaml_file,
            )

            out["tau"] = np.sqrt(out["T_s"] ** 2 + out["T_d"] ** 2) / 1e6
            out["pos_pn0"] = np.maximum(0, -out["T_n"]) / 1e6

            ax.plot(y, out["mu_d"] * out["pos_pn0"], label="Dynamic strength")
            ax.plot(y, out["tau"], label="Shear stress")
            ax.plot(y, out["mu_s"] * out["pos_pn0"], label="Static strength")

            # Axis titles
            fault_name = os.path.splitext(os.path.basename(fault_yaml_file))[0]
            sigma_n = float(fault_name.split("sn")[-1])
            depth_km = depth / 1e3
            ax.set_title(rf"$\sigma_n$ = {sigma_n} MPa, depth = {depth_km:.1f} km")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(direction="out")

            if j == 0:
                ax.set_ylabel("Stress (MPa)")
            if i == n_files - 1:
                ax.set_xlabel("Distance along fault (km)")

        # Shared y-label

    # Common legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for legend
    fn = "R_profile_row.pdf"
    plt.savefig(fn, dpi=200, bbox_inches="tight")
    full_path = os.path.abspath(fn)
    print(f"full path: {full_path}")
