import numpy as np
import matplotlib.pyplot as plt
import os

# Load velocity model from file
#fn = "velocity_model_Xiang_Xiaoyu.txt"
fn = "velocity_model.txt"
model_data = np.loadtxt(fn, skiprows=1)


depth_top = model_data[:, 0]
depth_bottom = np.append(depth_top[1:], depth_top[-1] + 1)  # Extend last layer by 1 km
p_vel = model_data[:, 1]
s_vel = model_data[:, 2]
dens = model_data[:, 3]


def build_step_profile(values, depth_top, depth_bottom):
    x_step = []
    y_step = []
    for i in range(len(values)):
        x_step.extend([values[i], values[i]])
        y_step.extend([depth_top[i], depth_bottom[i]])
    return x_step, y_step


def plot_profile(ax, x, depth_top, depth_bottom, label, color):
    for i in range(len(depth_top)):
        ax.plot([x[i], x[i]], [depth_top[i], depth_bottom[i]], color=color)
    x_step, y_step = build_step_profile(x, depth_top, depth_bottom)
    ax.plot(x_step, y_step, color=color)

    ax.set_ylabel("Depth (km)")
    ax.invert_yaxis()
    ax.set_xlabel(label)
    ax.grid(True)


fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharey=True)

plot_profile(axs[0], p_vel, depth_top, depth_bottom, "P-wave Velocity (km/s)", "blue")
plot_profile(axs[1], s_vel, depth_top, depth_bottom, "S-wave Velocity (km/s)", "green")
plot_profile(axs[2], dens, depth_top, depth_bottom, "Density (g/cm³)", "red")

fig.suptitle("Velocity Model Profile")
plt.tight_layout()
fnout = os.path.splitext(fn)[0] + ".png"
plt.savefig(fnout)
print(f"done generating {fnout}")
#plt.show()
