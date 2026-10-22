import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import aerosandbox as asb

def load_airfoil_asb(name):
    airfoil = asb.Airfoil(name)
    coords = airfoil.coordinates
    return coords[:, 0], coords[:, 1]


def plot_rotor_blade_3d_airfoil(
    airfoil_name,
    root_chord,
    tip_chord,
    hub_radius,
    blade_length,
    twist_root_deg,
    twist_tip_deg,
    num_stations=25
):
    # Load OA212 airfoil
    x_af, z_af = load_airfoil_asb(airfoil_name)

    # Shift airfoil so quarter-chord is rotation axis
    x_af = -x_af + 0.25

    # Spanwise stations
    r = np.linspace(hub_radius, hub_radius + blade_length, num_stations)
    chord = np.linspace(root_chord, tip_chord, num_stations)
    twist = np.deg2rad(np.linspace(twist_root_deg, twist_tip_deg, num_stations))

    X, Y, Z = [], [], []

    for i in range(num_stations):
        # Scale airfoil by local chord
        y_local = x_af * chord[i]
        z_local = z_af * chord[i]

        # Apply twist (rotation about radial axis)
        y_rot = y_local * np.cos(twist[i]) - z_local * np.sin(twist[i])
        z_rot = y_local * np.sin(twist[i]) + z_local * np.cos(twist[i])

        # Spanwise position
        x_rot = np.ones_like(y_rot) * r[i]

        X.append(x_rot)
        Y.append(y_rot)
        Z.append(z_rot)

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    # -----------------------------
    # Plot
    # -----------------------------
    fig = plt.figure(figsize=(11, 4))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X, Y, Z,
        color="lightsteelblue",
        edgecolor="k",
        linewidth=0.25,
        alpha=0.95
    )

    # Hub reference
    ax.plot([0, hub_radius], [0, 0], [0, 0], "k", linewidth=3)

    ax.set_xlabel("Radial Direction (m)")
    ax.set_ylabel("Chordwise Direction (m)")
    ax.set_zlabel("Thickness Direction (m)")
    ax.set_title("3D Rotor Blade with OA212 Airfoil")

    ax.set_box_aspect([blade_length, root_chord, tip_chord])
    plt.tight_layout()
    # plt.show()


# # -----------------------------
# # Example usage
# # -----------------------------
# plot_rotor_blade_3d_airfoil(
#     airfoil_name="oa212",
#     root_chord=0.45,
#     tip_chord=0.20,
#     hub_radius=0.15,
#     blade_length=2.0,
#     twist_root_deg=12,
#     twist_tip_deg=2
# )
