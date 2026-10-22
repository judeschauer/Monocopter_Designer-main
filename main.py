import numpy as np
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from bemt import bemt2, airfoil_model, AirfoilParams
from apc_prop import get_motor_thrust, get_thrusts_at_rpm, get_motor_dict, get_prop_perf_dict
from structure import  WingStructuralParams, structural_analysis
import neuralfoil as nf
from visualization_3D import plot_rotor_blade_3d_airfoil

@dataclass
class MotorParams:
    motor_kv: float
    internal_resistance: float
    K_q: float
    K_e: float
    K_1: float
    K_2: float

# -----------------------------
# User Inputs
# -----------------------------
SCALE = 3.0 # Scale reduction factor for the rotor size
R = 110 * 0.0254 / SCALE# radius
B = 3 # no. of blades
rho = 1.221 # density
A = np.pi * R**2

airfoil_params = AirfoilParams(
    airfoil_name="naca6412",  # path to file
    Re=0.7e6/SCALE,
    Mach=0.0,
    alpha_min_deg=-13,
    alpha_max_deg=15,
    alpha_step_deg=0.2,
    alpha_stall_high = np.deg2rad(12.0),
    alpha_stall_low = np.deg2rad(-10.0),
)

spar_root_ID = 1.375 * 0.0254 / SCALE# Spar root inner diameter
spar_root_OD = 1.455 * 0.0254 / SCALE# Spar root outer diameter
# spar_root_ID = 1.0 * 0.0254 # Spar root inner diameter
# spar_root_OD = 1.06 * 0.0254 # Spar root outer diameter

# spar_middle_ID = spar_root_ID
# spar_middle_OD = spar_root_OD
# spar_middle_ID = 1.0 * 0.0254 # Spar root inner diameter
# spar_middle_OD = 1.125 * 0.0254 # Spar root outer diameter
# spar_middle_ID = 0.75 * 0.0254 # Spar root inner diameter
# spar_middle_OD = 0.875 * 0.0254 # Spar root outer diameter
spar_middle_ID = 1.125 * 0.0254 / SCALE# Spar root inner diameter
spar_middle_OD = 1.375 * 0.0254 / SCALE# Spar root outer diameter
spar_middle_start = 0.2 * R

# spar_tip_ID = spar_root_ID
# spar_tip_OD = spar_root_OD
# spar_tip_ID = 0.625 * 0.0254 # Spar tip inner diameter
# spar_tip_OD = 0.81 * 0.0254 # Spar tip outer diameter
# spar_tip_ID = 0.5 * 0.0254 # Spar tip inner diameter
# spar_tip_OD = 0.58 * 0.0254 # Spar tip outer diameter
# spar_tip_ID = 0.5 * 0.0254 # Spar tip inner diameter
# spar_tip_OD = 0.625 * 0.0254 # Spar tip outer diameter
# spar_tip_ID = 0.625 * 0.0254 # Spar tip inner diameter
# spar_tip_OD = 0.75 * 0.0254 # Spar tip outer diameter
spar_tip_ID = 0.625 * 0.0254 / SCALE# Spar tip inner diameter
spar_tip_OD = 0.81 * 0.0254 / SCALE# Spar tip outer diameter
spar_tip_start = 0.45 * R

min_chord = spar_tip_OD / 0.12 # Max chord to avoid interference

chord_tip = 0.2 / SCALE
chord_root = 0.4 / SCALE
chord_const_root_end = 0.2 * R

pitch_root = 19
pitch_tip = 7

precone_angle_deg = 8
R_eff = R * np.cos(np.deg2rad(precone_angle_deg))

print(f"root_spar_max_diameter = {(0.12/0.0254) * (chord_root + ((spar_middle_start-chord_const_root_end)/(R-chord_const_root_end))*(chord_tip - chord_root)) - 0.131 : .3f} inches" )
print(f"middle_spar_max_diameter = {(0.12/0.0254) * (chord_root + ((spar_tip_start-chord_const_root_end)/(R-chord_const_root_end))*(chord_tip - chord_root)) - 0.131 : .3f} inches" )
print(f"tip_spar_max_diameter = {(0.12/0.0254) * chord_tip - 0.131 : .3f} inches")

mean_solidity = B * (0.5 * (chord_root + chord_tip) * 0.8*R) / (np.pi * R*R)
print(f"Mean solidity is {mean_solidity:.3f}")

# rpm_range = np.linspace(50, 250, 20) # RPM sweep for BEMT
# climb_speed = 0.0
monocopter_rpm = 200 * SCALE
thrust_target = 62.5 * 9.81 / (SCALE**2) # Target thrust in N
collective_pitch = 0
climb_speed_range = [0, 4]
zero_speed_index = climb_speed_range.index(0)

n_sections = 80 # No. of blade sections

######## Drive Motor Params ###########3###
n_prop_blades = 3

apc_prop_selection = "PER3_5x4E-3"
apc_prop_blades = 3

# apc_prop_selection = "PER3_6x4"
# apc_prop_selection = "PER3_6x5"
# apc_prop_blades = 2

# apc_prop_selection = "PER3_20x13E"
# apc_prop_blades = 2

R_motor = np.linspace(0.3, 1.0, 8) # Locations of drive motor as a ratio of R
# motor_rpm_range = np.linspace(2000, 5000, 10)
motor_tilt_deg = 10 # Motor tilt angle in degrees

prop_km = 10
motor_kv = 2200 # motor kv rating
# motor_kv = 240 # motor kv rating
torque_data_point = 3.519
current_data_point = 81.58
rpm_data_point = 8440
voltage_data_point = 47.32
internal_resistance = 0.1
K_q = torque_data_point / current_data_point  # Nm/A
K_e = (voltage_data_point - internal_resistance * current_data_point) / rpm_data_point
K_1 = K_e / voltage_data_point
K_2 = internal_resistance * prop_km / (K_q * voltage_data_point)

motor_params = MotorParams(
    motor_kv = motor_kv,
    internal_resistance = internal_resistance,
    K_q = K_q,
    K_e = K_e,
    K_1 = K_1,
    K_2 = K_2
)

N_series = 4
# N_series = 12
max_cell_voltage = 4.2
min_cell_voltage = 2.8
rpm_reduction_factor = 0.7 # RPM reduction factor from motor to propeller
motor_rpm_at_max_v = int(motor_kv * N_series * max_cell_voltage * rpm_reduction_factor)
motor_rpm_at_min_v = int(motor_kv * N_series * min_cell_voltage * rpm_reduction_factor)
motor_rpm_at_max_v_int = math.ceil(motor_rpm_at_max_v/1000-1)*1000
motor_rpm_at_min_v_int = math.floor(motor_rpm_at_min_v/1000-1)*1000
motor_rpm_range = np.linspace(motor_rpm_at_min_v_int, motor_rpm_at_max_v_int, int((motor_rpm_at_max_v_int - motor_rpm_at_min_v_int)/1000)+1)
# -----------------------------
# Structural parameters
glass_fiber_areal_weight = 0.05 # kg/m2
glass_fiber_thickness = 0.08 * 0.001 # m
glass_fiber_density = glass_fiber_areal_weight / glass_fiber_thickness # kg/m3
epoxy_density = 1200 # kg/m3
structural_params = WingStructuralParams(
    No_Blades = B,
    spar_root_ID = spar_root_ID,
    spar_root_OD = spar_root_OD,
    spar_middle_ID = spar_middle_ID,
    spar_middle_OD = spar_middle_OD,
    spar_middle_start = spar_middle_start,
    spar_tip_ID = spar_tip_ID,
    spar_tip_OD = spar_tip_OD,
    spar_tip_start = spar_tip_start,
    spar_length = R,
    # spar_material_density = 0.044 * 27679.9, # Braided tube
    spar_material_density = 0.057 * 27679.9, # Twill tube
    spar_material_strength = 570 * 10**6,
    chord_root = chord_root,
    chord_tip = chord_tip,
    wing_span = 0.8*R,
    # wing_material_density = 16.0,
    wing_material_density = 24.03,
    # airfoil_area_coeff = 0.685 * airfoil_params.airfoil_thickness
    airfoil_area_coeff = 0.081605,
    airfoil_name = airfoil_params.airfoil_name,
    skin_thickness = 2.0*glass_fiber_thickness, # m
    skin_density = 0.5*glass_fiber_density + 0.5*epoxy_density # kg/m3,
)

# -----------------------------
# Climb speed Sweep
# -----------------------------
thrusts_N, collective_pitches, powers_W, foms, axial_speeds, pitch_moments, FM_values, kappas  = [], [], [], [], [], [], [], []
leadlag_torques_root, leadlag_torques_middle, leadlag_torques_tip, flap_torques_root, flap_torques_middle, flap_torques_tip = [], [], [], [], [], []
# N = len(rpm_range)
N = len(climb_speed_range)
angle_of_attack_array = np.zeros((N, n_sections-1))
dT_r_array = np.zeros((N, n_sections-1))
dF_r_array = np.zeros((N, n_sections-1))
dLD_r_array = np.zeros((N, n_sections-1))
dM_r_array = np.zeros((N, n_sections-1))
Re_array = np.zeros((N, n_sections-1))

T_motors_vs_climb = np.zeros((N, len(R_motor)))
# for rpm in rpm_range:
omega = monocopter_rpm * 2 * np.pi / 60
for climb_speed in climb_speed_range:

    print(f"\n Climb speed = {climb_speed:.2f}")
    thrust_err = 1
    while (np.abs(thrust_err) > 0.01):
        eff_pitch_root = pitch_root + collective_pitch
        eff_pitch_tip = pitch_tip + collective_pitch
        thrust, leadlag_torques, flap_torques, power, aoa_r, dT_r, dF_r, dLD_r, dM_r, pitch_moment, Re_r, FM_value, kappa, stalled = bemt2(
            omega,
            climb_speed,
            R_eff,
            B,
            rho,
            chord_root,
            chord_const_root_end,
            chord_tip,
            eff_pitch_root,
            eff_pitch_tip,
            airfoil_params,
            structural_params,
            n_sections
        )
        thrust_err = thrust_target - thrust
        collective_pitch += 8*thrust_err/thrust_target
        print(f"Thrust target = {thrust_target:.3f}     Thrust = {thrust:.3f}      collective_pitch = {collective_pitch:.3f} deg")
    print(f"Power = {power/1000.0:.3f} kW")

    thrusts_N.append(thrust)
    collective_pitches.append(collective_pitch)
    leadlag_torques_root.append(leadlag_torques[0])
    leadlag_torques_middle.append(leadlag_torques[1])
    leadlag_torques_tip.append(leadlag_torques[2])
    flap_torques_root.append(flap_torques[0])
    flap_torques_middle.append(flap_torques[1])
    flap_torques_tip.append(flap_torques[2])
    powers_W.append(power)
    pitch_moments.append(pitch_moment)
    FM_values.append(FM_value)
    kappas.append(kappa)

    # i = rpm_range.tolist().index(rpm)
    i = climb_speed_range.index(climb_speed)

    angle_of_attack_array[i :] = aoa_r
    dT_r_array[i, :] = dT_r
    dF_r_array[i, :] = dF_r
    Re_array[i, :] = Re_r
    dLD_r_array[i, :] = dLD_r
    dM_r_array[i, :] = dM_r

    motor_tilt_deg = eff_pitch_root + R_motor * ( eff_pitch_tip - eff_pitch_root )
    T_motors_vs_climb[i, :] = (leadlag_torques[0]/B)/(R_motor*R*np.cos(np.deg2rad(motor_tilt_deg)))

stress_list, fos_list, skin_weight, wing_weight, spar_weight, cgx, cgy, dcf_pitch_m_r, dm_r = structural_analysis(
        leadlag_torques_root,
        leadlag_torques_middle,
        leadlag_torques_tip,
        flap_torques_root,
        flap_torques_middle,
        flap_torques_tip,
        precone_angle_deg,
        omega,
        structural_params)

desired_motor_thrusts_dict, axial_speed_dict, p_motors_dict = {}, {}, {}
for r_motor in R_motor:

    T_motors, Axial_speeds = [], []
    for i in range(len(climb_speed_range)):
        climb_speed = climb_speed_range[i]
        torque = leadlag_torques_root[i]

        eff_pitch_root = pitch_root + collective_pitches[i]
        eff_pitch_tip = pitch_tip + collective_pitches[i]
        motor_tilt_deg = eff_pitch_root + r_motor * ( eff_pitch_tip - eff_pitch_root )

        T_motors.append((torque/B)/(r_motor*R*np.cos(np.deg2rad(motor_tilt_deg))))
        # Axial_speeds = [(r_motor*R)*(rpm*2*np.pi/60) for rpm in rpm_range]
        Axial_speeds.append((r_motor*R)*(monocopter_rpm*2*np.pi/60))
        # P_motors = [T_motor * Axial_speed * B for T_motor, Axial_speed in zip(T_motors, Axial_speeds)]

    axial_speed_dict[r_motor] = Axial_speeds
    # p_motors_dict[r_motor] = P_motors
    desired_motor_thrusts_dict[r_motor] = T_motors

# combined_motor_thrusts_dict = get_motor_dict(apc_prop_selection, motor_rpm_range, axial_speed_dict)

#### Background rpm, thrust, and axial speed plot #####
axial_speed_range = np.linspace(0, 1.5*R*(monocopter_rpm*2*np.pi/60), 100)
prop_performance_dict = get_prop_perf_dict(apc_prop_selection, motor_rpm_range, axial_speed_range)
######################################################

# Convert for plotting
thrusts_kgf = [T/9.81 for T in thrusts_N]
powers_kW = [P/1000.0 for P in powers_W]
leadlag_torques = [t*0.101971621 for t in leadlag_torques]  # Nm to kg-m
flap_torques = [t*0.101971621 for t in flap_torques]
pitch_moments_kgm = [t*0.101971621 for t in pitch_moments]

print(f"Max wing pitching moments is {-max(np.abs(pitch_moments_kgm)) * 100:.2f} kg-cm")
print(f"Centrifugal pitching moments is {-np.sum(dcf_pitch_m_r) * 0.101971621 * 100:.2f} kg-cm")

# motor_mass = 0.35 # kg, estimated motor mass
motor_mass = 0.5 # kg, with articulated hub
motor_location_x = 0.45*R
chord_at_motor = chord_root + ((motor_location_x-chord_const_root_end)/(R-chord_const_root_end))*(chord_tip - chord_root)
# print(f"chord at motor location is {chord_at_motor:.3f} m")
motor_location_y = chord_at_motor/4 + 2*26/1000 # Assume motor is located at 25% chord + 2*26 mm (for prop hub)
motor_cf = motor_mass * motor_location_x * omega**2
motor_cf_pitch_moment = motor_cf * np.sin(np.deg2rad(precone_angle_deg)) * motor_location_y
print(f"Motor location y is {motor_location_y:.3f} m")
print(f"Centrifugal pitching moment from motor mass is {-motor_cf_pitch_moment * 0.101971621 * 100:.2f} kg-cm")

motor_thrust = 4.0*9.81
mounting_angle_deg = 0
mounting_pitch_torque = -motor_thrust * np.sin(np.deg2rad(mounting_angle_deg)) * motor_location_y
print(f"Mounting pitch torque from motor thrust is {mounting_pitch_torque * 0.101971621 * 100:.2f} kg-cm")

motor_coriolis_pitch_moment = (258/10.1971621) # Assume 230 kg-cm of additional pitching moment from motor acceleration/coriolis effects

total_moment = motor_coriolis_pitch_moment - motor_cf_pitch_moment - np.sum(dcf_pitch_m_r) - max(np.abs(pitch_moments)) + mounting_pitch_torque
print("Total pitching moment including motor effects is {:.2f} kg-cm".format(total_moment * 0.101971621 * 100))

import matplotlib.pyplot as plt
# -------------------------------------------------
# Pitching Moment Distribution over Span
# -------------------------------------------------
r_cut_idx = int(0.2*len(dcf_pitch_m_r))
mot_idx = int(motor_location_x/R * len(dcf_pitch_m_r))
dr = 1.0/len(dcf_pitch_m_r)
section_moment = np.zeros(len(dcf_pitch_m_r))
for i in range(len(dcf_pitch_m_r)-1):
    if (i < r_cut_idx):
        section_moment[i] = -dcf_pitch_m_r[i]
    else:
        section_moment[i] = dM_r_array[0, i-r_cut_idx]*dr - dcf_pitch_m_r[i]
    # if (i == mot_idx):
    #     section_moment[i] += -motor_cf_pitch_moment
        # section_moment[i] += motor_coriolis_pitch_moment
# plot pitching moment distribution over span
span_positions = np.linspace(0, 1, len(section_moment))
span_wing_positions = np.linspace(0.2, 1, len(dM_r_array[0, :]))
plt.figure(figsize=(10, 6))
plt.plot(span_positions, section_moment*0.101971621*100, label='Total Pitching Moment (kg-cm)')
plt.plot(span_positions, -np.array(dcf_pitch_m_r)*0.101971621*100, label='Centrifugal Pitching Moment (kg-cm)', linestyle='--')
plt.plot(span_wing_positions, np.array(dM_r_array[0, :])*0.101971621*100*dr, label='Aerodynamic Pitching Moment (kg-cm)', linestyle='--')
plt.xlabel("Spanwise Position")
plt.ylabel("Pitching Moment (kg-cm)")
plt.title("Pitching Moment Distribution over Span")
plt.grid(True)
plt.legend()


# -------------------------------------------------
# Rotor Performance Plots (separate subplots)
# -------------------------------------------------
fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
# fig.suptitle("BEMT Performance vs RPM", fontsize=14)
# x_axis_data = rpm_range
x_axis_data = climb_speed_range
# # ----------------------
# # Thrust
# # ----------------------
# ax = axes[0]
# # ax.plot(rpm_range, thrusts_kgf, color="tab:blue")
# ax.plot(x_axis_data, thrusts_kgf, color="tab:blue")
# ax.set_ylabel("Thrust [kgf]")
# ax.set_ylim(0, max(thrusts_kgf) * 1.2)
# ax.grid(True)

# ----------------------
# Figure of Merit
# ----------------------
ax = axes[0]
ax.plot(x_axis_data, FM_values, color="tab:blue")
ax.set_ylabel("Figure of Merit [kgf]")
ax.grid(True)

# ----------------------
# Power
# ----------------------
ax = axes[1]
# ax.plot(rpm_range, powers_kW, color="tab:green")
ax.plot(x_axis_data, powers_kW, color="tab:green")
ax.set_ylabel("Power [kW]")
ax.grid(True)

# ----------------------
# Torque
# ----------------------
ax = axes[2]
ax.plot(x_axis_data, leadlag_torques_root, color="tab:red")
ax.set_ylabel("Lead-Lag Torque at root [kg-m]")
ax.grid(True)
ax = axes[3]
ax.plot(x_axis_data, flap_torques_root, color="tab:orange")
ax.set_ylabel("Flap Torque at root[kg-m]")
ax.grid(True)
ax = axes[4]
ax.plot(x_axis_data, pitch_moments_kgm, color="tab:purple")
ax.set_ylabel("Pitching Moment [kg-m]")
ax.grid(True)
# ax.legend(["Azimuthal Torque", "Axial Torque", "Pitching Moment"])

# # -----------------------
# # Angle of Attack
# # -----------------------
# ax = axes[3]
# for i in range(0, n_sections-1, 5): # skip every 5th section to reduce clutter
#     line, = ax.plot(x_axis_data, np.rad2deg(angle_of_attack_array[:, i]), label=f"Section {i+1}")
#     # Label at end of line
#     ax.text(
#         x_axis_data[-1],
#         np.rad2deg(angle_of_attack_array[-1, i]),
#         f"S{i+1}",
#         color=line.get_color(),
#         fontsize=9,
#         va="bottom",
#         ha="left"
#     )
# ax.set_ylabel("Angle of Attack [deg]")
# ax.set_xlabel("RPM")
# ax.grid(True)

# ----------------------
# Axial Speed
# ----------------------
# ax = axes[3]
# for r_motor, axial_speeds in axial_speed_dict.items():
#     line, = ax.plot(x_axis_data, axial_speeds, label=f"{r_motor} R_motor")
#     # Label at end of line
#     ax.text(
#         x_axis_data[-1],
#         axial_speeds[-1],
#         f"{r_motor} R",
#         color=line.get_color(),
#         fontsize=9,
#         va="bottom",
#         ha="left"
#     )
# ax.set_ylabel("Axial Speed [m/s]")
# ax.set_xlabel("RPM")
ax.set_xlabel("Climb speed [m/s]")
ax.grid(True)

# ----------------------
# Figure of Merit annotation
# ----------------------
axes[0].text(
    0.5, 1.05,
    f"Figure of Merit (FM) at hover = {FM_values[zero_speed_index]:.3f}",
    transform=axes[0].transAxes,
    ha="center", va="bottom",
    fontsize=12,
    bbox=dict(boxstyle="round", facecolor="white")
)

if stalled:
    axes[0].text(
        0.5, 1.00,
        "Warning: Stall occurred during BEMT calculation!",
        transform=axes[0].transAxes,
        ha="center", va="bottom",
        fontsize=10,
        color="red"
    )

plt.tight_layout(rect=[0, 0, 1, 0.96])
# Save figure as JPG
plt.savefig("Rotor Performance.jpg", dpi=300)  # dpi=resolution, quality=JPEG compression

# -------------------------
# Performance over Span at zero climb
# -------------------------
# New figure for Angle of Attack vs Span at max RPM
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Performance over rotor span at zero climb speed", fontsize=14)
span_positions = np.linspace(0.2, 1, n_sections-1)
aoa_ref = angle_of_attack_array[zero_speed_index, :]
ax.plot(span_positions, np.rad2deg(aoa_ref), color="tab:purple")
ax.set_ylabel("Angle of Attack [deg]", color="tab:purple")
ax.tick_params(axis='y', labelcolor="tab:purple")
ax.set_xlabel("Span Position normalized")
# ax.set_title(f"Angle of Attack Distribution")
ax.grid(True)

# # -------------------------
# # Normalized dT_r vs Span at ref
# # -------------------------
# dT_r_normalized = dT_r_array[zero_speed_index, :]/max(dT_r_array[zero_speed_index, :])
# ax2 = ax.twinx()  # Create a secondary y-axis
# ax2.plot(span_positions, dT_r_normalized, color="tab:orange", linestyle="--")
# ax2.set_ylabel("F_Z normalized", color="tab:orange")  # Add padding to separate labels
# ax2.tick_params(axis='y', labelcolor="tab:orange")
# ax2.grid(False)  # Disable grid for secondary y-axis

# -------------------------
# dT_r vs Span at ref
# -------------------------
ax2 = ax.twinx()  # Create a secondary y-axis
ax2.plot(span_positions, dT_r_array[zero_speed_index, :]/(B*9.81), color="tab:orange", linestyle="--")
ax2.set_ylabel("F_Z (kgf)", color="tab:orange")  # Add padding to separate labels
ax2.tick_params(axis='y', labelcolor="tab:orange")
ax2.grid(False)  # Disable grid for secondary y-axis

# # -------------------------
# # Normalized dF_r vs Span at ref
# # -------------------------
# dF_r_normalized = dF_r_array[zero_speed_index, :]/max(dF_r_array[zero_speed_index, :])
# ax3 = ax.twinx()  # Create a secondary y-axis on the right
# ax3.spines["right"].set_position(("outward", 60))  # Offset the third axis further to avoid overlap
# ax3.plot(span_positions, dF_r_normalized, color="tab:blue", linestyle="--")
# ax3.set_ylabel("-F_y normalized", color="tab:blue")  # Add padding to separate labels
# ax3.tick_params(axis='y', labelcolor="tab:blue")
# ax3.grid(False)  # Disable grid for secondary y-axis

# -------------------------
# dF_r vs Span at ref
# -------------------------
ax3 = ax.twinx()  # Create a secondary y-axis on the right
ax3.spines["right"].set_position(("outward", 60))  # Offset the third axis further to avoid overlap
ax3.plot(span_positions, dF_r_array[zero_speed_index, :]/(B*9.81), color="tab:blue", linestyle="--")
ax3.set_ylabel("-F_y (kgf)", color="tab:blue")  # Add padding to separate labels
ax3.tick_params(axis='y', labelcolor="tab:blue")
ax3.grid(False)  # Disable grid for secondary y-axis

# -------------------------
# dLD_r vs Span at ref
# -------------------------
ax4 = ax.twinx()  # Create a secondary y-axis on the right
ax4.spines["right"].set_position(("outward", 60*2))  # Offset the third axis further to avoid overlap
ax4.plot(span_positions, dLD_r_array[zero_speed_index, :], color="tab:red", linestyle="--")
ax4.set_ylabel("L/D", color="tab:red")  # Add padding to separate labels
ax4.tick_params(axis='y', labelcolor="tab:red")
ax4.grid(False)  # Disable grid for secondary y-axis

# -------------------------
# dM_r vs Span at ref
# -------------------------
ax5 = ax.twinx()  # Create a secondary y-axis on the right
ax5.spines["right"].set_position(("outward", 60*3))  # Offset the third axis further to avoid overlap
ax5.plot(span_positions, dM_r_array[zero_speed_index, :]*10.197162129779, color="tab:green", linestyle="--")
ax5.set_ylabel("Pitch Moment (kg-cm)", color="tab:green")  # Add padding to separate labels
ax5.tick_params(axis='y', labelcolor="tab:green")
ax5.grid(False)  # Disable grid for secondary y-axis

# # -------------------------
# # Reynolds Number Re_r vs Span at ref
# # -------------------------
# ax5 = ax.twinx()
# Re_r = Re_array[zero_speed_index, :]
# ax5.spines["right"].set_position(("outward", 60*3))  # Offset the fourth axis further to avoid overlap
# ax5.plot(span_positions, Re_r, color="tab:green", linestyle="--")
# ax5.set_ylabel("Reynold number", color="tab:green")
# ax5.tick_params(axis='y', labelcolor="tab:green")
# ax5.grid(False)  # Disable grid for the fourth y-axis

# Adjust layout to ensure the third axis is not cut off
fig.tight_layout()

# Save figure as JPG
plt.savefig("Performance over rotor span at zero climb speed.jpg", dpi=300)  # dpi=resolution,


# -------------------------
# Performance over Span at max climb speed
# -------------------------
# New figure for Angle of Attack vs Span at max RPM
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle(f"Performance over rotor span at max climb speed {climb_speed_range[-1]:.1f} m/s", fontsize=14)
span_positions = np.linspace(0.2, 1, n_sections-1)
aoa_ref = angle_of_attack_array[-1, :]
ax.plot(span_positions, np.rad2deg(aoa_ref), color="tab:purple")
ax.set_ylabel("Angle of Attack [deg]", color="tab:purple")
ax.tick_params(axis='y', labelcolor="tab:purple")
ax.set_xlabel("Span Position normalized")
# ax.set_title(f"Angle of Attack Distribution")
ax.grid(True)

# # -------------------------
# # Normalized dT_r vs Span at ref
# # -------------------------
# dT_r_normalized = dT_r_array[-1, :]/max(dT_r_array[-1, :])
# ax2 = ax.twinx()  # Create a secondary y-axis
# ax2.plot(span_positions, dT_r_normalized, color="tab:orange", linestyle="--")
# ax2.set_ylabel("F_Z normalized", color="tab:orange")  # Add padding to separate labels
# ax2.tick_params(axis='y', labelcolor="tab:orange")
# ax2.grid(False)  # Disable grid for secondary y-axis

# -------------------------
# dT_r vs Span at ref
# -------------------------
ax2 = ax.twinx()  # Create a secondary y-axis
ax2.plot(span_positions, dT_r_array[-1, :]/(B*9.81), color="tab:orange", linestyle="--")
ax2.set_ylabel("F_Z (kgf))", color="tab:orange")  # Add padding to separate labels
ax2.tick_params(axis='y', labelcolor="tab:orange")
ax2.grid(False)  # Disable grid for secondary y-axis

# # -------------------------
# # Normalized dF_r vs Span at ref
# # -------------------------
# dF_r_normalized = dF_r_array[-1, :]/max(dF_r_array[-1, :])
# ax3 = ax.twinx()  # Create a secondary y-axis on the right
# ax3.spines["right"].set_position(("outward", 60))  # Offset the third axis further to avoid overlap
# ax3.plot(span_positions, dF_r_normalized, color="tab:blue", linestyle="--")
# ax3.set_ylabel("-F_y normalized", color="tab:blue")  # Add padding to separate labels
# ax3.tick_params(axis='y', labelcolor="tab:blue")
# ax3.grid(False)  # Disable grid for secondary y-axis

# -------------------------
# dF_r vs Span at ref
# -------------------------
ax3 = ax.twinx()  # Create a secondary y-axis on the right
ax3.spines["right"].set_position(("outward", 60))  # Offset the third axis further to avoid overlap
ax3.plot(span_positions, dF_r_array[-1, :]/(B*9.81), color="tab:blue", linestyle="--")
ax3.set_ylabel("-F_y (kgf))", color="tab:blue")  # Add padding to separate labels
ax3.tick_params(axis='y', labelcolor="tab:blue")
ax3.grid(False)  # Disable grid for secondary y-axis

# -------------------------
# dLD_r vs Span at ref
# -------------------------
ax4 = ax.twinx()  # Create a secondary y-axis on the right
ax4.spines["right"].set_position(("outward", 60*2))  # Offset the third axis further to avoid overlap
ax4.plot(span_positions, dLD_r_array[-1, :], color="tab:red", linestyle="--")
ax4.set_ylabel("L/D", color="tab:red")  # Add padding to separate labels
ax4.tick_params(axis='y', labelcolor="tab:red")
ax4.grid(False)  # Disable grid for secondary y-axis

# -------------------------
# dM_r vs Span at ref
# -------------------------
ax5 = ax.twinx()  # Create a secondary y-axis on the right
ax5.spines["right"].set_position(("outward", 60*3))  # Offset the third axis further to avoid overlap
ax5.plot(span_positions, dM_r_array[-1, :]*10.197162129779, color="tab:green", linestyle="--")
ax5.set_ylabel("Pitch Moment (kg-cm)", color="tab:green")  # Add padding to separate labels
ax5.tick_params(axis='y', labelcolor="tab:green")
ax5.grid(False)  # Disable grid for secondary y-axis

# # -------------------------
# # Reynolds Number Re_r vs Span at ref
# # -------------------------
# ax5 = ax.twinx()
# Re_r = Re_array[-1, :]
# ax5.spines["right"].set_position(("outward", 60*3))  # Offset the fourth axis further to avoid overlap
# ax5.plot(span_positions, Re_r, color="tab:green", linestyle="--")
# ax5.set_ylabel("Reynold number", color="tab:green")
# ax5.tick_params(axis='y', labelcolor="tab:green")
# ax5.grid(False)  # Disable grid for the fourth y-axis

# Adjust layout to ensure the third axis is not cut off
fig.tight_layout()

# Save figure as JPG
plt.savefig("Performance over rotor span at max climb speed.jpg", dpi=300)  # dpi=resolution,

# -------------------------
# Mass distribution over Span
# -------------------------
# New figure for Angle of Attack vs Span at max RPM
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle(f"Wing mass distribution", fontsize=14)
span_positions = np.linspace(0.0, 1, len(dm_r))
ax.plot(span_positions, dm_r, color="tab:purple")
ax.set_ylabel("Mass (kg)", color="tab:purple")
ax.tick_params(axis='y', labelcolor="tab:purple")
ax.set_xlabel("Span Position normalized")
ax.grid(True)

plt.savefig("Wing Mass Distribution.jpg", dpi=300)  # dpi=resolution,

# Angle sweep
alpha = np.deg2rad(np.linspace(airfoil_params.alpha_min_deg, airfoil_params.alpha_max_deg, 300))

# -------------------------------------------------
# Airfoil Plots
# -------------------------------------------------
# Storage
Cl = np.zeros_like(alpha)
Cd = np.zeros_like(alpha)
Cm = np.zeros_like(alpha)
# Evaluate model
for i, a in enumerate(alpha):
    Cl[i], Cd[i], Cm[i] = airfoil_model(airfoil_params, a)
# -------------------------------------------------
# Plotting
# -------------------------------------------------
fig, ax1 = plt.subplots(figsize=(10, 6))
fig.suptitle(airfoil_params.airfoil_name + " Airfoil Performance", fontsize=14)
# ----------------------
# Primary y-axis: Cl
# ----------------------
color1 = "tab:blue"
line1, = ax1.plot(np.rad2deg(alpha), Cl, color=color1, label="Cl")
ax1.axvline(np.rad2deg(airfoil_params.alpha_stall_low), linestyle="--", color="red", label="Stall")
ax1.axvline(np.rad2deg(airfoil_params.alpha_stall_high), linestyle="--", color="red")
ax1.set_xlabel("Angle of attack (deg)")
ax1.set_ylabel("Cl", color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True)

# ----------------------
# Secondary y-axis: Cd
# ----------------------
ax2 = ax1.twinx()
color2 = "tab:green"
line2, = ax2.plot(np.rad2deg(alpha), Cd, color=color2, label="Cd")
ax2.set_ylabel("Cd", color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

# ----------------------
# Tertiary y-axis: Cm
# ----------------------
ax3 = ax1.twinx()
color3 = "tab:purple"
ax3.spines["right"].set_position(("outward", 60))  # Offset the third axis further to avoid overlap
line3, = ax3.plot(np.rad2deg(alpha), Cm, color=color3, label="Cm")
ax3.set_ylabel("Cm", color=color3)
ax3.tick_params(axis='y', labelcolor=color3)
ax3.grid(False)  # Disable grid for tertiary y-axis

# ----------------------
# Fourth y-axis: Cl/Cd
# ----------------------
ax4 = ax1.twinx()
color4 = "tab:orange"
ax4.spines["right"].set_position(("outward", 60*2))  # Offset the third axis further to avoid overlap
line4, = ax4.plot(np.rad2deg(alpha), Cl / Cd, color=color4, label="Cl/Cd")
ax4.set_ylabel("Cl / Cd", color=color4)
ax4.tick_params(axis='y', labelcolor=color4)

# Adjust layout to ensure the third axis is not cut off
fig.tight_layout()
# ----------------------
# Combine legends
# ----------------------
lines = [line1, line2, line3]
labels = [line.get_label() for line in lines]
# Include the stall line separately
stall_line = ax1.lines[1]  # the first axvline
lines.append(stall_line)
labels.append("Stall")
ax1.legend(lines, labels, loc="upper left")
# Save figure as JPG
plt.savefig("Airfoil Performance.jpg", dpi=300)  # dpi=resolution, quality=JPEG compression


# -------------------------------------------------
# Plotting Motor Thrust Data (line-labeled)
# -------------------------------------------------
fig, ax1 = plt.subplots(figsize=(10, 6))
fig.suptitle(apc_prop_selection + " Propeller Performance", fontsize=14)
ax1.text(
    0.5, 1.0,
    f"Motor KV: {motor_kv} RPM/V, Cells in Series: {N_series}\n"
    +f"RPM Range: {motor_rpm_at_min_v} RPM at {min_cell_voltage} V to {motor_rpm_at_max_v} RPM at {max_cell_voltage} V",
    transform=ax1.transAxes,
    ha="center", va="bottom",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white")
)
# ----------------------
# Primary axis: Motor thrust curves
# ----------------------
max_motor_thrust_kgf = max(np.max(np.array(thrusts) / 9.81)
                           for thrusts in desired_motor_thrusts_dict.values())
y_lim = 1.2 * max_motor_thrust_kgf

min_axial_speed = min(min(axial_speeds) for axial_speeds in axial_speed_dict.values())
max_axial_speed = max(max(axial_speeds) for axial_speeds in axial_speed_dict.values())
x_min = 0.6 * min_axial_speed
x_max = 1.1 * max_axial_speed

# ----------------------
# Background curves
# ----------------------
y_pos_dict, x_pos_dict = {}, {}

color_array = plt.cm.plasma(np.linspace(0, 0.8, len(motor_rpm_range)))

ax2 = ax1.twinx()  # Create a secondary y-axis

for rpm, data in prop_performance_dict.items():

    thrusts = data[0]
    power = data[1]
    efficiency = data[2]

    color = color_array[motor_rpm_range.tolist().index(rpm)]
    thrusts_kgf = np.array(thrusts) / 9.81

    # thrusts_kgf = n_prop_blades/apc_prop_blades * thrusts_kgf  # Scale thrust for different blade counts
    if (n_prop_blades == 3 and apc_prop_blades == 2):
        thrusts_kgf = 1.34 * thrusts_kgf
        efficiency = (1-0.0427)*efficiency
    elif (n_prop_blades == 3 and apc_prop_blades == 3):
        thrusts_kgf = thrusts_kgf;
        efficiency = efficiency;
    else:
       print(f"Blade count scaling not implemented for {n_prop_blades} blades vs {apc_prop_blades} blades")

    # Plot curve
    ax1.plot(axial_speed_range, thrusts_kgf, color=color, linestyle="--")
    ax2.plot(axial_speed_range, efficiency, color=color, linestyle=":")

    # print(thrusts_kgf)
    # print(y_lim)

    # Find index that is within the plot limits
    idx1 = np.where(axial_speed_range >= x_min)[0][0]
    idx2 = np.where(thrusts_kgf <= y_lim*0.95)[0][0]
    idx = max([idx1,idx2])

    x_pos_dict[rpm] = axial_speed_range[idx]
    y_pos_dict[rpm] = thrusts_kgf[idx]

# ----------------------
# Plot labels
# ----------------------
for rpm in motor_rpm_range:
    ax1.text(
        x_pos_dict[rpm],
        y_pos_dict[rpm],
        f"{rpm} RPM",
        color=color_array[motor_rpm_range.tolist().index(rpm)],
        fontsize=9,
        va="bottom",
        ha="left"
    )

ax1.set_xlabel("Axial Speed [m/s]")
ax1.set_ylabel("Prop Thrust [kgf]")
ax1.grid(True)
ax2.set_ylabel("Prop Efficiency")

# ----------------------
# Secondary axis: Required thrust
# ----------------------
for r_motor, desired_thrusts in desired_motor_thrusts_dict.items():
    axial_speeds = axial_speed_dict[r_motor]
    motor_thrusts_kgf = np.array(desired_thrusts) / 9.81
    line, = ax1.plot(
        axial_speeds,
        motor_thrusts_kgf
    )

    # Label at end of line
    ax1.text(
        axial_speeds[-1],
        motor_thrusts_kgf[-1],
        f"{r_motor:.1f} R",
        color=line.get_color(),
        fontsize=9,
        va="bottom",
        ha="center"
    )

for i in range(len(climb_speed_range)):
    climb_speed = climb_speed_range[i]
    motor_thrusts_kgf = np.array(T_motors_vs_climb[i,:]) / 9.81
    axial_speeds = (R_motor*R)*(monocopter_rpm*2*np.pi/60)
    line, = ax1.plot(axial_speeds, motor_thrusts_kgf)

    # Label at end of line
    ax1.text(
        axial_speeds[-1],
        motor_thrusts_kgf[-1],
        f"{climb_speed:.1f} m/s",
        color=line.get_color(),
        fontsize=9,
        va="top",
        ha="left"
    )

ax1.set_ylim(0, y_lim)
ax1.set_xlim(x_min, x_max)
# Save figure as JPG
plt.savefig("Motor Thrust Requirements.jpg", dpi=300)  # dpi=resolution,


# # -------------------------------------------------
# # Plotting Motor Power Data (line-labeled)
# # -------------------------------------------------
# efficiency = 0.75  # Assume a constant efficiency for the required power calculation

# fig, ax1 = plt.subplots(figsize=(10, 6))
# fig.suptitle(apc_prop_selection + " Propeller Power Performance", fontsize=14)
# ax1.text(
#     0.5, 1.0,
#     f"Motor KV: {motor_kv} RPM/V, Cells in Series: {N_series}\n"
#     +f"RPM Range: {motor_rpm_at_min_v} RPM at {min_cell_voltage} V to {motor_rpm_at_max_v} RPM at {max_cell_voltage} V",
#     transform=ax1.transAxes,
#     ha="center", va="bottom",
#     fontsize=10,
#     bbox=dict(boxstyle="round", facecolor="white")
# )
# # ----------------------
# # Primary axis: Motor thrust curves
# # ----------------------
# max_motor_power = max(np.max(np.array(thrusts) * np.array(axial_speeds))
#                            for thrusts, axial_speeds in zip( desired_motor_thrusts_dict.values(), axial_speed_dict.values() ))  / efficiency
# y_lim = 1.2 * max_motor_power

# min_axial_speed = min(min(axial_speeds) for axial_speeds in axial_speed_dict.values())
# max_axial_speed = max(max(axial_speeds) for axial_speeds in axial_speed_dict.values())
# x_min = 0.6 * min_axial_speed
# x_max = 1.1 * max_axial_speed

# # ----------------------
# # Background curves
# # ----------------------
# y_pos_dict, x_pos_dict = {}, {}

# color_array = plt.cm.plasma(np.linspace(0, 0.8, len(motor_rpm_range)))

# # ax2 = ax1.twinx()  # Create a secondary y-axis

# for rpm, data in prop_performance_dict.items():

#     thrusts = data[0]
#     power = data[1]
#     efficiency = data[2]

#     color = color_array[motor_rpm_range.tolist().index(rpm)]
#     thrusts_kgf = np.array(thrusts) / 9.81

#     # Plot curve
#     ax1.plot(axial_speed_range, power, color=color, linestyle="--")
#     # ax2.plot(axial_speed_range, efficiency, color=color, linestyle=":")

#     # Find index that is within the plot limits
#     idx1 = np.where(axial_speed_range >= x_min)[0][0]
#     idx2 = np.where(power <= y_lim*0.95)[0][0]
#     idx = max([idx1,idx2])

#     x_pos_dict[rpm] = axial_speed_range[idx]
#     y_pos_dict[rpm] = power[idx]
# # ----------------------
# # Plot labels
# # ----------------------
# for rpm in motor_rpm_range:
#     ax1.text(
#         x_pos_dict[rpm],
#         y_pos_dict[rpm],
#         f"{rpm} RPM",
#         color=color_array[motor_rpm_range.tolist().index(rpm)],
#         fontsize=9,
#         va="bottom",
#         ha="left"
#     )

# ax1.set_xlabel("Axial Speed [m/s]")
# ax1.set_ylabel("Prop Power [W]")
# ax1.grid(True)


# # ----------------------
# # Secondary axis: Required power
# # ----------------------
# efficiency = 0.75  # Assume a constant efficiency for the required power calculation
# for r_motor, desired_thrusts in desired_motor_thrusts_dict.items():
#     axial_speeds = axial_speed_dict[r_motor]
#     motor_powers = np.array(desired_thrusts) * np.array(axial_speeds) / efficiency
#     line, = ax1.plot(
#         axial_speeds,
#         motor_powers
#     )

#     # Label at end of line
#     ax1.text(
#         axial_speeds[-1],
#         motor_powers[-1],
#         f"{r_motor:.1f} R",
#         color=line.get_color(),
#         fontsize=9,
#         va="bottom",
#         ha="center"
#     )

# for i in range(len(climb_speed_range)):
#     climb_speed = climb_speed_range[i]
#     axial_speeds = (R_motor*R)*(monocopter_rpm*2*np.pi/60)
#     motor_powers = np.array(T_motors_vs_climb[i,:]) * axial_speeds / efficiency
#     line, = ax1.plot(axial_speeds, motor_powers)

#     # Label at end of line
#     ax1.text(
#         axial_speeds[-1],
#         motor_powers[-1],
#         f"{climb_speed:.1f} m/s",
#         color=line.get_color(),
#         fontsize=9,
#         va="top",
#         ha="left"
#     )

# ax1.set_ylim(0, y_lim)
# ax1.set_xlim(x_min, x_max)
# # Save figure as JPG
# plt.savefig("Motor Power Requirements.jpg", dpi=300)  # dpi=resolution,

# -------------------------------------------------
# Structural Analysis Summary Plot
# -------------------------------------------------
# New figure for structural analysis
fig, ax1 = plt.subplots(figsize=(8, 10))
fig.suptitle("Rotor Summary", fontsize=14)

ax1.axis('off')
ax1.text(
    0.2, 1.0,
    f"Rotor Radius: {R:.3f} m\n"
    +f"Precone angle: {precone_angle_deg:.3f} deg\n"
    +f"Root chord: {chord_root:.3f} m\n"
    +f"Tip chord: {chord_tip:.3f} m\n"
    +f"Blade twist: {(pitch_root-pitch_tip):.3f} deg\n"
    +f"Nominal Root pitch: {(pitch_root + collective_pitches[zero_speed_index]):.3f} deg\n"
    +f"Nominal Tip pitch: {(pitch_tip + collective_pitches[zero_speed_index]):.3f} deg\n"
    +f"Aspect Ratio: {R/((chord_root+chord_tip)/2):.3f}\n"
    +f"Airfoil: {airfoil_params.airfoil_name}\n"
    +f"Monocopter RPM: {monocopter_rpm:.1f}\n"
    +f"\n"
    +f"Figure of Merit at hover: {FM_values[zero_speed_index]:.3f}\n"
    +f"Solidity: {mean_solidity:.3f}\n"
    +f"kappa: {kappas[zero_speed_index]:.3f}\n"
    +f"Max Pitching Moment: {max(np.abs(pitch_moments_kgm)) * 100:.3f} kg-cm\n"
    +f"\n"
    +f"Spar Root Outer Diameter: {spar_root_OD / 0.0254 :.3f} in\n"
    +f"Spar Root Inner Diameter: {spar_root_ID / 0.0254 :.3f} in\n"
    +f"Spar Middle Outer Diameter: {spar_middle_OD / 0.0254 :.3f} in\n"
    +f"Spar Middle Inner Diameter: {spar_middle_ID / 0.0254 :.3f} in\n"
    +f"Spar Tip Outer Diameter: {spar_tip_OD / 0.0254 :.3f} in\n"
    +f"Spar Tip Inner Diameter: {spar_tip_ID / 0.0254 :.3f} in\n"
    +f"\n"
    +f"Max root stress: {max(stress_list[0]/1e6):.3f} MPa\n"
    +f"Max middle stress: {max(stress_list[1]/1e6):.3f} MPa\n"
    +f"Max tip stress: {max(stress_list[2]/1e6):.3f} MPa\n"
    +f"Min root Factor of Safety: {min(fos_list[0]):.3f}\n"
    +f"Min middle Factor of Safety: {min(fos_list[1]):.3f}\n"
    +f"Min tip Factor of Safety: {min(fos_list[2]):.3f}\n"
    +f"\n"
    +f"Wing skin thickness: {structural_params.skin_thickness*1000:.3f} mm\n"
    +f"Wing Skin Weight Estimate: {skin_weight:.3f} kg\n"
    +f"Wing Weight Estimate: {wing_weight:.3f} kg\n"
    +f"Spar Weight Estimate: {spar_weight:.3f} kg\n"
    +f"Total Blade Weight Estimate: {(wing_weight+spar_weight+skin_weight):.3f} kg\n"
    +f"Center of Gravity (spanwise): {cgx:.3f} m\n"
    +f"Center of Gravity (chordwise): {cgy:.3f} m",
    transform=ax1.transAxes,
    ha="left", va="top",
    fontsize=12,
    bbox=dict(boxstyle="round", facecolor="white")
)

# Save figure as JPG
plt.savefig("Rotor Summary.jpg", dpi=300)  # dpi=resolution, quality=JPEG compression

# plot_rotor_blade_3d_airfoil(
#     airfoil_name=airfoil_params.airfoil_name,
#     root_chord=chord_root,
#     tip_chord=chord_tip,
#     hub_radius=0.2*R,
#     blade_length=0.8*R,
#     twist_root_deg=pitch_root + collective_pitches[zero_speed_index],
#     twist_tip_deg=pitch_tip + collective_pitches[zero_speed_index]
# )

# plt.tight_layout()


# ## Store arrays of lift and drag coeeficients for later use
# alpha = np.deg2rad(np.linspace(-180, 180, 360))
# Cl = np.zeros_like(alpha)
# Cd = np.zeros_like(alpha)
# Cm = np.zeros_like(alpha)

# # reinitialize airfoil model with parameters
# airfoil_params = AirfoilParams(
#     airfoil_name='naca6412',  # path to file
#     Re=0.6e6,
#     Mach=0.0,
#     alpha_min_deg=-180,
#     alpha_max_deg=180,
#     alpha_step_deg=1.0,
#     alpha_stall_high = np.deg2rad(12.0),
#     alpha_stall_low = np.deg2rad(-10.0),
# )

# # Evaluate model
# for i, a in enumerate(alpha):
#     Cl[i], Cd[i], Cm[i] = airfoil_model(airfoil_params, a)

# # subplots of cl, cd, and cm to verify
# fig, ax1 = plt.subplots(figsize=(10, 6))
# fig.suptitle(airfoil_params.airfoil_name + " Airfoil Performance", fontsize=14)
# color1 = "tab:blue"
# line1, = ax1.plot(np.rad2deg(alpha), Cl, color=color1, label="Cl")
# ax1.axvline(np.rad2deg(airfoil_params.alpha_stall_low), linestyle="--", color="red", label="Stall")
# ax1.axvline(np.rad2deg(airfoil_params.alpha_stall_high), linestyle="--", color="red")
# ax1.set_xlabel("Angle of attack (deg)")
# ax1.set_ylabel("Cl", color=color1)
# ax1.tick_params(axis='y', labelcolor=color1)
# ax1.grid(True)
# ax1.legend()

# fig, ax2 = plt.subplots(figsize=(10, 6))
# color2 = "tab:green"
# line2, = ax2.plot(np.rad2deg(alpha), Cd, color=color2, label="Cd")
# ax2.set_xlabel("Angle of attack (deg)")
# ax2.set_ylabel("Cd", color=color2)
# ax2.tick_params(axis='y', labelcolor=color2)
# ax2.grid(True)
# ax2.legend()

# fig, ax3 = plt.subplots(figsize=(10, 6))
# color3 = "tab:purple"
# line3, = ax3.plot(np.rad2deg(alpha), Cm, color=color3, label="Cm")
# ax3.set_xlabel("Angle of attack (deg)")
# ax3.set_ylabel("Cm", color=color3)
# ax3.tick_params(axis='y', labelcolor=color3)
# ax3.grid(True)
# ax3.legend()

# # Save data to a text file in c# format
# with open("airfoil_coefficients.txt", "w") as f:
#     # f.write("aoa, cl, cd, cm\n")
#     for a, cl, cd, cm in zip(np.rad2deg(alpha), Cl, Cd, Cm):
#         f.write(f" {{ {a:.2f}f, {cl:.6f}f, {cd:.6f}f, {cm:.6f}f }},\n")


plt.show()
