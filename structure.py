import numpy as np
import aerosandbox as asb
from dataclasses import dataclass, field

@dataclass
class WingStructuralParams:
    No_Blades: float                        # no. of blades
    spar_root_ID: float                     # m
    spar_root_OD: float                     # m
    spar_middle_ID: float                   # m
    spar_middle_OD: float                   # m
    spar_middle_start: float                # m
    spar_tip_ID: float                      # m
    spar_tip_OD: float                      # m
    spar_tip_start: float                   # m
    spar_length: float                      # m
    spar_material_density: float            # kg/m^3
    spar_material_strength: float           # Pa
    chord_root: float                       # m
    chord_tip: float                        # m
    wing_span: float                        # m
    wing_material_density: float            # kg/m^3
    airfoil_area_coeff: float               # A = coeff * c^2
    airfoil_name: str                       # e.g. "NACA 2412"
    x_af: np.ndarray = field(init=False)
    y_af: np.ndarray = field(init=False)
    skin_thickness: float                   # m
    skin_density: float                     # kg/m^3

    def __post_init__(self):
        airfoil = asb.Airfoil(self.airfoil_name)
        coords = airfoil.coordinates
        self.x_af = coords[:, 0]
        self.y_af = coords[:, 1]

def airfoil_perimeter_length(x_af, y_af, chord=1.0):
    x = np.asarray(x_af) * chord
    y = np.asarray(y_af) * chord
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    perimeter = np.sum(ds)
    return perimeter

def compute_skin_mass(params: WingStructuralParams, n_sections: int = 50):
    mass = 0.0
    for i in range(n_sections):
        c = params.chord_root + (params.chord_tip - params.chord_root) * (i / (n_sections - 1))
        perimeter = airfoil_perimeter_length(params.x_af, params.y_af, chord=c)
        dx = params.wing_span / (n_sections - 1)
        mass += perimeter * params.skin_thickness * dx * params.skin_density
    return mass

def compute_wing_mass(params: WingStructuralParams):
    root_area = params.airfoil_area_coeff * params.chord_root**2
    tip_area = params.airfoil_area_coeff * params.chord_tip**2
    wing_span = params.wing_span
    volume = (root_area + tip_area) * wing_span / 2.0
    return params.wing_material_density * volume

def compute_tube_mass(params: WingStructuralParams):
    length = params.spar_length
    cross_section_area_root = (np.pi / 4) * (params.spar_root_OD**2 - params.spar_root_ID**2)
    cross_section_area_middle = (np.pi / 4) * (params.spar_middle_OD**2 - params.spar_middle_ID**2)
    cross_section_area_tip = (np.pi / 4) * (params.spar_tip_OD**2 - params.spar_tip_ID**2)
    length_root = params.spar_middle_start
    length_middle = params.spar_tip_start - params.spar_middle_start
    length_tip = length - params.spar_tip_start
    volume = (cross_section_area_root * length_root +
              cross_section_area_middle * length_middle +
              cross_section_area_tip * length_tip)
    # volume = cross_section_area * length
    # volume_outer = (1/12)*np.pi*length*(params.spar_root_OD**2 + params.spar_tip_OD**2 + params.spar_root_OD*params.spar_tip_OD)
    # volume_inner = (1/12)*np.pi*length*(params.spar_root_ID**2 + params.spar_tip_ID**2 + params.spar_root_ID*params.spar_tip_ID)
    # volume = volume_outer - volume_inner
    return params.spar_material_density * volume

def compute_center_of_gravity(params: WingStructuralParams, omega, precone_angle_deg,n_sections: int):
    dx = params.spar_length / n_sections
    total_mass = 0.0
    moment_x = 0.0 # about spanwise axis
    moment_y = 0.0 # about chordwise axis

    r = []  # spanwise locations
    dm_r = []  # mass distribution along the span
    dcf_r = []  # centrifugal force distribution along the span
    dcf_flap_r = []  # centrifugal force distribution along the span - flapwise
    dcf_leadlag_r = []  # centrifugal force distribution along the span - lead-lag wise
    dcf_pitch_m_r = []  # centrifugal pitch moment distribution along the span
    dcf_flap_m_r = []  # centrifugal flap moment distribution along the span
    dcf_leadlag_m_r = []  # centrifugal lead-lag moment distribution along the span
    dec_r = []  # e/c ratio distribution along the span
    for i in range(n_sections):
        x = dx * (i + 0.5)
        r.append(x)
        # first add spar mass
        # spar_OD = params.spar_root_OD - (params.spar_root_OD - params.spar_tip_OD) * x / params.spar_length
        # spar_ID = params.spar_root_ID - (params.spar_root_ID - params.spar_tip_ID) * x / params.spar_length

        if x < params.spar_middle_start:
            spar_OD = params.spar_root_OD
            spar_ID = params.spar_root_ID
        if x >= params.spar_middle_start and x < params.spar_tip_start:
            spar_OD = params.spar_middle_OD
            spar_ID = params.spar_middle_ID
        if x >= params.spar_tip_start:
            spar_OD = params.spar_tip_OD
            spar_ID = params.spar_tip_ID

        spar_mass = (np.pi / 4) * (spar_OD**2 - spar_ID**2) * dx * params.spar_material_density # approximate cylinder
        total_mass += spar_mass
        moment_x += spar_mass * x
        moment_y += 0.0 # spar assumed at center

        if x < 0.2 * params.spar_length:
            # no wing only spar
            section_mass = spar_mass
            section_cg_y = 0.0
            section_cg_y_ratio = 0.0
        else:
            # add wing mass
            x_wing = (x - 0.2 * params.spar_length)

            chord = params.chord_root + (params.chord_tip - params.chord_root) * (x_wing / params.wing_span)
            area = params.airfoil_area_coeff * chord**2
            wing_volume = area * dx
            wing_mass = wing_volume * params.wing_material_density

            total_mass += wing_mass
            moment_y += wing_mass * chord * (0.25-0.412)  # Uniform density assumption
            # moment_y += wing_mass * chord * (0.25-0.5)  # hollow airfoil assumption
            moment_x += wing_mass * x

            # add skin mass
            perimeter = airfoil_perimeter_length(params.x_af, params.y_af, chord)
            skin_mass = perimeter * params.skin_thickness * dx * params.skin_density
            total_mass += skin_mass
            moment_x += skin_mass * x
            moment_y += skin_mass * chord * (0.25-0.5)  # hollow airfoil assumption

            section_mass = spar_mass + wing_mass + skin_mass
            section_cg_y = (wing_mass * chord * (0.25-0.412) + skin_mass * chord * (0.25-0.5))/section_mass  # relative to spar center

            section_cg_y_ratio = section_cg_y / chord
            # print(f"e/c = {section_cg_y_ratio:.3f}")

        dec_r.append(section_cg_y_ratio)
        dm_r.append(section_mass)
        centrifugal_force_section = section_mass * x * omega**2
        centrifugal_force_section_flap_comp = centrifugal_force_section * np.sin(np.deg2rad(precone_angle_deg))
        centrifugal_force_section_lead_lag_comp = centrifugal_force_section * np.cos(np.deg2rad(precone_angle_deg))
        centrifugal_pitch_moment = centrifugal_force_section_flap_comp * section_cg_y
        centrifugal_flap_moment = centrifugal_force_section_flap_comp * x
        centrifugal_leadlag_moment = centrifugal_force_section_lead_lag_comp * section_cg_y
        dcf_r.append(centrifugal_force_section)
        dcf_flap_r.append(centrifugal_force_section_flap_comp)
        dcf_leadlag_r.append(centrifugal_force_section_lead_lag_comp)
        dcf_pitch_m_r.append(centrifugal_pitch_moment)
        dcf_flap_m_r.append(centrifugal_flap_moment)
        dcf_leadlag_m_r.append(centrifugal_leadlag_moment)

    cgx = moment_x / total_mass if total_mass != 0 else 0.0
    cgy = moment_y / total_mass if total_mass != 0 else 0.0

    return (cgx, cgy, dm_r, dcf_r, dcf_flap_r, dcf_leadlag_r, dcf_pitch_m_r, dcf_flap_m_r, dcf_leadlag_m_r, r, dec_r)

def structural_analysis(leadlag_torques_root, leadlag_torques_middle, leadlag_torques_tip, flap_torques_root, flap_torques_middle, flap_torques_tip, precone_angle_deg, omega, params: WingStructuralParams):
    # Simplified structural analysis placeholder
    skin_weight = compute_skin_mass(params)
    wing_weight = compute_wing_mass(params)
    spar_weight = compute_tube_mass(params)
    total_mass = wing_weight + spar_weight + skin_weight
    n_sections = 100
    (cgx, cgy, dm_r, dcf_r, dcf_flap_r, dcf_leadlag_r, dcf_pitch_m_r, dcf_flap_m_r, dcf_leadlag_m_r, r, dec_r) = compute_center_of_gravity(params, omega, precone_angle_deg, n_sections)

    root_spar_area_moment_of_inertia = (params.spar_root_OD**4 - params.spar_root_ID**4) * np.pi / 64
    middle_spar_area_moment_of_intertia = (params.spar_middle_OD**4 - params.spar_middle_ID**4) * np.pi / 64
    tip_spar_area_moment_of_intertia = (params.spar_tip_OD**4 - params.spar_tip_ID**4) * np.pi / 64

    print("root_spar_area_moment_of_inertia:", root_spar_area_moment_of_inertia)
    print("middle_spar_area_moment_of_inertia:", middle_spar_area_moment_of_intertia)
    print("tip_spar_area_moment_of_inertia:", tip_spar_area_moment_of_intertia)

    # dm_r_root = np.array(dm_r)
    # dm_r_middle = np.array(dm_r)[int(len(r)*params.spar_middle_start/params.spar_length):]
    # dm_r_tip = np.array(dm_r)[int(len(r)*params.spar_tip_start/params.spar_length):]

    r_root = np.array(r)
    r_middle = np.array(r)[int(len(r)*params.spar_middle_start/params.spar_length):]
    r_tip = np.array(r)[int(len(r)*params.spar_tip_start/params.spar_length):]

    # d_centrifugal_force_root = dm_r_root * r_root * omega**2 * np.sin(np.deg2rad(precone_angle_deg))
    # d_centrifugal_force_middle = dm_r_middle * r_middle * omega**2 * np.sin(np.deg2rad(precone_angle_deg))
    # d_centrifugal_force_tip = dm_r_tip * r_tip * omega**2 * np.sin(np.deg2rad(precone_angle_deg))

    # d_centrifugal_force_root_kgf_non_dim = (dm_r_root/(params.spar_length/float(n_sections))) * r_root * omega**2 * np.sin(np.deg2rad(precone_angle_deg)) / 9.81
    d_centrifugal_flap_force_kgf_non_dim = float(n_sections) * np.array(dcf_flap_r) / 9.81
    dcf_pitch_m_r_kgcm_non_dim = float(n_sections) * np.array(dcf_pitch_m_r) * 0.101971621 * 100
    # plot centrifugal force root distribution vs spanwise location
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.plot(r_root/params.spar_length, d_centrifugal_flap_force_kgf_non_dim, label='Centrifugal Force Flap (kgf)')
    plt.plot(r_root/params.spar_length, dcf_pitch_m_r_kgcm_non_dim, label='Centrifugal Pitching Moment (kg-cm)')
    plt.legend()
    ax1.set_xlabel('Spanwise Location Ratio')
    ax1.set_ylabel('Centrifugal')
    plt.title('Centrifugal Force Flap Component Along the Span')
    ax1.grid()

    # plot e/c ratio along the span
    fig, ax2 = plt.subplots(figsize=(10, 6))
    plt.plot(r_root/params.spar_length, np.array(dec_r)*100, label='e/c Ratio %')
    plt.legend()
    ax2.set_xlabel('Spanwise Location Ratio')
    ax2.set_ylabel('e/c Ratio %')
    plt.title('e/c Ratio Along the Span')
    ax2.grid()

    centrifugal_torque_flap_root = params.No_Blades * np.sum(np.array(dcf_flap_r) * r_root)
    centrifugal_torque_flap_middle = params.No_Blades * np.sum(np.array(dcf_flap_r)[int(len(r)*params.spar_middle_start/params.spar_length):] * (r_middle - params.spar_middle_start))
    centrifugal_torque_flap_tip = params.No_Blades * np.sum(np.array(dcf_flap_r)[int(len(r)*params.spar_tip_start/params.spar_length):] * (r_tip - params.spar_tip_start))

    centripetal_torque_lead_lag = cgy * total_mass * (cgx * omega)**2 / cgx

    leadlag_torques_root = np.array(leadlag_torques_root)
    leadlag_torques_middle = np.array(leadlag_torques_middle)
    leadlag_torques_tip = np.array(leadlag_torques_tip)

    # flap_torques = np.array(flap_torques) - 0.4 * params.spar_length * (50*9.81/3) * np.sin(np.deg2rad(30))  # subtracting distributed load effect
    # flap_torques_root = np.array(flap_torques_root)
    # flap_torques_middle = np.array(flap_torques_middle)
    # flap_torques_tip = np.array(flap_torques_tip)
    flap_torques_root = np.array(flap_torques_root) - centrifugal_torque_flap_root
    flap_torques_middle = np.array(flap_torques_middle) - centrifugal_torque_flap_middle
    flap_torques_tip = np.array(flap_torques_tip) - centrifugal_torque_flap_tip

    # leadlag_torques = np.array(leadlag_torques) + centripetal_torque_lead_lag
    total_torques_root = np.sqrt((leadlag_torques_root / params.No_Blades)**2 + (flap_torques_root / params.No_Blades)**2)
    total_torques_middle = np.sqrt((leadlag_torques_middle / params.No_Blades)**2 + (flap_torques_middle / params.No_Blades)**2)
    total_torques_tip = np.sqrt((leadlag_torques_tip / params.No_Blades)**2 + (flap_torques_tip / params.No_Blades)**2)

    root_stress = total_torques_root * params.spar_root_OD * 0.5 / root_spar_area_moment_of_inertia
    middle_stress = total_torques_middle * params.spar_middle_OD * 0.5 / middle_spar_area_moment_of_intertia
    tip_stress = total_torques_tip * params.spar_tip_OD * 0.5 / tip_spar_area_moment_of_intertia

    print("root_stress:", root_stress)
    print("middle_stress:", middle_stress)
    print("tip_stress:", tip_stress)

    fos_root = params.spar_material_strength / root_stress
    fos_middle = params.spar_material_strength / middle_stress
    fos_tip = params.spar_material_strength / tip_stress

    stress = [root_stress, middle_stress, tip_stress]
    fos = [fos_root, fos_middle, fos_tip]

    return stress, fos, skin_weight, wing_weight, spar_weight, cgx, cgy, dcf_pitch_m_r, float(n_sections)*np.array(dm_r)
