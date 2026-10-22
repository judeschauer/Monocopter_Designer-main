import numpy as np
from dataclasses import dataclass
import neuralfoil as nf
import aerosandbox as asb
from structure import  WingStructuralParams

from dataclasses import dataclass, field
from typing import Optional, List, Union


@dataclass
class AirfoilParams:
    # Input parameters
    airfoil_name: str                       # e.g. "4412", "OA212"
    Re: float                               # Reynolds number
    Mach: float                             # Mach number

    # Polar sampling range
    alpha_min_deg: float = -10.0                # degrees
    alpha_max_deg: float = 20.0                 # degrees
    alpha_step_deg: float = 1.0                 # degrees

    alpha_stall_low: float = -10.0
    alpha_stall_high: float = 10.0

    # Arrays will be filled automatically
    alpha_array: np.ndarray = field(init=False)
    CL_array: np.ndarray = field(init=False)
    CD_array: np.ndarray = field(init=False)
    CM_array: np.ndarray = field(init=False)

    def __post_init__(self):
        """
        Automatically generate the alpha range,
        call NeuralFoil to generate polars,
        and populate arrays.
        """
        # Generate alpha range
        self.alpha_array = np.arange(self.alpha_min_deg, self.alpha_max_deg + self.alpha_step_deg, self.alpha_step_deg)

        airfoil = asb.Airfoil(self.airfoil_name)
        results = airfoil.get_aero_from_neuralfoil(
            alpha=self.alpha_array,
            Re = self.Re,
            mach=self.Mach,
        )

        # Populate arrays
        self.CL_array = np.array(results["CL"])
        self.CD_array = np.array(results["CD"])
        self.CM_array = np.array(results["CM"])

    # ============================================================
    #                INTERPOLATION FUNCTIONS
    # ============================================================

    def Cl(self, alpha_deg: Union[float, np.ndarray]):
        return np.interp(alpha_deg, self.alpha_array, self.CL_array)

    def Cd(self, alpha_deg: Union[float, np.ndarray]):
        return np.interp(alpha_deg, self.alpha_array, self.CD_array)

    def Cm(self, alpha_deg: Union[float, np.ndarray]):
        return np.interp(alpha_deg, self.alpha_array, self.CM_array)


def airfoil_model(params: AirfoilParams, alpha):
    # # Linear lift
    # Cl_linear = params.Cl_alpha * (alpha - params.alpha_zero)

    # # Stall saturation
    # Cl_max = params.Cl_alpha * (params.alpha_stall - params.alpha_zero)
    # Cl = Cl_max * 1.0 * np.tanh(Cl_linear / Cl_max)

    # # Post-stall profile drag rise
    # stall_factor = 1 + 2.0 * (np.abs(Cl) / Cl_max)**4
    # Cd = params.Cd0 + stall_factor * params.Cd1 * (alpha)**2

    # Cm = params.Cm0

    alpha_deg = np.rad2deg(alpha)
    Cl = params.Cl(alpha_deg)
    Cd = params.Cd(alpha_deg)
    Cm = params.Cm(alpha_deg)

    return Cl, Cd, Cm


def BEMT_M(speed_sound, theta, Nb, x, sigma, lambda_c, lambda_h, airfoil_params: AirfoilParams):
    lam = lambda_h
    err = 1
    while abs(err) > 0.001:
        alpha = theta - np.arctan2(lam, x)
        u = np.sqrt(x**2 + lam**2)
        Mc = u / speed_sound

        Cl, Cd, Cm = airfoil_model(airfoil_params, alpha)

        f = Nb/2 * (x - 1) / np.abs(lam)
        F = (2 / np.pi) * np.arccos(np.exp(f))
        lam_old = lam

        if lambda_c == 0:
            lam_new = np.sign(Cl) * 0.5 * np.sqrt(0.5 * sigma * u * np.abs(Cl) / F)

        elif lambda_c > 0:
            if np.any(lambda_c**2 + 0.5 * sigma * x * Cl / F < 0):
                print('-ve values under sqrt for lambda_c>0 in BEMT function, Cl could be -ve for low lambda_c')
            lam_new = lambda_c / 2 + 0.5 * np.sqrt(lambda_c**2 + 0.5 * sigma * x * Cl / F)

        else:  # lambda_c < 0
            if np.any(lambda_c**2 - 0.5 * sigma * x * Cl / F < 0):
                print('-ve values under sqrt for lambda_c<0 in BEMT function, Cl could be -ve for low abs(lambda_c)')
            lam_new = lambda_c / 2 - 0.5 * np.sqrt(lambda_c**2 - 0.5 * sigma * x * Cl / F)

        err = (lam_new - lam_old) / lam_old
        lam = lam_old + 0.01 * (lam_new - lam_old)

    # print("converged %f, %f", lam, x)
    return lam

def Thrust_M(speed_sound, Nb, r_cut, r, sigma, theta, lambda_h, lambda_c, airfoil_params: AirfoilParams, stalled=False):
    N = len(r)
    if ((sigma.size < N) or (theta.size < N)):
        print('error: incompatible size of sigma,theta,r in Thrust function')

    lam = np.zeros(N)
    alpha = np.zeros(N)
    u = np.zeros(N)
    Mc = np.zeros(N)
    Cl = np.zeros(N)
    Cd = np.zeros(N)
    Cm = np.zeros(N)
    dC_T = np.zeros(N)
    dC_Pi = np.zeros(N)
    dC_Po = np.zeros(N)
    dC_P = np.zeros(N)

    for i in range(N):
        lam[i] = BEMT_M(speed_sound, theta[i], Nb, r[i], sigma[i], lambda_c, lambda_h, airfoil_params)
        alpha[i] = theta[i] - np.arctan(lam[i] / r[i])

        if alpha[i] > airfoil_params.alpha_stall_high or alpha[i] < airfoil_params.alpha_stall_low:
            print('Stall detected in Thrust_M function at section ', i)
            stalled=True

        u[i] = np.sqrt(r[i]**2 + lam[i]**2)
        Mc[i] = u[i] / speed_sound

        # if r[i] >= 0.95:
        #     Mc[i] *= 0.826 / (0.59 * (r[i] + 0.45))

        Cl[i], Cd[i], Cm[i] = airfoil_model(airfoil_params, alpha[i])

        dC_T[i] = (0.5 * sigma[i] * u[i] * r[i] * Cl[i] - 0.5 * sigma[i] * u[i] * lam[i] * Cd[i])  # formula valid always

        dC_Pi[i] = lam[i] * (0.5 * sigma[i] * u[i] * r[i] * Cl[i])
        if np.sign(dC_Pi[i]) < 0:
            print('-ve induced power')

        dC_Po[i] = 0.5 * sigma[i] * Cd[i] * u[i]**3
        if np.sign(dC_Po[i]) < 0:
            print('-ve profile power')

        dC_P[i] = dC_Pi[i] + dC_Po[i]  # valid only in hover

    r2 = np.zeros(N)
    r2[0] = (r[0] - r_cut) * 2

    for i in range(1, N):
        r2[i] = (r[i] - r[i - 1]) * 2 - r2[i - 1]

    C_T = np.sum(dC_T * r2)
    C_Pi = np.sum(dC_Pi * r2)
    C_P = np.sum(dC_P * r2)

    return C_T, C_Pi, C_P, lam, dC_T, dC_Pi, dC_Po, dC_P, Cl, Cd, Cm, alpha, stalled

def bemt2(
    omega,
    climb_speed,
    R,
    B,
    rho,
    chord_root,
    chord_const_root_end,
    chord_tip,
    pitch_root_deg,
    pitch_tip_deg,
    airfoil_params: AirfoilParams,
    structural_params: WingStructuralParams,
    n_sections=40
):
    r_cut = 0.2
    lambda_h_init = 0.05

    r = np.linspace(r_cut, 1, n_sections) # non-dimensional radial locations
    # mid-point rule
    r = 0.5 * (np.concatenate((r[1:], [0])) + r)
    r = r[:-1] # remove last point which is at r=1

    tip_speed = R*omega
    lambda_c = climb_speed/tip_speed

    # c_75 = chord_root - 0.75*(chord_root - chord_tip)
    # mean_solidity = B*c_75/(2*np.pi*R)
    # taper_ratio = chord_root/chord_tip
    # if taper_ratio == 1:
        # taper_rate = 0
    # else:
        # taper_rate = mean_solidity / (0.75 + taper_ratio / (1 - taper_ratio))
    # root_solidity = mean_solidity - 0.75 * taper_rate

    # root_solidity = B*chord_root/(2*np.pi*R)
    # taper_rate = B*(chord_tip-chord_root)/(2*np.pi*R)
    # sigma = root_solidity + taper_rate * r

    sigma = np.zeros_like(r)
    for i in range(len(r)):
        if r[i]*R <= chord_const_root_end:
            c = chord_root
        else:
            c = chord_root - ( (r[i]- (chord_const_root_end/R)) / (1 - (chord_const_root_end/R)) ) * (chord_root - chord_tip)
        sigma[i] = B * c / (2 * np.pi * R)

    theta_tip = np.deg2rad(pitch_tip_deg)
    theta_root = np.deg2rad(pitch_root_deg)
    theta = np.linspace(theta_root, theta_tip, n_sections)
    theta = 0.5 * (np.concatenate((theta[1:], [0])) + theta)
    theta = theta[:-1]

    speed_sound = 339.3/tip_speed

    stalled = False
    lambda_h = lambda_h_init
    C_T, C_Pi, C_P, lambda_, dC_T, dC_Pi, dC_Po, dC_P, Cl, Cd, Cm, alpha, stalled = Thrust_M(speed_sound, B, r_cut, r, sigma, theta, lambda_h, lambda_c, airfoil_params)

    thrust = C_T * rho * np.pi * (R**2) * (tip_speed**2)
    power = C_P *  rho * np.pi * (R**2) * (tip_speed**3)
    # leadlag_torque_root = power / omega

    pitch_moment, flap_torque_root, flap_torque_middle, flap_torque_tip, leadlag_torque_root, leadlag_torque_middle, leadlag_torque_tip = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    Re_r = np.zeros_like(r)
    dM_r = np.zeros_like(r)
    dr = np.zeros_like(r)

    for i in range(len(r)):
        if i==0:
            dr[i] = (r[i] - r_cut) * 2
        else:
            dr[i] = (r[i] - r[i - 1]) * 2 - dr[i - 1]

        if r[i]*R <= chord_const_root_end:
            c = chord_root
        else:
            c = chord_root - ( (r[i]- (chord_const_root_end/R)) / (1 - (chord_const_root_end/R)) ) * (chord_root - chord_tip)

        dM_r[i] = Cm[i] * 0.5 * rho * (omega * r[i] * R)**2 * (R) * (c**2)
        pitch_moment += dM_r[i] * dr[i]
        Re_r[i] = (omega * r[i] * R) * c / 1.4207E-5

        d_flap_force = (dr[i]) * dC_T[i] * rho * np.pi * (R**2) * (tip_speed**2)
        d_leadlag_force = (dr[i]) * dC_P[i] * rho * np.pi * (R**2) * (tip_speed**3) / (omega * r[i] * R)

        flap_torque_root += d_flap_force * r[i] * R
        leadlag_torque_root += d_leadlag_force * r[i] * R
        if r[i]*R >= structural_params.spar_middle_start:
            flap_torque_middle += d_flap_force * (r[i]*R - structural_params.spar_middle_start)
            leadlag_torque_middle += d_leadlag_force * (r[i]*R - structural_params.spar_middle_start)
        if r[i]*R >= structural_params.spar_tip_start:
            flap_torque_tip += d_flap_force * (r[i]*R - structural_params.spar_tip_start)
            leadlag_torque_tip += d_leadlag_force * (r[i]*R - structural_params.spar_tip_start)

    flap_torques = [flap_torque_root, flap_torque_middle, flap_torque_tip]
    leadlag_torques = [leadlag_torque_root, leadlag_torque_middle, leadlag_torque_tip]

    dT_r = dC_T * rho * np.pi * (R**2) * (tip_speed**2)
    dF_r = (dC_P / (r*R)) * rho * np.pi * (R**2) * (tip_speed**3) / omega
    dLD_r = Cl/Cd

    FM_value = (C_T**1.5 / np.sqrt(2))/C_P
    # mean_solidity = B * (0.5 * (chord_root + chord_tip)) / (2 * np.pi * R)
    mean_solidity = B * (0.5 * (chord_root + chord_tip) * 0.8*R) / (np.pi * R*R)
    Cd0 = np.min(airfoil_params.CD_array)
    kappa = (1/FM_value) - (mean_solidity * Cd0 / 8)/(C_T**1.5 / np.sqrt(2))

    return thrust, leadlag_torques, flap_torques, power, alpha, dT_r, dF_r, dLD_r, dM_r, pitch_moment, Re_r, FM_value, kappa, stalled


# def transcedental_solver(
#     sigma,
#     Cl_alpa,
#     beta,
#     theta,
#     alpha_zero,
#     alpha_stall,
#     Cd0,
#     Cd1
# ):
#     phi_neg = 0
#     alpha = beta + theta - phi_neg
#     Cl, Cd = airfoil_model(alpha, Cl_alpa, alpha_zero, alpha_stall, Cd0, Cd1)
#     f_neg = - sigma * Cl

#     phi_pos = np.pi/2
#     alpha = beta + theta - phi_pos
#     Cl, Cd = airfoil_model(alpha, Cl_alpa, alpha_zero, alpha_stall, Cd0, Cd1)
#     f_pos = 4 + sigma * Cd

#     for _ in range(20):

#         phi_new = 0.5*(phi_neg + phi_pos)

#         alpha = beta + theta - phi_new
#         Cl, Cd = airfoil_model(alpha, Cl_alpa, alpha_zero, alpha_stall, Cd0, Cd1)
#         gamma = np.arctan(Cd/Cl)

#         f_new = 4.0 * (np.sin(phi_new))**2 - sigma * Cl * (1.0/np.cos(gamma)) * np.cos(phi_new + gamma)

#         if f_new < 0.0:
#             phi_neg = phi_new
#             f_neg = f_new
#         else:
#             phi_pos = phi_new
#             f_pos = f_new

#     phi = phi_neg - f_neg / ((f_pos-f_neg) / (phi_pos-phi_neg))

#     return phi

# # -----------------------------
# # BEMT Solver (Hover)
# # -----------------------------
# def bemt(
#     omega,
#     R,
#     B,
#     rho,
#     chord_root,
#     chord_tip,
#     twist_root_deg,
#     twist_tip_deg,
#     alpha_zero_deg,
#     alpha_stall_deg,
#     beta_deg,
#     Cl_alpa,
#     Cd0,
#     Cd1,
#     n_sections=40
# ):

#     r = np.linspace(0.2 * R, R, n_sections)
#     dr = r[1] - r[0]

#     chord = np.linspace(chord_root, chord_tip, n_sections)
#     twist = np.deg2rad(
#         np.linspace(twist_root_deg, twist_tip_deg, n_sections)
#     )

#     alpha_zero = np.deg2rad(alpha_zero_deg)
#     beta = np.deg2rad(beta_deg)
#     alpha_stall = np.deg2rad(alpha_stall_deg)

#     # c_mean = np.mean(chord)
#     # AR = R / c_mean  # rotor effective aspect ratio

#     thrust = 0.0
#     azim_torque = 0.0
#     axial_torque = 0.0
#     stalled = False

#     for i in range(n_sections):
#         ri = r[i]
#         ci = chord[i]
#         theta = twist[i]

#         sigma = B*ci/(2*np.pi*ri)

#         phi = transcedental_solver(sigma,Cl_alpa,beta,theta,alpha_zero,alpha_stall,Cd0,Cd1)

#         alpha = beta + theta - phi

#         if (alpha > alpha_stall) or (alpha < -alpha_stall):
#             stalled=True

#         Cl, Cd = airfoil_model(alpha, Cl_alpa, alpha_zero, alpha_stall, Cd0, Cd1)
#         gamma = np.arctan(Cd/Cl)

#         b = 1.0/(1.0+((4.0*np.cos(gamma)*np.sin(phi)*np.cos(phi))/(sigma*Cl*np.sin(phi+gamma))))

#         V_rel = (1.0 - b) * omega * ri * (1.0/np.cos(phi))
#         # V_axial = V_tan * np.sin(phi)
#         # V_rel = np.sqrt(V_axial**2 + V_tan**2)

#         L = 0.5 * rho * V_rel**2 * ci * Cl * dr
#         D = 0.5 * rho * V_rel**2 * ci * Cd * dr

#         Fn = L * np.cos(phi) - D * np.sin(phi)
#         Ft = L * np.sin(phi) + D * np.cos(phi)

#         thrust += B * Fn
#         azim_torque += B * Ft * ri
#         axial_torque += B * Fn * ri

#         # thrust+= 4*np.pi*rho*ri*(V_rel*np.sin(phi))**2*dr
#         # azim_torque+= 4*np.pi*rho*(ri**3)*b*omega*V_rel*np.sin(phi)*dr

#     power = azim_torque * omega
#     return thrust, azim_torque, axial_torque, power, stalled

