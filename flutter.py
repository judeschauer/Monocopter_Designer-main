"""
=============================================================================
ROTOR BLADE FLUTTER SPEED CALCULATOR
Bending-Torsion Flutter using p-k Method with Strip Theory + Theodorsen
=============================================================================
Author: Aeroelasticity Analysis Tool
Method: p-k iterative method, strip theory aerodynamics, modal approach
Reference: Theodorsen (1935), Johnson "Helicopter Theory", Bielawa "Rotary
           Wing Structural Dynamics and Aeroelasticity"
=============================================================================
"""

import numpy as np
from scipy import linalg
from scipy.special import hankel2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# ============  ROTOR BLADE INPUT PARAMETERS  ================
# ============================================================

def get_default_params():
    """
    Returns default parameters for a medium helicopter rotor blade
    (representative of a UH-60 class rotor blade).
    Modify these values for your specific rotor.
    """
    params = {
        # --- Rotor geometry ---
        'R'         : 2.79,       # Blade radius [m]
        'c'         : 0.3,      # Blade chord [m] (constant for simplicity)
        'n_stations': 50,         # Number of spanwise stations for integration
        'r_root'    : 0.2,        # Root cutout as fraction of R [-]

        # --- Rotor operating conditions ---
        'Omega'     : 20.94,       # Rotor angular velocity [rad/s] (200 RPM)
        'rho'       : 1.225,      # Air density [kg/m³]

        # --- Blade structural properties (uniform, modify for tapered) ---
        'm'         : 0.716,       # Mass per unit span [kg/m]
        'EI'        : 1.5e5,      # Flapwise bending stiffness [N·m²]
        'GJ'        : 8.0e3,      # Torsional stiffness [N·m²/rad]
        'I_alpha'   : 0.12,       # Mass moment of inertia per span [kg·m]
        'x_alpha'   : 0.3,      # CG offset from EA, normalized by b [-]
                                  # negative = CG ahead of EA (stable)
        'a'         : -0.5,       # EA offset from mid-chord, normalized by b [-]
                                  # negative = EA forward of mid-chord

        # --- Structural damping ---
        'zeta_h'    : 0.01,       # Bending structural damping ratio [-]
        'zeta_alpha': 0.01,       # Torsion structural damping ratio [-]

        # --- Pitch-flap coupling ---
        'delta3_deg': 0.0,        # Delta-3 angle [degrees], 0 = no coupling

        # --- Aerodynamics ---
        'Cl_alpha'  : 2 * np.pi,  # Lift curve slope [1/rad]

        # --- Analysis settings ---
        'V_min'     : 20.0,       # Min tip speed for sweep [m/s]
        'V_max'     : 100.0,      # Max tip speed for sweep [m/s]
        'n_V'       : 100,        # Number of velocity steps
        'n_modes'   : 2,          # Number of modes: 1=flap+torsion, can use 2+2
    }
    return params


# ============================================================
# ============  THEODORSEN FUNCTION  =========================
# ============================================================

def theodorsen_exact(k):
    """
    Exact Theodorsen lift-deficiency function C(k) = F(k) + iG(k)
    Using Hankel functions of the second kind.
    k: reduced frequency = omega*b/U
    """
    if k < 1e-6:
        return complex(1.0, 0.0)
    H1 = hankel2(1, k)
    H0 = hankel2(0, k)
    C = H1 / (H1 + 1j * H0)
    return C


def theodorsen_jones(k):
    """
    Jones approximation of Theodorsen function (faster, good for k < 1)
    """
    if k < 1e-6:
        return complex(1.0, 0.0)
    C = (1.0 - 0.165 / (1.0 - 0.0455j / k)
             - 0.335 / (1.0 - 0.3j   / k))
    return C


# ============================================================
# ============  STRUCTURAL NATURAL FREQUENCIES  ==============
# ============================================================

def compute_rotating_frequencies(params):
    """
    Estimate 1st flap bending and 1st torsion natural frequencies
    for a rotating beam using Rayleigh-Ritz / closed-form estimates.
    For a tapered beam, FEM would replace this.
    """
    R        = params['R']
    Omega    = params['Omega']
    m        = params['m']
    EI       = params['EI']
    GJ       = params['GJ']
    I_alpha  = params['I_alpha']
    r_root   = params['r_root'] * R

    # Non-rotating 1st flap frequency (cantilever beam)
    # omega_h0 = (1.875)^2 * sqrt(EI / (m * R^4))
    beta1 = 1.8751
    omega_h0 = (beta1**2 / R**2) * np.sqrt(EI / m)

    # Centrifugal stiffening (Southwell coefficient)
    # omega_h^2 = omega_h0^2 + gamma_h * Omega^2
    # For uniform blade, gamma_h ≈ 1.0 (Southwell)
    gamma_h = 1.0
    omega_h = np.sqrt(omega_h0**2 + gamma_h * Omega**2)

    # Non-rotating 1st torsion (cantilever)
    # omega_alpha0 = (pi/2) * sqrt(GJ / (I_alpha * R^2))
    omega_alpha0 = (np.pi / (2 * R)) * np.sqrt(GJ / I_alpha)

    # Torsion: centrifugal effect small, small correction
    # omega_alpha^2 ≈ omega_alpha0^2 + 0.25 * Omega^2 * (e_cg/b) terms
    omega_alpha = omega_alpha0  # conservative (no centrifugal torsion stiffening)

    freq_h     = omega_h     / (2 * np.pi)
    freq_alpha = omega_alpha / (2 * np.pi)
    freq_1rev  = Omega       / (2 * np.pi)

    print(f"\n{'='*60}")
    print(f"  STRUCTURAL NATURAL FREQUENCIES (ROTATING)")
    print(f"{'='*60}")
    print(f"  Rotor speed         : {Omega:.2f} rad/s  ({Omega*60/(2*np.pi):.1f} RPM)")
    print(f"  1/rev frequency     : {freq_1rev:.3f} Hz  (1.000 /rev)")
    print(f"  1st Flap bending    : {freq_h:.3f} Hz  ({freq_h/freq_1rev:.3f} /rev)")
    print(f"  1st Torsion         : {freq_alpha:.3f} Hz  ({freq_alpha/freq_1rev:.3f} /rev)")
    print(f"  Frequency ratio γ   : {omega_h/omega_alpha:.4f}  (ωh/ωα)")
    print(f"{'='*60}")

    if omega_h / omega_alpha > 0.9:
        print("  ⚠️  WARNING: Frequency ratio close to 1 — flutter risk HIGH")

    return omega_h, omega_alpha


# ============================================================
# ============  GENERALIZED MASS & STIFFNESS  ================
# ============================================================

def compute_generalized_matrices(params, omega_h, omega_alpha):
    """
    Generalized 2x2 structural mass, damping and stiffness matrices
    for the (flap, torsion) 2-DOF system.
    Using assumed mode shapes: W1(r) = (r/R)^2, Phi1(r) = sin(pi*r/(2R))
    """
    R       = params['R']
    c       = params['c']
    b       = c / 2.0
    m       = params['m']
    I_alpha = params['I_alpha']
    x_alpha = params['x_alpha']     # CG offset (normalized by b)
    a       = params['a']           # EA offset (normalized by b)
    zeta_h  = params['zeta_h']
    zeta_a  = params['zeta_alpha']
    r_root  = params['r_root'] * R

    r = np.linspace(r_root, R, params['n_stations'])
    dr = r[1] - r[0]

    # Mode shapes
    W   = (r / R)**2                           # 1st flap shape
    Phi = np.sin(np.pi * r / (2 * R))          # 1st torsion shape

    # Normalize
    norm_W   = np.sqrt(np.trapz(m * W**2,   r))
    norm_Phi = np.sqrt(np.trapz(I_alpha * Phi**2, r))
    W   = W   / norm_W
    Phi = Phi / norm_Phi

    # Generalized masses
    Mhh = np.trapz(m       * W**2,              r)   # ≈ 1 (normalized)
    Maa = np.trapz(I_alpha * Phi**2,             r)   # ≈ 1
    Mha = np.trapz(m * b * x_alpha * W * Phi,   r)   # coupling via CG offset
    Mah = Mha

    M = np.array([[Mhh, Mha],
                  [Mah, Maa]])

    # Generalized stiffness
    Khh = omega_h**2     * Mhh
    Kaa = omega_alpha**2 * Maa
    K   = np.array([[Khh, 0.0],
                    [0.0, Kaa]])

    # Structural damping (proportional)
    Chh = 2 * zeta_h  * omega_h     * Mhh
    Caa = 2 * zeta_a  * omega_alpha * Maa
    C   = np.array([[Chh, 0.0],
                    [0.0, Caa]])

    return M, C, K, r, W, Phi, b


# ============================================================
# ============  AERODYNAMIC INFLUENCE COEFFICIENTS  ==========
# ============================================================

def compute_aero_matrices(params, omega, V_tip, r, W, Phi, b):
    """
    Compute generalized aerodynamic force (GAF) matrices via spanwise
    strip integration with Theodorsen unsteady aerodynamics.

    Returns A_qq, A_qp, A_pq, A_pp (2x2 complex AIC submatrices)
    where q=flap (h), p=torsion (alpha).
    """
    R       = params['R']
    c       = params['c']
    b_c     = c / 2.0
    rho     = params['rho']
    a       = params['a']
    Omega   = params['Omega']
    r_root  = params['r_root'] * R

    # Local velocity at each station
    U = Omega * r   # hover assumption

    # Avoid division by zero at root
    U = np.maximum(U, 1.0)

    # Local reduced frequency at each station
    k_local = np.abs(omega) * b_c / U
    k_local = np.maximum(k_local, 1e-6)

    # Theodorsen function at each station
    Ck = np.array([theodorsen_jones(ki) for ki in k_local])

    # Strip aerodynamic forces (per unit span) in complex form
    # Lift on flap mode from flap motion (h):
    #   L_h = rho*pi*b^2 * (-omega^2*h) + 2*pi*rho*U*b*C(k)*(i*omega*h)
    # Simplified generalized aero loading (frequency domain):

    # Aerodynamic coefficients per strip (Theodorsen formulation)
    # For harmonic q = q0*exp(i*omega*t):

    # Non-circulatory (added mass) terms:
    L_nc_h  =  np.pi * rho * b_c**2                         # coeff of (-omega^2 * h)
    L_nc_a  =  np.pi * rho * b_c**2 * (-b_c * a)            # coeff of (-omega^2 * alpha)
    M_nc_h  =  np.pi * rho * b_c**2 * (b_c * a)             # moment coeff of (-omega^2*h)
    M_nc_a  = -np.pi * rho * b_c**2 * (b_c**2*(0.125+a**2)) # moment coeff (-omega^2*alpha)

    # Circulatory (Theodorsen) terms:
    w_eff_h  = 1j * omega                                    # from h-dot
    w_eff_a  = U + 1j * omega * b_c * (0.5 - a)             # from alpha

    L_c_h  =  2 * np.pi * rho * U * b_c * Ck * w_eff_h
    L_c_a  =  2 * np.pi * rho * U * b_c * Ck * w_eff_a
    M_c_h  =  2 * np.pi * rho * U * b_c**2 * (a + 0.5) * Ck * w_eff_h
    M_c_a  =  2 * np.pi * rho * U * b_c**2 * (a + 0.5) * Ck * w_eff_a

    # Total aero force per unit span on each mode (frequency domain, p=i*omega):
    p = 1j * omega
    # Convention: [L, M] = aero([h, alpha]) in complex frequency domain
    # Lift on strip:
    dLh = L_nc_h * p**2 + L_c_h   # due to unit h
    dLa = L_nc_a * p**2 + L_c_a   # due to unit alpha
    dMh = M_nc_h * p**2 + M_c_h
    dMa = M_nc_a * p**2 + M_c_a

    # Generalized forces via modal projection (spanwise integration)
    Qhh = np.trapz(dLh * W * W,   r)
    Qha = np.trapz(dLa * W * Phi, r)
    Qah = np.trapz(dMh * Phi * W, r)
    Qaa = np.trapz(dMa * Phi * Phi, r)

    Q = np.array([[Qhh, Qha],
                  [Qah, Qaa]])
    return Q


# ============================================================
# ============  p-k FLUTTER SOLVER  ==========================
# ============================================================

def pk_flutter_solver(params):
    """
    p-k method flutter solution.
    Sweeps tip speed V_tip and finds eigenvalues p = sigma + i*omega.
    Flutter occurs when sigma (real part) crosses zero.
    """
    omega_h, omega_alpha = compute_rotating_frequencies(params)
    M, C, K, r, W, Phi, b = compute_generalized_matrices(params, omega_h, omega_alpha)

    V_range = np.linspace(params['V_min'], params['V_max'], params['n_V'])
    R       = params['R']
    Omega   = params['Omega']

    n_dof   = 2
    n_modes = n_dof

    # Storage: eigenvalues for each mode at each velocity
    sigma_hist = np.zeros((n_modes, len(V_range)))  # damping (real part)
    omega_hist = np.zeros((n_modes, len(V_range)))  # frequency (imag part)

    print(f"\n{'='*60}")
    print(f"  p-k FLUTTER ANALYSIS")
    print(f"{'='*60}")
    print(f"  Velocity sweep: {params['V_min']:.0f} – {params['V_max']:.0f} m/s")
    print(f"  Stations: {params['n_stations']}")

    flutter_found = False
    V_flutter     = None
    omega_flutter = None
    mode_flutter  = None

    for iv, V_tip in enumerate(V_range):
        # Initial guess for omega from structural frequencies
        omega_guess = [omega_h, omega_alpha]

        p_converged = []

        for imode in range(n_modes):
            omega_k = omega_guess[imode]

            # p-k iteration
            for _ in range(50):
                Q = compute_aero_matrices(params, omega_k, V_tip, r, W, Phi, b)

                # Build state-space matrices
                # [M]p² + [C - Q_damp]p + [K - Q_stiff] = 0
                # Split Q into p² terms (added mass), p terms (aero damping),
                # and constant terms (aero stiffness)
                # Full aeroelastic matrix: A2*p² + A1*p + A0 = 0

                # Q already encodes full complex aero load for p=i*omega
                # Build companion eigenvalue problem
                # A = [0, I; -M^-1*(K-Qk), -M^-1*(C-Qc)]

                # Extract real aero stiffness and damping from Q(p=i*omega)
                # Q_eff = Q (already includes p dependence via i*omega)
                # For p-k: use real part as stiffness, imag/omega as damping

                Q_stiff = np.real(Q)
                Q_damp  = np.imag(Q) / (omega_k + 1e-10)

                A0 = K - Q_stiff
                A1 = C - Q_damp
                A2 = M

                # State space: x = [q, q_dot]
                # [A2 0 ][q_ddot]   [A1 A0][q_dot]
                # [0  I ][q_dot ]  +[-I  0][q    ] = 0
                Ass = np.zeros((2*n_dof, 2*n_dof))
                try:
                    M_inv = np.linalg.inv(A2)
                except:
                    M_inv = np.eye(n_dof)

                Ass[:n_dof, :n_dof]    = -M_inv @ A1
                Ass[:n_dof, n_dof:]    = -M_inv @ A0
                Ass[n_dof:, :n_dof]    =  np.eye(n_dof)
                Ass[n_dof:, n_dof:]    =  np.zeros((n_dof, n_dof))

                evals = np.linalg.eigvals(Ass)
                # Pick eigenvalue closest to current omega_k
                omega_evals = np.abs(np.imag(evals))
                idx = np.argmin(np.abs(omega_evals - np.abs(omega_k)))

                p_new    = evals[idx]
                omega_new = np.abs(np.imag(p_new))

                if np.abs(omega_new - np.abs(omega_k)) / (np.abs(omega_k) + 1e-10) < 1e-5:
                    break
                omega_k = omega_new if omega_new > 0.1 else omega_k

            p_converged.append(p_new)

        # Sort by frequency magnitude
        p_converged.sort(key=lambda x: np.abs(np.imag(x)))

        for imode in range(n_modes):
            p = p_converged[imode] if imode < len(p_converged) else complex(0,0)
            sigma_hist[imode, iv] = np.real(p)
            omega_hist[imode, iv] = np.abs(np.imag(p)) / (2 * np.pi)

        # Check flutter crossing
        if iv > 0 and not flutter_found:
            for imode in range(n_modes):
                if (sigma_hist[imode, iv-1] < 0 and sigma_hist[imode, iv] >= 0):
                    # Linear interpolation
                    s0, s1 = sigma_hist[imode, iv-1], sigma_hist[imode, iv]
                    v0, v1 = V_range[iv-1], V_range[iv]
                    V_flutter     = v0 - s0 * (v1 - v0) / (s1 - s0)
                    omega_flutter = np.interp(V_flutter, [v0,v1],
                                              [omega_hist[imode,iv-1], omega_hist[imode,iv]])
                    mode_flutter  = imode + 1
                    flutter_found = True

    return V_range, sigma_hist, omega_hist, V_flutter, omega_flutter, mode_flutter, omega_h, omega_alpha


# ============================================================
# ============  RESULTS & REPORTING  =========================
# ============================================================

def print_results(params, V_flutter, omega_flutter, mode_flutter, omega_h, omega_alpha):
    R     = params['R']
    Omega = params['Omega']
    c     = params['c']
    b     = c / 2.0
    rho   = params['rho']

    V_tip_max = Omega * R
    freq_1rev = Omega / (2 * np.pi)

    print(f"\n{'='*60}")
    print(f"  FLUTTER ANALYSIS RESULTS")
    print(f"{'='*60}")

    if V_flutter is not None:
        mu = params['m'] / (rho * np.pi * b**2)
        k_flutter = omega_flutter * 2*np.pi * b / V_flutter if V_flutter > 0 else 0

        print(f"  ✅ FLUTTER DETECTED")
        print(f"  Flutter speed (tip)   : {V_flutter:.2f} m/s")
        print(f"  Flutter speed (75%R)  : {0.75*V_flutter:.2f} m/s")
        print(f"  Flutter frequency     : {omega_flutter:.3f} Hz")
        print(f"  Flutter mode          : Mode {mode_flutter}")
        print(f"  Reduced freq k_F      : {k_flutter:.4f}")
        print(f"  Flutter Ω (rad/s)     : {V_flutter/R:.3f} rad/s")
        print(f"  Flutter RPM           : {V_flutter/R * 60/(2*np.pi):.1f} RPM")
        print(f"\n  --- Operating conditions ---")
        print(f"  Operating tip speed   : {V_tip_max:.2f} m/s  ({Omega*60/(2*np.pi):.1f} RPM)")
        print(f"  Flutter margin        : {(V_flutter/V_tip_max - 1)*100:.1f}%")

        margin = V_flutter / V_tip_max
        if margin >= 1.2:
            print(f"  ✅ SAFE — Flutter margin ≥ 20% (MIL-SPEC)")
        elif margin >= 1.0:
            print(f"  ⚠️  MARGINAL — Flutter speed above operating but < 20% margin")
        else:
            print(f"  ❌ UNSAFE — Flutter speed BELOW operating tip speed!")

        print(f"\n  --- Key aeroelastic parameters ---")
        print(f"  Mass ratio μ          : {mu:.2f}")
        print(f"  Frequency ratio ωh/ωα : {omega_h/omega_alpha:.4f}")
        print(f"  CG offset xα (×b)     : {params['x_alpha']:.4f}")
        print(f"  EA offset a (×b)      : {params['a']:.4f}")
    else:
        print(f"  ✅ NO FLUTTER in sweep range ({params['V_min']}–{params['V_max']} m/s)")
        print(f"  Increase V_max to extend search range.")

    print(f"{'='*60}\n")


# ============================================================
# ============  PLOTTING  ====================================
# ============================================================

def plot_results(params, V_range, sigma_hist, omega_hist,
                 V_flutter, omega_flutter, omega_h, omega_alpha):

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#0f1117')

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    COLORS = ['#00d4ff', '#ff6b35', '#00ff88', '#ffd700']
    BG     = '#0f1117'
    PANEL  = '#1a1d27'
    GRID   = '#2a2d3a'
    TEXT   = '#e8eaf0'

    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel(xlabel, color=TEXT, fontsize=9)
        ax.set_ylabel(ylabel, color=TEXT, fontsize=9)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.grid(True, color=GRID, linewidth=0.6, linestyle='--')
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.title.set_color(TEXT)

    mode_names = ['Mode 1: Flap-dominant', 'Mode 2: Torsion-dominant']
    Omega      = params['Omega']
    R          = params['R']
    V_tip_op   = Omega * R

    # --- Plot 1: Damping (sigma) vs Velocity ---
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, 'Aeroelastic Damping vs Tip Speed', 'Tip Speed [m/s]', 'Damping σ [rad/s]')
    for i in range(sigma_hist.shape[0]):
        ax1.plot(V_range, sigma_hist[i], color=COLORS[i], lw=2,
                 label=mode_names[i] if i < len(mode_names) else f'Mode {i+1}')
    ax1.axhline(0, color='white', lw=1.2, linestyle='-', alpha=0.7)
    ax1.axvline(V_tip_op, color='#ffd700', lw=1.5, linestyle=':', label=f'Operating ({V_tip_op:.0f} m/s)')
    if V_flutter:
        ax1.axvline(V_flutter, color='#ff4444', lw=2, linestyle='--', label=f'Flutter ({V_flutter:.1f} m/s)')
        ax1.axvspan(V_flutter, V_range[-1], alpha=0.08, color='red')
    ax1.legend(fontsize=7.5, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # --- Plot 2: Frequency vs Velocity ---
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, 'Aeroelastic Frequency vs Tip Speed', 'Tip Speed [m/s]', 'Frequency [Hz]')
    for i in range(omega_hist.shape[0]):
        ax2.plot(V_range, omega_hist[i], color=COLORS[i], lw=2,
                 label=mode_names[i] if i < len(mode_names) else f'Mode {i+1}')
    # Mark structural frequencies
    freq_h     = omega_h     / (2*np.pi)
    freq_alpha = omega_alpha / (2*np.pi)
    ax2.axhline(freq_h,     color=COLORS[0], lw=1, linestyle=':', alpha=0.6, label=f'ωh={freq_h:.2f}Hz')
    ax2.axhline(freq_alpha, color=COLORS[1], lw=1, linestyle=':', alpha=0.6, label=f'ωα={freq_alpha:.2f}Hz')
    ax2.axvline(V_tip_op, color='#ffd700', lw=1.5, linestyle=':', label=f'Operating')
    if V_flutter:
        ax2.axvline(V_flutter, color='#ff4444', lw=2, linestyle='--', label=f'Flutter')
    ax2.legend(fontsize=7.5, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # --- Plot 3: Theodorsen function F(k), G(k) ---
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, 'Theodorsen Function C(k)', 'Reduced Frequency k', 'Value')
    k_arr = np.linspace(0.001, 2.0, 300)
    F_arr = np.array([np.real(theodorsen_jones(k)) for k in k_arr])
    G_arr = np.array([np.imag(theodorsen_jones(k)) for k in k_arr])
    ax3.plot(k_arr, F_arr, color=COLORS[0], lw=2, label='F(k) — Real part')
    ax3.plot(k_arr, G_arr, color=COLORS[1], lw=2, label='G(k) — Imaginary part')
    ax3.axhline(0, color='white', lw=0.8, alpha=0.5)
    ax3.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # --- Plot 4: Summary panel ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(PANEL)
    ax4.axis('off')
    for spine in ax4.spines.values():
        spine.set_edgecolor(GRID)

    freq_1rev = Omega / (2*np.pi)

    summary_lines = [
        ("ROTOR BLADE FLUTTER SUMMARY", None, '#00d4ff', 11, 'bold'),
        ("─"*38, None, GRID, 9, 'normal'),
        ("BLADE GEOMETRY", None, '#ffd700', 9, 'bold'),
        (f"  Blade radius R",         f"{params['R']:.2f} m",           TEXT, 9, 'normal'),
        (f"  Chord c",                f"{params['c']:.3f} m",           TEXT, 9, 'normal'),
        (f"  Semi-chord b",           f"{params['c']/2:.3f} m",         TEXT, 9, 'normal'),
        ("STRUCTURAL", None, '#ffd700', 9, 'bold'),
        (f"  Mass/span",              f"{params['m']:.2f} kg/m",        TEXT, 9, 'normal'),
        (f"  EI (flapwise)",          f"{params['EI']:.2e} N·m²",       TEXT, 9, 'normal'),
        (f"  GJ (torsion)",           f"{params['GJ']:.2e} N·m²/rad",   TEXT, 9, 'normal'),
        (f"  CG offset xα",           f"{params['x_alpha']:.3f}×b",     TEXT, 9, 'normal'),
        (f"  EA offset a",            f"{params['a']:.3f}×b",           TEXT, 9, 'normal'),
        ("FREQUENCIES", None, '#ffd700', 9, 'bold'),
        (f"  1/rev",                  f"{freq_1rev:.3f} Hz",            TEXT, 9, 'normal'),
        (f"  1st Flap ωh",            f"{freq_h:.3f} Hz  ({freq_h/freq_1rev:.2f}/rev)",  TEXT, 9, 'normal'),
        (f"  1st Torsion ωα",         f"{freq_alpha:.3f} Hz  ({freq_alpha/freq_1rev:.2f}/rev)", TEXT, 9, 'normal'),
        (f"  ωh/ωα ratio",            f"{omega_h/omega_alpha:.4f}",      TEXT, 9, 'normal'),
        ("FLUTTER RESULT", None, '#ffd700', 9, 'bold'),
    ]

    if V_flutter:
        margin = (V_flutter / V_tip_op - 1) * 100
        safe_color = '#00ff88' if margin >= 20 else ('#ffd700' if margin >= 0 else '#ff4444')
        summary_lines += [
            (f"  Flutter tip speed",   f"{V_flutter:.1f} m/s",           safe_color, 9, 'bold'),
            (f"  Flutter freq",        f"{omega_flutter:.3f} Hz",         safe_color, 9, 'normal'),
            (f"  Operating tip speed", f"{V_tip_op:.1f} m/s",            TEXT,       9, 'normal'),
            (f"  Flutter margin",      f"{margin:+.1f}%",                 safe_color, 9, 'bold'),
            (f"  Status",  "✅ SAFE" if margin>=20 else ("⚠ MARGINAL" if margin>=0 else "❌ UNSAFE"),
             safe_color, 9, 'bold'),
        ]
    else:
        summary_lines.append(("  No flutter in sweep", "SAFE ✅", '#00ff88', 9, 'bold'))

    y = 0.97
    for line in summary_lines:
        label, val, color, fs, fw = line
        if val is None:
            ax4.text(0.05, y, label, transform=ax4.transAxes,
                     color=color, fontsize=fs, fontweight=fw,
                     verticalalignment='top', fontfamily='monospace')
        else:
            ax4.text(0.05, y, label, transform=ax4.transAxes,
                     color=TEXT, fontsize=fs, fontweight='normal',
                     verticalalignment='top', fontfamily='monospace')
            ax4.text(0.72, y, val, transform=ax4.transAxes,
                     color=color, fontsize=fs, fontweight=fw,
                     verticalalignment='top', fontfamily='monospace')
        y -= 0.055

    fig.suptitle('Rotor Blade Bending-Torsion Flutter Analysis  |  p-k Method',
                 color=TEXT, fontsize=13, fontweight='bold', y=0.98)

    plt.savefig('./rotor_flutter_analysis.png',
                dpi=150, bbox_inches='tight', facecolor=BG)
    print("  Plot saved: rotor_flutter_analysis.png")
    plt.close()


# ============================================================
# ============  SENSITIVITY STUDY  ===========================
# ============================================================

def sensitivity_study(base_params):
    """Vary key parameters and show their effect on flutter speed."""
    print(f"\n{'='*60}")
    print(f"  SENSITIVITY STUDY")
    print(f"{'='*60}")
    print(f"  {'Parameter':<30} {'Value':<15} {'V_flutter (m/s)':<18} {'Margin %'}")
    print(f"  {'─'*30} {'─'*15} {'─'*18} {'─'*10}")

    # Parameters to vary
    studies = [
        ('x_alpha (CG offset)',   'x_alpha',  [-0.05, -0.02,  0.0,  0.02]),
        ('a (EA position)',       'a',         [-0.4,  -0.2,   0.0,  0.1]),
        ('GJ (torsion stiff.)',   'GJ',        [4e3,   8e3,   12e3, 16e3]),
        ('Rotor speed Omega',     'Omega',     [22.0,  27.0,  30.0, 35.0]),
    ]

    for param_name, param_key, values in studies:
        for val in values:
            p = base_params.copy()
            p['x_alpha'] = base_params['x_alpha']  # reset
            p[param_key] = val
            p['n_V'] = 60  # faster sweep
            p['n_stations'] = 12
            try:
                _, _, _, V_f, _, _, _, _ = pk_flutter_solver(p)
                V_op = p['Omega'] * p['R']
                margin = (V_f / V_op - 1)*100 if V_f else float('nan')
                V_f_str = f"{V_f:.1f}" if V_f else ">300"
                print(f"  {param_name:<30} {val:<15.4g} {V_f_str:<18} {margin:+.1f}%")
            except Exception as e:
                print(f"  {param_name:<30} {val:<15.4g} ERROR")
        print()


# ============================================================
# ============  MAIN  ========================================
# ============================================================

def main():
    print("\n" + "="*60)
    print("   ROTOR BLADE FLUTTER SPEED CALCULATOR")
    print("   Method: p-k  |  Aero: Strip + Theodorsen")
    print("="*60)

    # Load parameters (modify get_default_params() for your rotor)
    params = get_default_params()

    # --- Run flutter analysis ---
    (V_range, sigma_hist, omega_hist,
     V_flutter, omega_flutter, mode_flutter,
     omega_h, omega_alpha) = pk_flutter_solver(params)

    # --- Print results ---
    print_results(params, V_flutter, omega_flutter, mode_flutter, omega_h, omega_alpha)

    # --- Plot ---
    plot_results(params, V_range, sigma_hist, omega_hist,
                 V_flutter, omega_flutter, omega_h, omega_alpha)

    # --- Sensitivity study ---
    sensitivity_study(params)

    print("\nAnalysis complete.")
    return V_flutter, omega_flutter


if __name__ == '__main__':
    V_F, omega_F = main()