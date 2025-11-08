"""
Physical Constants and Parameters for Electrochemical Systems
Based on equations from the paper and standard references.
"""

import numpy as np

# ==================== UNIVERSAL CONSTANTS ====================

# Faraday constant [C/mol]
F = 96485.3329

# Universal gas constant [J/(mol·K)]
R = 8.314462618

# Standard temperature [K]
T_STD = 298.15

# Standard pressure [Pa]
P_STD = 101325.0

# Avogadro constant [1/mol]
N_A = 6.02214076e23


# ==================== PEMFC PARAMETERS ====================

class PEMFCParams:
    """
    Proton Exchange Membrane Fuel Cell parameters
    """
    
    # Standard cell potential [V] at 298.15 K
    E0_STANDARD = 1.229
    
    # Number of electrons in H2/O2 reaction
    n_electrons = 2
    
    # Temperature-dependent E0 [V/K]
    dE0_dT = -0.00085
    
    # Typical cathode transfer coefficient (dimensionless)
    alpha_cathode = 0.5
    
    # Typical anode transfer coefficient (dimensionless)
    alpha_anode = 0.5
    
    # Exchange current density ranges for Pt/C cathode [A/cm²]
    # From literature: varies with T, catalyst loading, membrane
    i0_cathode_min = 1e-9
    i0_cathode_max = 1e-6
    i0_cathode_typical = 1e-7  # At 60-80°C
    
    # Exchange current density for anode (typically much higher) [A/cm²]
    i0_anode_typical = 1e-3
    
    # Ohmic resistance ranges [Ω·cm²]
    R_ohm_min = 0.05
    R_ohm_max = 0.30
    R_ohm_typical = 0.15
    
    # Limiting current density ranges [A/cm²]
    i_L_min = 1.0
    i_L_max = 3.0
    i_L_typical = 2.0
    
    # Membrane properties
    class Membrane:
        # Nafion conductivity at fully hydrated state [S/m]
        kappa_max = 10.0  # At 80°C, λ=14
        
        # Typical thickness [μm]
        thickness_nafion117 = 183
        thickness_nafion212 = 50
        
        # Water content (λ = H2O per SO3-)
        lambda_min = 3
        lambda_max = 14
    
    # Gas diffusion layer (GDL)
    class GDL:
        # Porosity (dimensionless)
        porosity = 0.7
        
        # Tortuosity
        tortuosity = 2.0
        
        # Typical thickness [μm]
        thickness = 300
        
        # Permeability [m²]
        permeability = 1e-12
    
    # Operating conditions
    class Operating:
        # Temperature range [°C]
        T_min = 50
        T_max = 90
        T_typical = 70
        
        # Pressure range [atm]
        p_min = 1.0
        p_max = 3.0
        
        # Stoichiometry ranges
        stoich_anode_min = 1.2
        stoich_anode_max = 2.0
        stoich_cathode_min = 2.0
        stoich_cathode_max = 3.0


# ==================== VRFB PARAMETERS ====================

class VRFBParams:
    """
    Vanadium Redox Flow Battery parameters
    """
    
    # Standard cell potential [V]
    E0_standard = 1.26
    
    # Number of electrons transferred
    n_electrons = 1
    
    # Temperature coefficient [V/K]
    dE0_dT = 0.0001
    
    # Exchange current density ranges [A/cm²]
    # For carbon felt electrodes
    i0_min = 1e-3
    i0_max = 1e-2
    i0_typical = 5e-3
    
    # Transfer coefficients (both half-reactions similar)
    alpha_negative = 0.5
    alpha_positive = 0.5
    
    # Electrolyte properties (typical: 2M vanadium in 4M H2SO4)
    class Electrolyte:
        # Concentration [M]
        vanadium_conc = 2.0
        sulfuric_acid_conc = 4.0
        
        # Density [kg/m³]
        density = 1350
        
        # Viscosity [Pa·s]
        viscosity = 0.005
        
        # Diffusivity [m²/s]
        diffusivity_V2 = 2.4e-10
        diffusivity_V3 = 2.4e-10
        diffusivity_VO2 = 3.9e-10
        diffusivity_VO2plus = 3.9e-10
    
    # Electrode properties (carbon felt)
    class Electrode:
        # Thickness range [mm]
        thickness_min = 3
        thickness_max = 10
        thickness_typical = 5
        
        # Porosity
        porosity = 0.93
        
        # Permeability [m²]
        permeability = 1e-9
        
        # Specific surface area [m²/m³]
        specific_area = 3e5
        
        # Electrical conductivity [S/m]
        conductivity = 500
    
    # Membrane properties (Nafion)
    class Membrane:
        # Thickness [μm]
        thickness = 125
        
        # Ion conductivity [S/m]
        conductivity = 10.0
        
        # Area resistance [Ω·cm²]
        area_resistance = 0.125
    
    # Flow and hydraulic parameters
    class Flow:
        # Flow rate range [mL/s]
        Q_min = 10
        Q_max = 50
        Q_typical = 30
        
        # Channel dimensions [mm]
        channel_width = 5.0
        channel_depth = 3.0
        channel_length = 100.0
        
        # Hydraulic diameter [mm]
        d_h = 3.75  # For rectangular channel
        
        # Sherwood correlation coefficients
        # Sh = a * Re^b * Sc^(1/3)
        sherwood_a = 1.5
        sherwood_b = 0.5
    
    # Operating conditions
    class Operating:
        # Temperature range [°C]
        T_min = 15
        T_max = 50
        T_typical = 25
        
        # Current density range [mA/cm²]
        i_min = 50
        i_max = 300
        i_typical = 150
        
        # State of charge range
        SOC_min = 0.1
        SOC_max = 0.9
        
        # Pressure drop limit [Pa]
        delta_p_max = 5000


# ==================== NOISE PARAMETERS ====================

class NoiseParams:
    """
    Parameters for adding realistic measurement noise
    """
    
    # PEMFC voltage noise
    pemfc_voltage_std = 0.010  # [V] = 10 mV
    
    # PEMFC current noise
    pemfc_current_std_rel = 0.02  # 2% relative
    
    # VRFB voltage noise
    vrfb_voltage_std = 0.005  # [V] = 5 mV
    
    # VRFB efficiency noise
    vrfb_efficiency_std_rel = 0.02  # 2% relative
    
    # Random seed for reproducibility
    random_seed = 42


# ==================== LITERATURE PARAMETER RANGES ====================

# Temperature-dependent exchange current density (Arrhenius)
# i0(T) = i0_ref * exp(-Ea / (R*T))

# Activation energy ranges [J/mol]
ACTIVATION_ENERGY = {
    "PEMFC_cathode": 66000,  # 40-80 kJ/mol typical
    "PEMFC_anode": 20000,
    "VRFB_negative": 45000,
    "VRFB_positive": 48000,
}


# ==================== UTILITY FUNCTIONS ====================

def celsius_to_kelvin(T_celsius: float) -> float:
    """Convert Celsius to Kelvin"""
    return T_celsius + 273.15


def kelvin_to_celsius(T_kelvin: float) -> float:
    """Convert Kelvin to Celsius"""
    return T_kelvin - 273.15


def nernst_potential_pemfc(T: float, p_H2: float, p_O2: float, a_H2O: float = 1.0) -> float:
    """
    Calculate Nernst potential for PEMFC (Equation 4 from paper)
    
    Args:
        T: Temperature [K]
        p_H2: Hydrogen partial pressure [atm]
        p_O2: Oxygen partial pressure [atm]
        a_H2O: Water activity (default 1.0)
    
    Returns:
        E_Nernst: Nernst potential [V]
    """
    E0_T = PEMFCParams.E0_STANDARD + PEMFCParams.dE0_dT * (T - T_STD)
    RT_over_2F = R * T / (2 * F)
    E_nernst = E0_T + RT_over_2F * np.log((p_H2 * np.sqrt(p_O2)) / a_H2O)
    return E_nernst


def nernst_potential_vrfb(T: float, SOC: float) -> float:
    """
    Calculate Nernst potential for VRFB (Equation 6 from paper)
    
    Args:
        T: Temperature [K]
        SOC: State of charge [0-1]
    
    Returns:
        E_Nernst: Nernst potential [V]
    """
    E0_T = VRFBParams.E0_standard + VRFBParams.dE0_dT * (T - T_STD)
    RT_over_F = R * T / F
    
    # Concentration ratio (assuming ideal behavior)
    # [V3+][VO2+] / [V2+][VO2+]
    if SOC <= 0.01:
        SOC = 0.01  # Avoid log(0)
    if SOC >= 0.99:
        SOC = 0.99
    
    concentration_ratio = (SOC * SOC) / ((1 - SOC) * (1 - SOC))
    E_nernst = E0_T + RT_over_F * np.log(concentration_ratio)
    return E_nernst


def arrhenius_i0(i0_ref: float, Ea: float, T: float, T_ref: float = 298.15) -> float:
    """
    Calculate temperature-dependent exchange current density
    
    Args:
        i0_ref: Reference exchange current density [A/cm²]
        Ea: Activation energy [J/mol]
        T: Temperature [K]
        T_ref: Reference temperature [K]
    
    Returns:
        i0: Exchange current density at T [A/cm²]
    """
    return i0_ref * np.exp(-Ea / R * (1/T - 1/T_ref))


# ==================== VALIDATION BOUNDS ====================

VALIDATION_BOUNDS = {
    "PEMFC": {
        "voltage": (0.3, 1.2),  # [V]
        "current_density": (0.0, 2.0),  # [A/cm²]
        "temperature": (323.15, 363.15),  # [K] = 50-90°C
        "eta_act": (0.0, 0.5),  # [V]
        "R_ohm": (0.01, 0.5),  # [Ω·cm²]
    },
    "VRFB": {
        "voltage": (0.8, 1.6),  # [V]
        "current_density": (0.0, 0.4),  # [A/cm²]
        "temperature": (288.15, 323.15),  # [K] = 15-50°C
        "efficiency": (0.5, 1.0),  # dimensionless
        "SOC": (0.0, 1.0),  # dimensionless
    }
}
