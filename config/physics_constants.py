import numpy as np




F = 96485.3329


R = 8.314462618


T_STD = 298.15


P_STD = 101325.0


N_A = 6.02214076e23




class PEMFCParams:
    
    E0_STANDARD = 1.229
    
    
    n_electrons = 2
    
    
    dE0_dT = -0.00085
    
    
    alpha_cathode = 0.5
    
    
    alpha_anode = 0.5
    
    
    
    i0_cathode_min = 1e-9
    i0_cathode_max = 1e-6
    i0_cathode_typical = 1e-7  
    
    
    i0_anode_typical = 1e-3
    
    
    R_ohm_min = 0.05
    R_ohm_max = 0.30
    R_ohm_typical = 0.15
    
    
    i_L_min = 1.0
    i_L_max = 3.0
    i_L_typical = 2.0
    
    
    class Membrane:
        
        kappa_max = 10.0  
        
        
        thickness_nafion117 = 183
        thickness_nafion212 = 50
        
        
        lambda_min = 3
        lambda_max = 14
    
    
    class GDL:
        
        porosity = 0.7
        
        
        tortuosity = 2.0
        
        
        thickness = 300
        
        
        permeability = 1e-12
    
    
    class Operating:
        
        T_min = 50
        T_max = 90
        T_typical = 70
        
        
        p_min = 1.0
        p_max = 3.0
        
        
        stoich_anode_min = 1.2
        stoich_anode_max = 2.0
        stoich_cathode_min = 2.0
        stoich_cathode_max = 3.0




class VRFBParams:
    
    E0_standard = 1.26
    
    
    n_electrons = 1
    
    
    dE0_dT = 0.0001
    
    
    
    i0_min = 1e-3
    i0_max = 1e-2
    i0_typical = 5e-3
    
    
    alpha_negative = 0.5
    alpha_positive = 0.5
    
    
    class Electrolyte:
        
        vanadium_conc = 2.0
        sulfuric_acid_conc = 4.0
        
        
        density = 1350
        
        
        viscosity = 0.005
        
        
        diffusivity_V2 = 2.4e-10
        diffusivity_V3 = 2.4e-10
        diffusivity_VO2 = 3.9e-10
        diffusivity_VO2plus = 3.9e-10
    
    
    class Electrode:
        
        thickness_min = 3
        thickness_max = 10
        thickness_typical = 5
        
        
        porosity = 0.93
        
        
        permeability = 1e-9
        
        
        specific_area = 3e5
        
        
        conductivity = 500
    
    
    class Membrane:
        
        thickness = 125
        
        
        conductivity = 10.0
        
        
        area_resistance = 0.125
    
    
    class Flow:
        
        Q_min = 10
        Q_max = 50
        Q_typical = 30
        
        
        channel_width = 5.0
        channel_depth = 3.0
        channel_length = 100.0
        
        
        d_h = 3.75  
        
        
        
        sherwood_a = 1.5
        sherwood_b = 0.5
    
    
    class Operating:
        
        T_min = 15
        T_max = 50
        T_typical = 25
        
        
        i_min = 50
        i_max = 300
        i_typical = 150
        
        
        SOC_min = 0.1
        SOC_max = 0.9
        
        
        delta_p_max = 5000




class NoiseParams:
    
    
    pemfc_voltage_std = 0.010  
    
    
    pemfc_current_std_rel = 0.02  
    
    
    vrfb_voltage_std = 0.005  
    
    
    vrfb_efficiency_std_rel = 0.02  
    
    
    random_seed = 42








ACTIVATION_ENERGY = {
    "PEMFC_cathode": 66000,  
    "PEMFC_anode": 20000,
    "VRFB_negative": 45000,
    "VRFB_positive": 48000,
}




def celsius_to_kelvin(T_celsius: float) -> float:
    return T_celsius + 273.15


def kelvin_to_celsius(T_kelvin: float) -> float:
    return T_kelvin - 273.15


def nernst_potential_pemfc(T: float, p_H2: float, p_O2: float, a_H2O: float = 1.0) -> float:
    E0_T = PEMFCParams.E0_STANDARD + PEMFCParams.dE0_dT * (T - T_STD)
    RT_over_2F = R * T / (2 * F)
    E_nernst = E0_T + RT_over_2F * np.log((p_H2 * np.sqrt(p_O2)) / a_H2O)
    return E_nernst


def nernst_potential_vrfb(T: float, SOC: float) -> float:
    E0_T = VRFBParams.E0_standard + VRFBParams.dE0_dT * (T - T_STD)
    RT_over_F = R * T / F
    
    
    
    if SOC <= 0.01:
        SOC = 0.01  
    if SOC >= 0.99:
        SOC = 0.99
    
    concentration_ratio = (SOC * SOC) / ((1 - SOC) * (1 - SOC))
    E_nernst = E0_T + RT_over_F * np.log(concentration_ratio)
    return E_nernst


def arrhenius_i0(i0_ref: float, Ea: float, T: float, T_ref: float = 298.15) -> float:
    return i0_ref * np.exp(-Ea / R * (1/T - 1/T_ref))




VALIDATION_BOUNDS = {
    "PEMFC": {
        "voltage": (0.3, 1.2),  
        "current_density": (0.0, 2.0),  
        "temperature": (323.15, 363.15),  
        "eta_act": (0.0, 0.5),  
        "R_ohm": (0.01, 0.5),  
    },
    "VRFB": {
        "voltage": (0.8, 1.6),  
        "current_density": (0.0, 0.4),  
        "temperature": (288.15, 323.15),  
        "efficiency": (0.5, 1.0),  
        "SOC": (0.0, 1.0),  
    }
}
