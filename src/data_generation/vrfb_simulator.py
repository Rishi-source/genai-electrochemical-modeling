import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.physics_constants import (
    F, R, T_STD,
    VRFBParams,
    NoiseParams,
    celsius_to_kelvin,
    nernst_potential_vrfb,
    arrhenius_i0,
    ACTIVATION_ENERGY,
    VALIDATION_BOUNDS
)


class VRFBSimulator:    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        
        self.i0 = VRFBParams.i0_typical
        self.alpha = VRFBParams.alpha_positive
        self.T_celsius = VRFBParams.Operating.T_typical
    
    def nernst_potential(self, T_celsius: float, SOC: float) -> float:
        T_kelvin = celsius_to_kelvin(T_celsius)
        return nernst_potential_vrfb(T_kelvin, SOC)
    
    def activation_overpotential(
        self,
        i: float,
        i0: float,
        alpha: float,
        T_celsius: float
    ) -> float:
        T_kelvin = celsius_to_kelvin(T_celsius)
        RT_over_alphaF = (R * T_kelvin) / (alpha * F)
        
        
        i_safe = max(abs(i), 1e-8)
        eta_act = RT_over_alphaF * np.log(i_safe / i0)
        
        return eta_act
    
    def ohmic_resistance(self, delta_e: float) -> float:
        delta_e_m = delta_e * 1e-3  
        R_electrode = delta_e_m / (VRFBParams.Electrode.conductivity * 1e-4)  
        
        
        R_membrane = VRFBParams.Membrane.area_resistance
        
        return R_electrode + R_membrane
    
    def mass_transfer_coefficient(
        self,
        Q: float,
        T_celsius: float,
        d_h: float = None
    ) -> float:
        if d_h is None:
            d_h = VRFBParams.Flow.d_h
        
        
        Q_m3s = Q * 1e-6  
        d_h_m = d_h * 1e-3  
        
        
        rho = VRFBParams.Electrolyte.density
        mu = VRFBParams.Electrolyte.viscosity
        D = VRFBParams.Electrolyte.diffusivity_VO2  
        
        
        A_cross = VRFBParams.Flow.channel_width * VRFBParams.Flow.channel_depth * 1e-6  
        
        
        velocity = Q_m3s / A_cross if A_cross > 0 else 1e-3
        Re = (rho * velocity * d_h_m) / mu
        
        
        Sc = mu / (rho * D)
        
        
        a = VRFBParams.Flow.sherwood_a
        b = VRFBParams.Flow.sherwood_b
        Sh = a * (Re ** b) * (Sc ** (1/3))
        
        
        k_m = (Sh * D) / d_h_m
        
        return k_m
    
    def limiting_current(
        self,
        Q: float,
        T_celsius: float,
        c: float = None
    ) -> float:
        if c is None:
            c = VRFBParams.Electrolyte.vanadium_conc
        
        k_m = self.mass_transfer_coefficient(Q, T_celsius)
        
        
        c_m3 = c * 1000
        
        
        i_L = VRFBParams.n_electrons * F * k_m * c_m3
        
        
        i_L_cm2 = i_L / 1e4
        
        return i_L_cm2
    
    def mass_transfer_overpotential(
        self,
        i: float,
        i_L: float,
        T_celsius: float
    ) -> float:
        T_kelvin = celsius_to_kelvin(T_celsius)
        RT_over_nF = (R * T_kelvin) / (VRFBParams.n_electrons * F)
        
        
        ratio = min(abs(i) / i_L, 0.99)
        eta_mt = -RT_over_nF * np.log(1 - ratio)
        
        return eta_mt
    
    def cell_voltage(
        self,
        i: float,
        SOC: float,
        T_celsius: float,
        delta_e: float,
        Q: float,
        charge: bool = False
    ) -> Tuple[float, Dict[str, float]]:
        E_nernst = self.nernst_potential(T_celsius, SOC)
        
        
        i0_neg = VRFBParams.i0_typical
        i0_pos = VRFBParams.i0_typical
        alpha = VRFBParams.alpha_positive
        
        eta_act_neg = self.activation_overpotential(i, i0_neg, alpha, T_celsius)
        eta_act_pos = self.activation_overpotential(i, i0_pos, alpha, T_celsius)
        eta_act_total = eta_act_neg + eta_act_pos
        
        
        R_ohm = self.ohmic_resistance(delta_e)
        eta_ohm = i * R_ohm
        
        
        i_L = self.limiting_current(Q, T_celsius)
        eta_mt = self.mass_transfer_overpotential(i, i_L, T_celsius)
        
        
        if charge:
            V = E_nernst + eta_act_total + eta_ohm + eta_mt
        else:
            V = E_nernst - eta_act_total - eta_ohm - eta_mt
        
        
        V = max(V, 0.0)
        
        components = {
            "E_nernst": E_nernst,
            "eta_act": eta_act_total,
            "eta_ohm": eta_ohm,
            "eta_mt": eta_mt,
            "R_ohm": R_ohm,
            "i_L": i_L
        }
        
        return V, components
    
    def voltage_efficiency(
        self,
        i: float,
        SOC: float,
        T_celsius: float,
        delta_e: float,
        Q: float
    ) -> float:
        V_discharge, _ = self.cell_voltage(i, SOC, T_celsius, delta_e, Q, charge=False)
        V_charge, _ = self.cell_voltage(i, SOC, T_celsius, delta_e, Q, charge=True)
        
        if V_charge > 0:
            eta_V = V_discharge / V_charge
        else:
            eta_V = 0.0
        
        return eta_V
    
    def pumping_power(
        self,
        Q: float,
        delta_e: float
    ) -> float:
        
        Q_m3s = Q * 1e-6  
        delta_e_m = delta_e * 1e-3  
        
        
        
        rho = VRFBParams.Electrolyte.density
        mu = VRFBParams.Electrolyte.viscosity
        
        
        A_cross = VRFBParams.Flow.channel_width * VRFBParams.Flow.channel_depth * 1e-6
        v = Q_m3s / A_cross if A_cross > 0 else 1e-3
        
        
        d_h_m = VRFBParams.Flow.d_h * 1e-3
        Re = (rho * v * d_h_m) / mu
        
        
        if Re < 2300:
            f = 64 / Re
        else:
            
            f = 0.316 / (Re ** 0.25)
        
        
        L = VRFBParams.Flow.channel_length * 1e-3  
        delta_P = f * (L / d_h_m) * (rho * v**2 / 2)
        
        
        P_p = Q_m3s * delta_P
        
        return P_p
    
    def simulate_cycle(
        self,
        i: float,
        T_celsius: float,
        delta_e: float,
        Q: float,
        SOC_start: float = 0.5,
        add_noise: bool = True
    ) -> Dict[str, float]:
        
        V_discharge, comp_dis = self.cell_voltage(
            i, SOC_start, T_celsius, delta_e, Q, charge=False
        )
        V_charge, comp_ch = self.cell_voltage(
            i, SOC_start, T_celsius, delta_e, Q, charge=True
        )
        
        
        eta_V = self.voltage_efficiency(i, SOC_start, T_celsius, delta_e, Q)
        
        
        P_p = self.pumping_power(Q, delta_e)
        
        
        if add_noise:
            noise_factor = 1 + np.random.normal(0, NoiseParams.vrfb_efficiency_std_rel)
            eta_V_noisy = eta_V * noise_factor
            eta_V_noisy = np.clip(eta_V_noisy, 0.0, 1.0)
            
            V_noise = np.random.normal(0, NoiseParams.vrfb_voltage_std)
            V_discharge_noisy = V_discharge + V_noise
            V_charge_noisy = V_charge - V_noise  
        else:
            eta_V_noisy = eta_V
            V_discharge_noisy = V_discharge
            V_charge_noisy = V_charge
        
        result = {
            "current_density_mA_cm2": i * 1000,  
            "temperature_C": T_celsius,
            "electrode_thickness_mm": delta_e,
            "flow_rate_mL_s": Q,
            "SOC": SOC_start,
            "V_discharge_V": V_discharge_noisy,
            "V_charge_V": V_charge_noisy,
            "voltage_efficiency": eta_V_noisy,
            "pumping_power_W": P_p,
            "E_nernst_V": comp_dis["E_nernst"],
            "eta_act_discharge_V": comp_dis["eta_act"],
            "eta_act_charge_V": comp_ch["eta_act"],
            "eta_ohm_V": comp_dis["eta_ohm"],
            "eta_mt_V": comp_dis["eta_mt"],
            "R_ohm_ohm_cm2": comp_dis["R_ohm"],
            "i_L_A_cm2": comp_dis["i_L"],
            
            "V_discharge_clean_V": V_discharge,
            "V_charge_clean_V": V_charge,
            "voltage_efficiency_clean": eta_V
        }
        
        return result
    
    def generate_dataset(
        self,
        current_densities: List[float] = [50, 100, 150, 200, 250, 300],  
        electrode_thicknesses: List[float] = [3, 5, 7, 10],  
        flow_rates: List[float] = [10, 20, 30, 50],  
        SOCs: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
        T_celsius: float = 25.0,
        add_noise: bool = True
    ) -> pd.DataFrame:
        all_data = []
        cycle_id = 0
        
        print(f"Generating VRFB dataset...")
        print(f"  Current densities: {current_densities} mA/cm²")
        print(f"  Electrode thicknesses: {electrode_thicknesses} mm")
        print(f"  Flow rates: {flow_rates} mL/s")
        print(f"  SOC levels: {SOCs}")
        print(f"  Temperature: {T_celsius}°C")
        
        total_cycles = (len(current_densities) * len(electrode_thicknesses) * 
                       len(flow_rates) * len(SOCs))
        print(f"  Total cycles: {total_cycles}")
        
        for i_mA in current_densities:
            i_A = i_mA / 1000  
            for delta_e in electrode_thicknesses:
                for Q in flow_rates:
                    for SOC in SOCs:
                        result = self.simulate_cycle(
                            i_A, T_celsius, delta_e, Q, SOC, add_noise
                        )
                        result["cycle_id"] = cycle_id
                        all_data.append(result)
                        cycle_id += 1
                        
                        if cycle_id % 50 == 0:
                            print(f"  Generated {cycle_id} cycles...")
        
        
        dataset = pd.DataFrame(all_data)
        
        print(f"✓ Generated {len(dataset)} cycles")
        print(f"  Voltage efficiency range: {dataset['voltage_efficiency'].min():.3f} - {dataset['voltage_efficiency'].max():.3f}")
        print(f"  Discharge voltage range: {dataset['V_discharge_V'].min():.3f} - {dataset['V_discharge_V'].max():.3f} V")
        
        return dataset
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, bool]:
        bounds = VALIDATION_BOUNDS["VRFB"]
        
        checks = {
            "voltage_range": (
                df["V_discharge_V"].min() >= bounds["voltage"][0] and
                df["V_charge_V"].max() <= bounds["voltage"][1]
            ),
            "efficiency_range": (
                df["voltage_efficiency"].min() >= bounds["efficiency"][0] and
                df["voltage_efficiency"].max() <= bounds["efficiency"][1]
            ),
            "SOC_range": (
                df["SOC"].min() >= bounds["SOC"][0] and
                df["SOC"].max() <= bounds["SOC"][1]
            ),
            "current_below_limiting": (
                df["current_density_mA_cm2"] / 1000 < df["i_L_A_cm2"]
            ).all(),
            "positive_resistance": (df["R_ohm_ohm_cm2"] > 0).all(),
            "positive_nernst": (df["E_nernst_V"] > 0).all()
        }
        
        all_pass = all(checks.values())
        checks["all_checks_passed"] = all_pass
        
        return checks


def main():
    simulator = VRFBSimulator(random_seed=NoiseParams.random_seed)
    
    
    dataset = simulator.generate_dataset(
        current_densities=[50, 100, 150, 200, 250, 300],
        electrode_thicknesses=[3, 5, 7, 10],
        flow_rates=[10, 20, 30, 50],
        SOCs=[0.1, 0.3, 0.5, 0.7, 0.9],
        T_celsius=25.0,
        add_noise=True
    )
    
    
    validation = simulator.validate_data(dataset)
    print("\nValidation Results:")
    for check, passed in validation.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}: {passed}")
    
    
    output_path = "data/synthetic/vrfb_cycles.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset.to_csv(output_path, index=False)
    print(f"\n✓ Saved dataset to: {output_path}")
    
    
    metadata = {
        "n_cycles": int(len(dataset)),
        "current_densities_mA_cm2": sorted(dataset["current_density_mA_cm2"].unique().tolist()),
        "electrode_thicknesses_mm": sorted(dataset["electrode_thickness_mm"].unique().tolist()),
        "flow_rates_mL_s": sorted(dataset["flow_rate_mL_s"].unique().tolist()),
        "SOC_levels": sorted(dataset["SOC"].unique().tolist()),
        "temperature_C": float(dataset["temperature_C"].iloc[0]),
        "voltage_efficiency_range": [
            float(dataset["voltage_efficiency"].min()),
            float(dataset["voltage_efficiency"].max())
        ],
        "discharge_voltage_range": [
            float(dataset["V_discharge_V"].min()),
            float(dataset["V_discharge_V"].max())
        ],
        "validation": {k: bool(v) for k, v in validation.items()}
    }
    
    import json
    metadata_path = "data/synthetic/vrfb_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to: {metadata_path}")
    
    return dataset


if __name__ == "__main__":
    dataset = main()
