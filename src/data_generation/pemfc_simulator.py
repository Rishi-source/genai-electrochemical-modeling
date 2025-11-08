"""
PEMFC Polarization Curve Simulator
Generates synthetic polarization data based on governing equations (1-5) from paper.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.physics_constants import (
    F, R, T_STD,
    PEMFCParams,
    NoiseParams,
    celsius_to_kelvin,
    nernst_potential_pemfc,
    arrhenius_i0,
    ACTIVATION_ENERGY,
    VALIDATION_BOUNDS
)


class PEMFCSimulator:
    """
    Simulates PEMFC polarization behavior based on physics equations.
    
    Implements:
    - Equation (1): V = E_Nernst - η_act - η_Ω - η_mt
    - Equation (2): Butler-Volmer kinetics (Tafel approximation)
    - Equation (4): Nernst potential
    - Equation (5): Ohmic resistance
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize PEMFC simulator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Default parameters (will be varied for different conditions)
        self.i0_cathode = PEMFCParams.i0_cathode_typical
        self.alpha = PEMFCParams.alpha_cathode
        self.R_ohm = PEMFCParams.R_ohm_typical
        self.i_L = PEMFCParams.i_L_typical
    
    def nernst_potential(
        self,
        T_celsius: float,
        p_H2: float = 1.0,
        p_O2: float = 0.21,
        a_H2O: float = 1.0
    ) -> float:
        """
        Calculate Nernst potential (Equation 4).
        
        Args:
            T_celsius: Temperature [°C]
            p_H2: Hydrogen partial pressure [atm]
            p_O2: Oxygen partial pressure [atm]
            a_H2O: Water activity
        
        Returns:
            E_Nernst: Nernst potential [V]
        """
        T_kelvin = celsius_to_kelvin(T_celsius)
        return nernst_potential_pemfc(T_kelvin, p_H2, p_O2, a_H2O)
    
    def activation_overpotential(
        self,
        i: np.ndarray,
        i0: float,
        alpha: float,
        T_celsius: float
    ) -> np.ndarray:
        """
        Calculate activation overpotential using Tafel approximation.
        
        η_act = (RT/αF) * ln(i/i0)
        
        Args:
            i: Current density [A/cm²]
            i0: Exchange current density [A/cm²]
            alpha: Transfer coefficient
            T_celsius: Temperature [°C]
        
        Returns:
            η_act: Activation overpotential [V]
        """
        T_kelvin = celsius_to_kelvin(T_celsius)
        RT_over_alphaF = (R * T_kelvin) / (alpha * F)
        
        # Avoid log(0) for very small currents
        i_safe = np.maximum(i, 1e-8)
        eta_act = RT_over_alphaF * np.log(i_safe / i0)
        
        return eta_act
    
    def ohmic_overpotential(
        self,
        i: np.ndarray,
        R_ohm: float
    ) -> np.ndarray:
        """
        Calculate ohmic overpotential (Equation 5).
        
        η_Ω = i * R_Ω
        
        Args:
            i: Current density [A/cm²]
            R_ohm: Ohmic resistance [Ω·cm²]
        
        Returns:
            η_Ω: Ohmic overpotential [V]
        """
        return i * R_ohm
    
    def mass_transfer_overpotential(
        self,
        i: np.ndarray,
        i_L: float,
        T_celsius: float,
        n: int = 4  # 4 electrons for O2 reduction
    ) -> np.ndarray:
        """
        Calculate mass transfer overpotential.
        
        η_mt = -(RT/nF) * ln(1 - i/i_L)
        
        Args:
            i: Current density [A/cm²]
            i_L: Limiting current density [A/cm²]
            T_celsius: Temperature [°C]
            n: Number of electrons
        
        Returns:
            η_mt: Mass transfer overpotential [V]
        """
        T_kelvin = celsius_to_kelvin(T_celsius)
        RT_over_nF = (R * T_kelvin) / (n * F)
        
        # Prevent division issues and log of negative numbers
        ratio = np.minimum(i / i_L, 0.99)
        eta_mt = -RT_over_nF * np.log(1 - ratio)
        
        return eta_mt
    
    def cell_voltage(
        self,
        i: np.ndarray,
        T_celsius: float,
        p_H2: float,
        p_O2: float,
        i0: float,
        alpha: float,
        R_ohm: float,
        i_L: float
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Calculate cell voltage and loss breakdown (Equation 1).
        
        V = E_Nernst - η_act - η_Ω - η_mt
        
        Args:
            i: Current density array [A/cm²]
            T_celsius: Temperature [°C]
            p_H2: H2 partial pressure [atm]
            p_O2: O2 partial pressure [atm]
            i0: Exchange current density [A/cm²]
            alpha: Transfer coefficient
            R_ohm: Ohmic resistance [Ω·cm²]
            i_L: Limiting current density [A/cm²]
        
        Returns:
            V: Cell voltage [V]
            losses: Dictionary of loss components
        """
        # Calculate components
        E_nernst = self.nernst_potential(T_celsius, p_H2, p_O2)
        eta_act = self.activation_overpotential(i, i0, alpha, T_celsius)
        eta_ohm = self.ohmic_overpotential(i, R_ohm)
        eta_mt = self.mass_transfer_overpotential(i, i_L, T_celsius)
        
        # Cell voltage
        V = E_nernst - eta_act - eta_ohm - eta_mt
        
        # Ensure physical bounds
        V = np.maximum(V, 0.0)
        
        losses = {
            "E_nernst": E_nernst * np.ones_like(i),
            "eta_act": eta_act,
            "eta_ohm": eta_ohm,
            "eta_mt": eta_mt
        }
        
        return V, losses
    
    def generate_polarization_curve(
        self,
        T_celsius: float,
        stoich_anode: float,
        stoich_cathode: float,
        i_min: float = 0.1,
        i_max: float = 1.5,
        n_points: int = 20,
        add_noise: bool = True
    ) -> pd.DataFrame:
        """
        Generate a single polarization curve for given conditions.
        
        Args:
            T_celsius: Temperature [°C]
            stoich_anode: Anode stoichiometry
            stoich_cathode: Cathode stoichiometry
            i_min: Minimum current density [A/cm²]
            i_max: Maximum current density [A/cm²]
            n_points: Number of data points
            add_noise: Whether to add measurement noise
        
        Returns:
            DataFrame with polarization data
        """
        # Generate current density range
        i = np.linspace(i_min, i_max, n_points)
        
        # Temperature-dependent exchange current density (Arrhenius)
        T_kelvin = celsius_to_kelvin(T_celsius)
        i0 = arrhenius_i0(
            PEMFCParams.i0_cathode_typical,
            ACTIVATION_ENERGY["PEMFC_cathode"],
            T_kelvin
        )
        
        # Parameters (could be functions of stoichiometry, but simplified here)
        alpha = PEMFCParams.alpha_cathode
        R_ohm = PEMFCParams.R_ohm_typical
        
        # Limiting current depends on flow rate (related to stoichiometry)
        # Higher stoichiometry → higher flow → higher i_L
        i_L = PEMFCParams.i_L_typical * (stoich_cathode / 2.0)
        
        # Partial pressures (assuming air cathode, pure H2 anode)
        p_H2 = 1.0
        p_O2 = 0.21 * (stoich_cathode / 2.0)  # Simplified: more flow, more O2 available
        
        # Calculate voltage and losses
        V, losses = self.cell_voltage(
            i, T_celsius, p_H2, p_O2, i0, alpha, R_ohm, i_L
        )
        
        # Add noise if requested
        if add_noise:
            noise = np.random.normal(0, NoiseParams.pemfc_voltage_std, len(V))
            V_noisy = V + noise
            V_noisy = np.maximum(V_noisy, 0.0)  # Ensure non-negative
        else:
            V_noisy = V
        
        # Create DataFrame
        df = pd.DataFrame({
            "current_density_A_cm2": i,
            "voltage_V": V_noisy,
            "voltage_clean_V": V,
            "temperature_C": T_celsius,
            "stoich_anode": stoich_anode,
            "stoich_cathode": stoich_cathode,
            "E_nernst_V": losses["E_nernst"],
            "eta_act_V": losses["eta_act"],
            "eta_ohm_V": losses["eta_ohm"],
            "eta_mt_V": losses["eta_mt"],
            "i0_A_cm2": i0,
            "alpha": alpha,
            "R_ohm_ohm_cm2": R_ohm,
            "i_L_A_cm2": i_L,
            "p_H2_atm": p_H2,
            "p_O2_atm": p_O2
        })
        
        return df
    
    def generate_dataset(
        self,
        temperatures: List[float] = [50, 60, 70, 80, 90],
        stoich_anodes: List[float] = [1.2, 1.5, 2.0],
        stoich_cathodes: List[float] = [2.0, 2.5, 3.0],
        i_min: float = 0.1,
        i_max: float = 1.5,
        n_points: int = 20,
        add_noise: bool = True
    ) -> pd.DataFrame:
        """
        Generate complete PEMFC dataset with multiple operating conditions.
        
        Args:
            temperatures: List of temperatures [°C]
            stoich_anodes: List of anode stoichiometries
            stoich_cathodes: List of cathode stoichiometries
            i_min: Minimum current density [A/cm²]
            i_max: Maximum current density [A/cm²]
            n_points: Points per curve
            add_noise: Add measurement noise
        
        Returns:
            DataFrame with all polarization curves
        """
        all_data = []
        curve_id = 0
        
        print(f"Generating PEMFC dataset...")
        print(f"  Temperatures: {temperatures}")
        print(f"  Anode stoichiometries: {stoich_anodes}")
        print(f"  Cathode stoichiometries: {stoich_cathodes}")
        print(f"  Total curves: {len(temperatures) * len(stoich_anodes) * len(stoich_cathodes)}")
        
        for T in temperatures:
            for stoich_a in stoich_anodes:
                for stoich_c in stoich_cathodes:
                    df = self.generate_polarization_curve(
                        T, stoich_a, stoich_c,
                        i_min, i_max, n_points, add_noise
                    )
                    df["curve_id"] = curve_id
                    all_data.append(df)
                    curve_id += 1
                    
                    if curve_id % 5 == 0:
                        print(f"  Generated {curve_id} curves...")
        
        # Combine all curves
        dataset = pd.concat(all_data, ignore_index=True)
        
        print(f"✓ Generated {len(dataset)} data points across {curve_id} curves")
        print(f"  Voltage range: {dataset['voltage_V'].min():.3f} - {dataset['voltage_V'].max():.3f} V")
        print(f"  Current range: {dataset['current_density_A_cm2'].min():.3f} - {dataset['current_density_A_cm2'].max():.3f} A/cm²")
        
        return dataset
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate generated data against physical bounds.
        
        Args:
            df: DataFrame with PEMFC data
        
        Returns:
            Dictionary of validation results
        """
        bounds = VALIDATION_BOUNDS["PEMFC"]
        
        checks = {
            "voltage_range": (
                df["voltage_V"].min() >= bounds["voltage"][0] and
                df["voltage_V"].max() <= bounds["voltage"][1]
            ),
            "current_range": (
                df["current_density_A_cm2"].min() >= bounds["current_density"][0] and
                df["current_density_A_cm2"].max() <= bounds["current_density"][1]
            ),
            "activation_positive": (df["eta_act_V"] >= 0).all(),
            "ohmic_positive": (df["eta_ohm_V"] >= 0).all(),
            "mass_transfer_positive": (df["eta_mt_V"] >= 0).all(),
            "current_below_limiting": (
                df["current_density_A_cm2"] < df["i_L_A_cm2"]
            ).all()
        }
        
        all_pass = all(checks.values())
        checks["all_checks_passed"] = all_pass
        
        return checks


def main():
    """
    Main function to generate PEMFC dataset.
    """
    # Initialize simulator
    simulator = PEMFCSimulator(random_seed=NoiseParams.random_seed)
    
    # Generate dataset
    dataset = simulator.generate_dataset(
        temperatures=[50, 60, 70, 80, 90],
        stoich_anodes=[1.2, 1.5, 2.0],
        stoich_cathodes=[2.0, 2.5, 3.0],
        i_min=0.1,
        i_max=1.5,
        n_points=20,
        add_noise=True
    )
    
    # Validate
    validation = simulator.validate_data(dataset)
    print("\nValidation Results:")
    for check, passed in validation.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}: {passed}")
    
    # Save to CSV
    output_path = "data/synthetic/pemfc_polarization.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset.to_csv(output_path, index=False)
    print(f"\n✓ Saved dataset to: {output_path}")
    
    # Save metadata
    metadata = {
        "n_curves": int(dataset["curve_id"].nunique()),
        "n_points": int(len(dataset)),
        "temperatures": sorted(dataset["temperature_C"].unique().tolist()),
        "stoich_anodes": sorted(dataset["stoich_anode"].unique().tolist()),
        "stoich_cathodes": sorted(dataset["stoich_cathode"].unique().tolist()),
        "voltage_range": [float(dataset["voltage_V"].min()), float(dataset["voltage_V"].max())],
        "current_range": [float(dataset["current_density_A_cm2"].min()), float(dataset["current_density_A_cm2"].max())],
        "validation": {k: bool(v) for k, v in validation.items()}
    }
    
    import json
    metadata_path = "data/synthetic/pemfc_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to: {metadata_path}")
    
    return dataset


if __name__ == "__main__":
    dataset = main()
