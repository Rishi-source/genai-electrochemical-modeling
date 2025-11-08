import os
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
from dataclasses import dataclass

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.physics_constants import R, F


@dataclass
class PEMFCFitResult:
    i0_A_cm2: float  
    alpha: float  
    R_ohm_cm2: float  
    i_L_A_cm2: float  
    
    rmse_V: float  
    mae_V: float  
    r_squared: float  
    
    fitted_voltage: np.ndarray  
    residuals: np.ndarray  
    
    convergence_info: Dict  


class PEMFCFitter:
    
    def __init__(
        self,
        temperature_C: float = 80.0,
        pressure_H2_atm: float = 1.0,
        pressure_O2_atm: float = 0.21,
        verbose: bool = True
    ):
        self.T_C = temperature_C
        self.T_K = temperature_C + 273.15
        self.p_H2 = pressure_H2_atm
        self.p_O2 = pressure_O2_atm
        self.verbose = verbose
        
        
        self.E_nernst = self.compute_nernst_potential()
        
        if self.verbose:
            print(f"✓ PEMFC Fitter initialized")
            print(f"  Temperature: {self.T_C}°C")
            print(f"  Nernst potential: {self.E_nernst:.3f} V")
    
    def compute_nernst_potential(self) -> float:
        E_std = 1.229 - 0.00085 * (self.T_K - 298.15)
        
        
        a_H2O = 1.0
        
        
        RT_2F = (R * self.T_K) / (2 * F)
        E_nernst = E_std + RT_2F * np.log(self.p_H2 * np.sqrt(self.p_O2) / a_H2O)
        
        return E_nernst
    
    def compute_activation_loss(self, i: np.ndarray, i0: float, alpha: float) -> np.ndarray:
        RT_alphaF = (R * self.T_K) / (alpha * F)
        
        
        i_safe = np.maximum(i, 1e-10)
        eta_act = RT_alphaF * np.log(i_safe / i0)
        
        return eta_act
    
    def compute_ohmic_loss(self, i: np.ndarray, R_ohm: float) -> np.ndarray:
        return i * R_ohm
    
    def compute_mass_transfer_loss(self, i: np.ndarray, i_L: float) -> np.ndarray:
        RT_nF = (R * self.T_K) / (2 * F)  
        
        
        ratio = np.minimum(i / i_L, 0.999)
        eta_mt = -RT_nF * np.log(1 - ratio)
        
        return eta_mt
    
    def voltage_model(self, params: np.ndarray, i: np.ndarray) -> np.ndarray:
        i0, alpha, R_ohm, i_L = params
        eta_act = self.compute_activation_loss(i, i0, alpha)
        eta_ohm = self.compute_ohmic_loss(i, R_ohm)
        eta_mt = self.compute_mass_transfer_loss(i, i_L)
        
        
        V = self.E_nernst - eta_act - eta_ohm - eta_mt
        
        return V
    
    def residuals(self, params: np.ndarray, i: np.ndarray, V_meas: np.ndarray) -> np.ndarray:
        V_pred = self.voltage_model(params, i)
        return V_meas - V_pred
    
    def fit(
        self,
        current_density: np.ndarray,
        voltage: np.ndarray,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        initial_guess: Optional[Dict[str, float]] = None
    ) -> PEMFCFitResult:
        if bounds is None:
            bounds = {
                'i0': (1e-10, 1e-4),  
                'alpha': (0.1, 1.0),  
                'R_ohm': (0.01, 1.0),  
                'i_L': (max(current_density) * 1.1, 10.0)  
            }
        
        
        if initial_guess is None:
            initial_guess = {
                'i0': np.sqrt(bounds['i0'][0] * bounds['i0'][1]),  
                'alpha': 0.5,
                'R_ohm': 0.15,
                'i_L': max(current_density) * 1.5
            }
        
        
        x0 = np.array([
            initial_guess['i0'],
            initial_guess['alpha'],
            initial_guess['R_ohm'],
            initial_guess['i_L']
        ])
        
        lower_bounds = np.array([
            bounds['i0'][0],
            bounds['alpha'][0],
            bounds['R_ohm'][0],
            bounds['i_L'][0]
        ])
        
        upper_bounds = np.array([
            bounds['i0'][1],
            bounds['alpha'][1],
            bounds['R_ohm'][1],
            bounds['i_L'][1]
        ])
        
        if self.verbose:
            print(f"\nFitting PEMFC parameters...")
            print(f"  Data points: {len(current_density)}")
            print(f"  Current range: {current_density.min():.3f} - {current_density.max():.3f} A/cm²")
            print(f"  Voltage range: {voltage.min():.3f} - {voltage.max():.3f} V")
        
        
        result = least_squares(
            self.residuals,
            x0,
            bounds=(lower_bounds, upper_bounds),
            args=(current_density, voltage),
            method='trf',  
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=1000,
            verbose=0
        )
        
        
        i0_fit, alpha_fit, R_ohm_fit, i_L_fit = result.x
        
        
        V_fit = self.voltage_model(result.x, current_density)
        residuals = voltage - V_fit
        
        
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((voltage - voltage.mean())**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        if self.verbose:
            print(f"\n✓ Fitting complete")
            print(f"  i₀: {i0_fit:.3e} A/cm²")
            print(f"  α: {alpha_fit:.3f}")
            print(f"  R_Ω: {R_ohm_fit:.3f} Ω·cm²")
            print(f"  i_L: {i_L_fit:.3f} A/cm²")
            print(f"\nDiagnostics:")
            print(f"  RMSE: {rmse*1000:.2f} mV")
            print(f"  MAE: {mae*1000:.2f} mV")
            print(f"  R²: {r_squared:.4f}")
            print(f"  Iterations: {result.nfev}")
            print(f"  Success: {result.success}")
        
        return PEMFCFitResult(
            i0_A_cm2=i0_fit,
            alpha=alpha_fit,
            R_ohm_cm2=R_ohm_fit,
            i_L_A_cm2=i_L_fit,
            rmse_V=rmse,
            mae_V=mae,
            r_squared=r_squared,
            fitted_voltage=V_fit,
            residuals=residuals,
            convergence_info={
                'success': result.success,
                'iterations': result.nfev,
                'optimality': result.optimality,
                'message': result.message
            }
        )
    
    def plot_fit(
        self,
        current_density: np.ndarray,
        voltage_measured: np.ndarray,
        fit_result: PEMFCFitResult,
        save_path: Optional[str] = None
    ):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        
        ax = axes[0, 0]
        ax.scatter(current_density, voltage_measured, 
                  label='Measured', alpha=0.6, s=50, color='blue')
        ax.plot(current_density, fit_result.fitted_voltage, 
               label='Fitted', linewidth=2, color='red')
        ax.set_xlabel('Current Density [A/cm²]')
        ax.set_ylabel('Voltage [V]')
        ax.set_title(f'PEMFC Polarization Curve (R²={fit_result.r_squared:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        
        ax = axes[0, 1]
        
        
        params = np.array([
            fit_result.i0_A_cm2,
            fit_result.alpha,
            fit_result.R_ohm_cm2,
            fit_result.i_L_A_cm2
        ])
        
        eta_act = self.compute_activation_loss(current_density, fit_result.i0_A_cm2, fit_result.alpha)
        eta_ohm = self.compute_ohmic_loss(current_density, fit_result.R_ohm_cm2)
        eta_mt = self.compute_mass_transfer_loss(current_density, fit_result.i_L_A_cm2)
        
        ax.plot(current_density, np.ones_like(current_density) * self.E_nernst, 
               '--', label='E_Nernst', linewidth=2)
        ax.plot(current_density, self.E_nernst - eta_act, 
               label='After activation', linewidth=1.5)
        ax.plot(current_density, self.E_nernst - eta_act - eta_ohm, 
               label='After ohmic', linewidth=1.5)
        ax.plot(current_density, fit_result.fitted_voltage, 
               label='After mass transfer', linewidth=2, color='red')
        ax.set_xlabel('Current Density [A/cm²]')
        ax.set_ylabel('Voltage [V]')
        ax.set_title('Loss Decomposition')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        
        ax = axes[1, 0]
        
        
        kinetic_mask = (current_density < 0.5 * fit_result.i_L_A_cm2) & (current_density > 0.01)
        if np.any(kinetic_mask):
            i_kinetic = current_density[kinetic_mask]
            eta_kinetic = eta_act[kinetic_mask]
            
            ax.semilogy(eta_kinetic, i_kinetic, 'o', label='Data', alpha=0.6)
            
            
            tafel_slope = (R * self.T_K) / (fit_result.alpha * F)
            ax.text(0.05, 0.95, f'Tafel slope: {tafel_slope*1000:.1f} mV/decade', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Activation Overpotential [V]')
        ax.set_ylabel('Current Density [A/cm²]')
        ax.set_title('Tafel Plot (Kinetic Region)')
        ax.grid(True, alpha=0.3, which='both')
        
        
        ax = axes[1, 1]
        ax.scatter(current_density, fit_result.residuals * 1000, alpha=0.6, s=30)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax.set_xlabel('Current Density [A/cm²]')
        ax.set_ylabel('Residuals [mV]')
        ax.set_title(f'Residual Analysis (RMSE={fit_result.rmse_V*1000:.2f} mV)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved figure to {save_path}")
        
        plt.show()


def main():
    print("Testing PEMFC Parameter Fitter...")
    
    
    data_path = "data/synthetic/pemfc_polarization.csv"
    
    if not os.path.exists(data_path):
        print(f"✗ Data file not found: {data_path}")
        print("  Run src/data_generation/pemfc_simulator.py first")
        return
    
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} data points")
    
    
    test_curve = df[
        (df['temperature_C'] == 80) &
        (df['stoich_anode'] == 1.5) &
        (df['stoich_cathode'] == 2.5)
    ].sort_values('current_density_A_cm2')
    
    print(f"✓ Selected test curve: {len(test_curve)} points")
    
    
    fitter = PEMFCFitter(
        temperature_C=80.0,
        pressure_H2_atm=1.0,
        pressure_O2_atm=0.21,
        verbose=True
    )
    
    
    fit_result = fitter.fit(
        current_density=test_curve['current_density_A_cm2'].values,
        voltage=test_curve['voltage_V'].values
    )
    
    
    fitter.plot_fit(
        current_density=test_curve['current_density_A_cm2'].values,
        voltage_measured=test_curve['voltage_V'].values,
        fit_result=fit_result,
        save_path="results/figures/pemfc_fit_test.png"
    )
    
    print("\n✓ PEMFC Fitter test complete")


if __name__ == "__main__":
    main()
