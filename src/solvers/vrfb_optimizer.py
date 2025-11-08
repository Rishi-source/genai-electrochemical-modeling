import os
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
from dataclasses import dataclass

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.physics_constants import R, F, VRFBParams


@dataclass
class VRFBOptimizationResult:
    optimal_thickness_mm: float  
    optimal_flow_rate_mL_s: float  
    
    voltage_efficiency: float  
    pumping_power_W: float  
    normalized_pumping_power: float  
    
    design_points: np.ndarray  
    pareto_front: np.ndarray  
    
    constraint_violations: Dict  


class VRFBOptimizer:
    
    def __init__(
        self,
        current_density_mA_cm2: float = 150.0,
        temperature_C: float = 25.0,
        SOC: float = 0.5,
        electrolyte_conc_M: float = 2.0,
        verbose: bool = True
    ):
        self.i = current_density_mA_cm2 / 1000.0  
        self.T_C = temperature_C
        self.T_K = temperature_C + 273.15
        self.SOC = SOC
        self.c = electrolyte_conc_M
        self.verbose = verbose
        
        
        self.E_nernst = self.compute_nernst_potential()
        
        
        self.rho = VRFBParams.Electrolyte.density  
        self.mu = VRFBParams.Electrolyte.viscosity  
        self.D = VRFBParams.Electrolyte.diffusivity_V2  
        
        
        self.porosity = VRFBParams.Electrode.porosity
        self.conductivity = VRFBParams.Electrode.conductivity  
        
        
        self.delta_m = VRFBParams.Membrane.thickness * 1e-6  
        self.kappa_m = VRFBParams.Membrane.conductivity  
        
        
        self.delta_p_max = VRFBParams.Operating.delta_p_max  
        
        if self.verbose:
            print(f"✓ VRFB Optimizer initialized")
            print(f"  Current density: {current_density_mA_cm2} mA/cm²")
            print(f"  Temperature: {self.T_C}°C")
            print(f"  SOC: {self.SOC}")
            print(f"  Nernst potential: {self.E_nernst:.3f} V")
    
    def compute_nernst_potential(self) -> float:
        E_std = VRFBParams.E0_standard + VRFBParams.dE0_dT * (self.T_K - 298.15)
        if self.SOC <= 0.01:
            self.SOC = 0.01  
        if self.SOC >= 0.99:
            self.SOC = 0.99
        
        concentration_ratio = (self.SOC * self.SOC) / ((1 - self.SOC) * (1 - self.SOC))
        
        RT_F = (R * self.T_K) / F
        E_nernst = E_std + RT_F * np.log(concentration_ratio)
        
        return E_nernst
    
    def compute_mass_transfer_coefficient(
        self,
        Q_mL_s: float,
        d_h_mm: float = 3.75
    ) -> float:        
        Q = Q_mL_s * 1e-6  
        d_h = d_h_mm * 1e-3  
        
        
        A_electrode = 0.01  
        A_cross = 5e-6  
        
        
        u = Q / A_cross  
        
        
        Re = (self.rho * u * d_h) / self.mu
        
        
        Sc = self.mu / (self.rho * self.D)
        
        
        a = VRFBParams.Flow.sherwood_a
        b = VRFBParams.Flow.sherwood_b
        Sh = a * (Re ** b) * (Sc ** (1/3))
        
        
        k_m = Sh * self.D / d_h
        
        return k_m
    
    def compute_limiting_current(
        self,
        Q_mL_s: float,
        d_h_mm: float = 3.75
    ) -> float:
        k_m = self.compute_mass_transfer_coefficient(Q_mL_s, d_h_mm)
        
        
        c_mol_m3 = self.c * 1000  
        
        
        i_L = F * k_m * c_mol_m3  
        i_L_A_cm2 = i_L / 10000  
        
        return i_L_A_cm2
    
    def compute_activation_loss(
        self,
        i0: float = 0.005,
        alpha: float = 0.5
    ) -> Tuple[float, float]:
        RT_alphaF = (R * self.T_K) / (alpha * F)
        
        
        i_safe = max(self.i, 1e-10)
        eta_act = RT_alphaF * np.log(i_safe / i0)
        
        return eta_act, eta_act  
    
    def compute_ohmic_resistance(
        self,
        delta_e_mm: float
    ) -> float:
        delta_e = delta_e_mm * 1e-3  
        
        
        R_felt = delta_e / self.conductivity  
        
        
        R_mem = self.delta_m / self.kappa_m  
        
        
        R_ohm = R_felt + R_mem  
        R_ohm_cm2 = R_ohm * 10000  
        
        return R_ohm_cm2
    
    def compute_mass_transfer_loss(
        self,
        Q_mL_s: float
    ) -> Tuple[float, float]:
        i_L = self.compute_limiting_current(Q_mL_s)
        
        RT_F = (R * self.T_K) / F
        
        
        ratio = min(self.i / i_L, 0.999)
        eta_mt = -RT_F * np.log(1 - ratio)
        
        return eta_mt, eta_mt  
    
    def compute_voltage_efficiency(
        self,
        delta_e_mm: float,
        Q_mL_s: float,
        i0: float = 0.005,
        alpha: float = 0.5
    ) -> float:
        eta_act_ch, eta_act_dis = self.compute_activation_loss(i0, alpha)
        R_ohm = self.compute_ohmic_resistance(delta_e_mm)
        eta_ohm = self.i * R_ohm
        eta_mt_ch, eta_mt_dis = self.compute_mass_transfer_loss(Q_mL_s)
        
        
        eta_ch = eta_act_ch + eta_ohm + eta_mt_ch
        eta_dis = eta_act_dis + eta_ohm + eta_mt_dis
        
        
        V_dis = self.E_nernst - eta_dis
        V_ch = self.E_nernst + eta_ch
        eta_V = V_dis / V_ch
        
        return eta_V
    
    def compute_pumping_power(
        self,
        Q_mL_s: float,
        delta_e_mm: float
    ) -> float:
        Q = Q_mL_s * 1e-6  
        delta_e = delta_e_mm * 1e-3  
        
        
        A_cross = 5e-6  
        
        
        d_h = 3.75e-3  
        
        
        u = Q / A_cross
        
        
        Re = (self.rho * u * d_h) / self.mu
        
        
        if Re < 2300:
            f = 64 / Re  
        else:
            f = 0.316 / (Re ** 0.25)  
        
        
        L = delta_e  
        delta_p = f * (L / d_h) * (0.5 * self.rho * u**2)
        
        
        P_p = Q * delta_p
        
        return P_p
    
    def objective_function(
        self,
        design_vars: np.ndarray,
        beta: float = 0.5
    ) -> float:
        delta_e_mm, Q_mL_s = design_vars
        
        
        eta_V = self.compute_voltage_efficiency(delta_e_mm, Q_mL_s)
        P_p = self.compute_pumping_power(Q_mL_s, delta_e_mm)
        
        
        A_electrode = 0.01  
        P_ref = self.i * A_electrode * 10000 * self.E_nernst  
        
        
        P_p_norm = P_p / P_ref
        
        
        obj = -eta_V + beta * P_p_norm
        
        return obj
    
    def check_constraints(
        self,
        delta_e_mm: float,
        Q_mL_s: float
    ) -> Dict[str, bool]:        
        i_L = self.compute_limiting_current(Q_mL_s)
        current_ok = self.i < i_L
        
        
        P_p = self.compute_pumping_power(Q_mL_s, delta_e_mm)
        Q = Q_mL_s * 1e-6
        delta_p = P_p / Q if Q > 0 else 0
        pressure_ok = delta_p < self.delta_p_max
        
        
        thickness_ok = (VRFBParams.Electrode.thickness_min <= delta_e_mm <= 
                       VRFBParams.Electrode.thickness_max)
        
        
        flow_ok = (VRFBParams.Flow.Q_min <= Q_mL_s <= VRFBParams.Flow.Q_max)
        
        return {
            'current_below_limiting': current_ok,
            'pressure_within_limit': pressure_ok,
            'thickness_in_range': thickness_ok,
            'flow_in_range': flow_ok,
            'all_satisfied': all([current_ok, pressure_ok, thickness_ok, flow_ok])
        }
    
    def optimize(
        self,
        beta: float = 0.5,
        method: str = 'differential_evolution'
    ) -> VRFBOptimizationResult:
        bounds = [
            (VRFBParams.Electrode.thickness_min, VRFBParams.Electrode.thickness_max),  
            (VRFBParams.Flow.Q_min, VRFBParams.Flow.Q_max)  
        ]
        
        if self.verbose:
            print(f"\nOptimizing VRFB design...")
            print(f"  Trade-off parameter β: {beta}")
            print(f"  Method: {method}")
        
        
        if method == 'differential_evolution':
            result = differential_evolution(
                lambda x: self.objective_function(x, beta),
                bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                atol=1e-6,
                tol=1e-6
            )
        else:
            x0 = [5.0, 30.0]  
            result = minimize(
                lambda x: self.objective_function(x, beta),
                x0,
                bounds=bounds,
                method='L-BFGS-B'
            )
        
        
        delta_e_opt, Q_opt = result.x
        eta_V_opt = self.compute_voltage_efficiency(delta_e_opt, Q_opt)
        P_p_opt = self.compute_pumping_power(Q_opt, delta_e_opt)
        
        
        A_electrode = 0.01
        P_ref = self.i * A_electrode * 10000 * self.E_nernst
        P_p_norm = P_p_opt / P_ref
        
        
        constraints = self.check_constraints(delta_e_opt, Q_opt)
        
        if self.verbose:
            print(f"\n✓ Optimization complete")
            print(f"  Optimal thickness: {delta_e_opt:.2f} mm")
            print(f"  Optimal flow rate: {Q_opt:.2f} mL/s")
            print(f"\nPerformance:")
            print(f"  Voltage efficiency: {eta_V_opt*100:.2f}%")
            print(f"  Pumping power: {P_p_opt*1000:.2f} mW")
            print(f"  Normalized P_p: {P_p_norm*100:.4f}%")
            print(f"\nConstraints:")
            for key, val in constraints.items():
                print(f"  {key}: {'✓' if val else '✗'}")
        
        return VRFBOptimizationResult(
            optimal_thickness_mm=delta_e_opt,
            optimal_flow_rate_mL_s=Q_opt,
            voltage_efficiency=eta_V_opt,
            pumping_power_W=P_p_opt,
            normalized_pumping_power=P_p_norm,
            design_points=np.array([delta_e_opt, Q_opt]),
            pareto_front=np.array([[eta_V_opt, P_p_norm]]),
            constraint_violations=constraints
        )
    
    def generate_pareto_front(
        self,
        n_points: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        beta_values = np.linspace(0, 1, n_points)
        pareto_points = []
        design_points = []
        
        if self.verbose:
            print(f"\nGenerating Pareto front with {n_points} points...")
        
        for i, beta in enumerate(beta_values):
            result = self.optimize(beta=beta, method='differential_evolution')
            pareto_points.append([
                result.voltage_efficiency,
                result.normalized_pumping_power
            ])
            design_points.append([
                result.optimal_thickness_mm,
                result.optimal_flow_rate_mL_s
            ])
            
            if self.verbose and (i+1) % 5 == 0:
                print(f"  Generated {i+1}/{n_points} points...")
        
        if self.verbose:
            print(f"✓ Pareto front complete")
        
        return np.array(design_points), np.array(pareto_points)


    def plot_pareto_front(
        self,
        design_points: np.ndarray,
        pareto_points: np.ndarray,
        save_path: Optional[str] = None
    ):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        
        ax = axes[0]
        scatter = ax.scatter(
            pareto_points[:, 1] * 100,  
            pareto_points[:, 0] * 100,  
            c=design_points[:, 0],      
            s=100,
            cmap='viridis',
            edgecolors='black',
            linewidth=1.5,
            alpha=0.8
        )
        
        ax.set_xlabel('Normalized Pumping Power [%]', fontsize=12)
        ax.set_ylabel('Voltage Efficiency [%]', fontsize=12)
        ax.set_title('VRFB Pareto Front', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Electrode Thickness [mm]', fontsize=10)
        
        
        best_eff_idx = np.argmax(pareto_points[:, 0])
        best_power_idx = np.argmin(pareto_points[:, 1])
        
        ax.scatter(
            pareto_points[best_eff_idx, 1] * 100,
            pareto_points[best_eff_idx, 0] * 100,
            marker='*',
            s=400,
            c='red',
            edgecolors='black',
            linewidth=2,
            label='Max Efficiency',
            zorder=5
        )
        
        ax.scatter(
            pareto_points[best_power_idx, 1] * 100,
            pareto_points[best_power_idx, 0] * 100,
            marker='s',
            s=200,
            c='blue',
            edgecolors='black',
            linewidth=2,
            label='Min Pumping',
            zorder=5
        )
        
        ax.legend(fontsize=10)
        
        
        ax = axes[1]
        scatter = ax.scatter(
            design_points[:, 1],  
            design_points[:, 0],  
            c=pareto_points[:, 0] * 100,  
            s=100,
            cmap='plasma',
            edgecolors='black',
            linewidth=1.5,
            alpha=0.8
        )
        
        ax.set_xlabel('Flow Rate [mL/s]', fontsize=12)
        ax.set_ylabel('Electrode Thickness [mm]', fontsize=12)
        ax.set_title('Design Space', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Voltage Efficiency [%]', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"✓ Saved Pareto front to {save_path}")
        
        plt.show()
    
    def sensitivity_analysis(
        self,
        delta_e_mm: float,
        Q_mL_s: float,
        delta_e_range: Optional[Tuple[float, float]] = None,
        Q_range: Optional[Tuple[float, float]] = None,
        n_points: int = 20
    ) -> Dict[str, np.ndarray]:
        if delta_e_range is None:
            delta_e_range = (delta_e_mm * 0.7, delta_e_mm * 1.3)
        if Q_range is None:
            Q_range = (Q_mL_s * 0.7, Q_mL_s * 1.3)
        
        
        delta_e_vals = np.linspace(delta_e_range[0], delta_e_range[1], n_points)
        eta_V_vs_delta_e = np.array([
            self.compute_voltage_efficiency(d, Q_mL_s) for d in delta_e_vals
        ])
        P_p_vs_delta_e = np.array([
            self.compute_pumping_power(Q_mL_s, d) for d in delta_e_vals
        ])
        
        
        Q_vals = np.linspace(Q_range[0], Q_range[1], n_points)
        eta_V_vs_Q = np.array([
            self.compute_voltage_efficiency(delta_e_mm, q) for q in Q_vals
        ])
        P_p_vs_Q = np.array([
            self.compute_pumping_power(q, delta_e_mm) for q in Q_vals
        ])
        
        
        d_eta_d_delta_e = np.gradient(eta_V_vs_delta_e, delta_e_vals)
        d_eta_d_Q = np.gradient(eta_V_vs_Q, Q_vals)
        
        if self.verbose:
            nominal_eta = self.compute_voltage_efficiency(delta_e_mm, Q_mL_s)
            print(f"\n✓ Sensitivity Analysis Complete")
            print(f"  Nominal point: δ_e={delta_e_mm:.2f}mm, Q={Q_mL_s:.2f}mL/s")
            print(f"  Nominal η_V: {nominal_eta*100:.2f}%")
            print(f"\nSensitivity at nominal point:")
            
            idx_delta = np.argmin(np.abs(delta_e_vals - delta_e_mm))
            idx_Q = np.argmin(np.abs(Q_vals - Q_mL_s))
            print(f"  ∂η_V/∂δ_e: {d_eta_d_delta_e[idx_delta]*100:.4f} %/mm")
            print(f"  ∂η_V/∂Q: {d_eta_d_Q[idx_Q]*100:.4f} %/(mL/s)")
        
        return {
            'delta_e_vals': delta_e_vals,
            'Q_vals': Q_vals,
            'eta_V_vs_delta_e': eta_V_vs_delta_e,
            'eta_V_vs_Q': eta_V_vs_Q,
            'P_p_vs_delta_e': P_p_vs_delta_e,
            'P_p_vs_Q': P_p_vs_Q,
            'd_eta_d_delta_e': d_eta_d_delta_e,
            'd_eta_d_Q': d_eta_d_Q
        }
    
    def plot_sensitivity(
        self,
        sensitivity_data: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        
        ax = axes[0, 0]
        ax.plot(sensitivity_data['delta_e_vals'],
                sensitivity_data['eta_V_vs_delta_e'] * 100,
                'o-', linewidth=2, markersize=6, color='blue')
        ax.set_xlabel('Electrode Thickness [mm]', fontsize=11)
        ax.set_ylabel('Voltage Efficiency [%]', fontsize=11)
        ax.set_title('Efficiency vs Thickness', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        
        ax = axes[0, 1]
        ax.plot(sensitivity_data['Q_vals'],
                sensitivity_data['eta_V_vs_Q'] * 100,
                'o-', linewidth=2, markersize=6, color='green')
        ax.set_xlabel('Flow Rate [mL/s]', fontsize=11)
        ax.set_ylabel('Voltage Efficiency [%]', fontsize=11)
        ax.set_title('Efficiency vs Flow Rate', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        
        ax = axes[1, 0]
        ax.plot(sensitivity_data['delta_e_vals'],
                sensitivity_data['P_p_vs_delta_e'] * 1000,
                'o-', linewidth=2, markersize=6, color='red')
        ax.set_xlabel('Electrode Thickness [mm]', fontsize=11)
        ax.set_ylabel('Pumping Power [mW]', fontsize=11)
        ax.set_title('Pumping Power vs Thickness', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        
        ax = axes[1, 1]
        ax.plot(sensitivity_data['Q_vals'],
                sensitivity_data['P_p_vs_Q'] * 1000,
                'o-', linewidth=2, markersize=6, color='orange')
        ax.set_xlabel('Flow Rate [mL/s]', fontsize=11)
        ax.set_ylabel('Pumping Power [mW]', fontsize=11)
        ax.set_title('Pumping Power vs Flow Rate', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"✓ Saved sensitivity plot to {save_path}")
        
        plt.show()
    
    def generate_operating_map(
        self,
        delta_e_range: Tuple[float, float] = (3, 10),
        Q_range: Tuple[float, float] = (10, 50),
        n_delta: int = 20,
        n_Q: int = 20
    ) -> Dict[str, np.ndarray]:
        delta_e_grid = np.linspace(delta_e_range[0], delta_e_range[1], n_delta)
        Q_grid = np.linspace(Q_range[0], Q_range[1], n_Q)
        
        Delta_E, Q_mesh = np.meshgrid(delta_e_grid, Q_grid)
        
        eta_V_map = np.zeros_like(Delta_E)
        P_p_map = np.zeros_like(Delta_E)
        
        if self.verbose:
            print(f"\nGenerating operating map...")
            print(f"  δ_e: {delta_e_range[0]}-{delta_e_range[1]} mm ({n_delta} points)")
            print(f"  Q: {Q_range[0]}-{Q_range[1]} mL/s ({n_Q} points)")
        
        for i in range(n_Q):
            for j in range(n_delta):
                delta_e = Delta_E[i, j]
                Q = Q_mesh[i, j]
                eta_V_map[i, j] = self.compute_voltage_efficiency(delta_e, Q)
                P_p_map[i, j] = self.compute_pumping_power(Q, delta_e)
        
        if self.verbose:
            print(f"✓ Operating map complete")
            print(f"  η_V range: {eta_V_map.min()*100:.2f}% - {eta_V_map.max()*100:.2f}%")
            print(f"  P_p range: {P_p_map.min()*1000:.2f} - {P_p_map.max()*1000:.2f} mW")
        
        return {
            'delta_e_grid': delta_e_grid,
            'Q_grid': Q_grid,
            'Delta_E_mesh': Delta_E,
            'Q_mesh': Q_mesh,
            'eta_V_map': eta_V_map,
            'P_p_map': P_p_map
        }
    
    def plot_operating_map(
        self,
        operating_map: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        
        ax = axes[0]
        contourf = ax.contourf(
            operating_map['Q_mesh'],
            operating_map['Delta_E_mesh'],
            operating_map['eta_V_map'] * 100,
            levels=15,
            cmap='RdYlGn'
        )
        contour = ax.contour(
            operating_map['Q_mesh'],
            operating_map['Delta_E_mesh'],
            operating_map['eta_V_map'] * 100,
            levels=10,
            colors='black',
            linewidths=0.5,
            alpha=0.4
        )
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f%%')
        
        ax.set_xlabel('Flow Rate [mL/s]', fontsize=12)
        ax.set_ylabel('Electrode Thickness [mm]', fontsize=12)
        ax.set_title('Voltage Efficiency Map', fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label('Voltage Efficiency [%]', fontsize=11)
        
        
        ax = axes[1]
        contourf = ax.contourf(
            operating_map['Q_mesh'],
            operating_map['Delta_E_mesh'],
            operating_map['P_p_map'] * 1000,
            levels=15,
            cmap='YlOrRd'
        )
        contour = ax.contour(
            operating_map['Q_mesh'],
            operating_map['Delta_E_mesh'],
            operating_map['P_p_map'] * 1000,
            levels=10,
            colors='black',
            linewidths=0.5,
            alpha=0.4
        )
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
        
        ax.set_xlabel('Flow Rate [mL/s]', fontsize=12)
        ax.set_ylabel('Electrode Thickness [mm]', fontsize=12)
        ax.set_title('Pumping Power Map', fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label('Pumping Power [mW]', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"✓ Saved operating map to {save_path}")
        
        plt.show()


def main():
    print("="*70)
    print("VRFB Multi-Objective Optimizer - Complete Test Suite")
    print("="*70)
    
    
    optimizer = VRFBOptimizer(
        current_density_mA_cm2=150.0,
        temperature_C=25.0,
        SOC=0.5,
        electrolyte_conc_M=2.0,
        verbose=True
    )
    
    
    print("\n" + "="*70)
    print("1. SINGLE-POINT OPTIMIZATION")
    print("="*70)
    result = optimizer.optimize(beta=0.5)
    
    
    print("\n" + "="*70)
    print("2. PARETO FRONT GENERATION")
    print("="*70)
    design_pts, pareto_pts = optimizer.generate_pareto_front(n_points=10)
    
    
    print("\n" + "="*70)
    print("3. PARETO FRONT VISUALIZATION")
    print("="*70)
    optimizer.plot_pareto_front(
        design_pts,
        pareto_pts,
        save_path="results/figures/vrfb_pareto_front.png"
    )
    
    
    print("\n" + "="*70)
    print("4. SENSITIVITY ANALYSIS")
    print("="*70)
    sens_data = optimizer.sensitivity_analysis(
        delta_e_mm=result.optimal_thickness_mm,
        Q_mL_s=result.optimal_flow_rate_mL_s,
        n_points=20
    )
    
    
    print("\n" + "="*70)
    print("5. SENSITIVITY VISUALIZATION")
    print("="*70)
    optimizer.plot_sensitivity(
        sens_data,
        save_path="results/figures/vrfb_sensitivity.png"
    )
    
    
    print("\n" + "="*70)
    print("6. OPERATING MAP GENERATION")
    print("="*70)
    op_map = optimizer.generate_operating_map(
        delta_e_range=(3, 10),
        Q_range=(10, 50),
        n_delta=20,
        n_Q=20
    )
    
    
    print("\n" + "="*70)
    print("7. OPERATING MAP VISUALIZATION")
    print("="*70)
    optimizer.plot_operating_map(
        op_map,
        save_path="results/figures/vrfb_operating_map.png"
    )
    
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ VRFB Optimizer test complete")
    print(f"\nPareto front:")
    print(f"  η_V range: {pareto_pts[:,0].min()*100:.2f}% - {pareto_pts[:,0].max()*100:.2f}%")
    print(f"  P_p range: {pareto_pts[:,1].min()*100:.4f}% - {pareto_pts[:,1].max()*100:.4f}%")
    print(f"\nOptimal design (β=0.5):")
    print(f"  Thickness: {result.optimal_thickness_mm:.2f} mm")
    print(f"  Flow rate: {result.optimal_flow_rate_mL_s:.2f} mL/s")
    print(f"  Efficiency: {result.voltage_efficiency*100:.2f}%")
    print(f"  Pumping power: {result.pumping_power_W*1000:.2f} mW")
    print(f"\nGenerated figures:")
    print(f"  ✓ results/figures/vrfb_pareto_front.png")
    print(f"  ✓ results/figures/vrfb_sensitivity.png")
    print(f"  ✓ results/figures/vrfb_operating_map.png")
    print("="*70)


if __name__ == "__main__":
    main()
