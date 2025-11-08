"""
VRFB Multi-Objective Optimizer
Optimizes VRFB design for voltage efficiency and pumping power.

Implements Equations 6-7 from the paper.
"""

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
    """Results from VRFB multi-objective optimization."""
    optimal_thickness_mm: float  # Optimal electrode thickness
    optimal_flow_rate_mL_s: float  # Optimal flow rate
    
    voltage_efficiency: float  # Voltage efficiency
    pumping_power_W: float  # Pumping power
    normalized_pumping_power: float  # P_p / (i*A*E_Nernst)
    
    design_points: np.ndarray  # All evaluated design points
    pareto_front: np.ndarray  # Pareto optimal points
    
    constraint_violations: Dict  # Constraint check results


class VRFBOptimizer:
    """
    Multi-objective optimizer for VRFB design.
    
    Optimizes electrode thickness and flow rate to maximize voltage
    efficiency while minimizing pumping losses.
    """
    
    def __init__(
        self,
        current_density_mA_cm2: float = 150.0,
        temperature_C: float = 25.0,
        SOC: float = 0.5,
        electrolyte_conc_M: float = 2.0,
        verbose: bool = True
    ):
        """
        Initialize VRFB optimizer.
        
        Args:
            current_density_mA_cm2: Operating current density [mA/cm²]
            temperature_C: Operating temperature [°C]
            SOC: State of charge [0-1]
            electrolyte_conc_M: Vanadium concentration [M]
            verbose: Print progress messages
        """
        self.i = current_density_mA_cm2 / 1000.0  # Convert to A/cm²
        self.T_C = temperature_C
        self.T_K = temperature_C + 273.15
        self.SOC = SOC
        self.c = electrolyte_conc_M
        self.verbose = verbose
        
        # Precompute Nernst potential (Eq 6)
        self.E_nernst = self.compute_nernst_potential()
        
        # Physical constants from VRFBParams
        self.rho = VRFBParams.Electrolyte.density  # kg/m³
        self.mu = VRFBParams.Electrolyte.viscosity  # Pa·s
        self.D = VRFBParams.Electrolyte.diffusivity_V2  # m²/s
        
        # Electrode properties
        self.porosity = VRFBParams.Electrode.porosity
        self.conductivity = VRFBParams.Electrode.conductivity  # S/m
        
        # Membrane properties
        self.delta_m = VRFBParams.Membrane.thickness * 1e-6  # Convert μm to m
        self.kappa_m = VRFBParams.Membrane.conductivity  # S/m
        
        # Operating constraints
        self.delta_p_max = VRFBParams.Operating.delta_p_max  # Pa
        
        if self.verbose:
            print(f"✓ VRFB Optimizer initialized")
            print(f"  Current density: {current_density_mA_cm2} mA/cm²")
            print(f"  Temperature: {self.T_C}°C")
            print(f"  SOC: {self.SOC}")
            print(f"  Nernst potential: {self.E_nernst:.3f} V")
    
    def compute_nernst_potential(self) -> float:
        """
        Compute Nernst potential for VRFB (Equation 6).
        
        E_Nernst = E°(T) + (RT/F) * ln([V³⁺][VO₂⁺] / [V²⁺][VO₂²⁺])
        
        Returns:
            Nernst potential [V]
        """
        # Standard potential (temperature dependent)
        E_std = VRFBParams.E0_standard + VRFBParams.dE0_dT * (self.T_K - 298.15)
        
        # Concentration ratio
        # Assuming [V³⁺] = [VO₂⁺] = SOC * c
        # and [V²⁺] = [VO₂²⁺] = (1-SOC) * c
        if self.SOC <= 0.01:
            self.SOC = 0.01  # Avoid log(0)
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
        """
        Compute mass transfer coefficient using Sherwood correlation.
        
        Sh = a * Re^b * Sc^(1/3)
        k_m = Sh * D / d_h
        
        Args:
            Q_mL_s: Flow rate [mL/s]
            d_h_mm: Hydraulic diameter [mm]
        
        Returns:
            Mass transfer coefficient [m/s]
        """
        # Convert units
        Q = Q_mL_s * 1e-6  # mL/s to m³/s
        d_h = d_h_mm * 1e-3  # mm to m
        
        # Assume electrode area and cross-sectional area
        A_electrode = 0.01  # 10 cm² = 0.001 m²
        A_cross = 5e-6  # Approximate cross-sectional area [m²]
        
        # Velocity
        u = Q / A_cross  # m/s
        
        # Reynolds number
        Re = (self.rho * u * d_h) / self.mu
        
        # Schmidt number
        Sc = self.mu / (self.rho * self.D)
        
        # Sherwood correlation (from VRFBParams)
        a = VRFBParams.Flow.sherwood_a
        b = VRFBParams.Flow.sherwood_b
        Sh = a * (Re ** b) * (Sc ** (1/3))
        
        # Mass transfer coefficient
        k_m = Sh * self.D / d_h
        
        return k_m
    
    def compute_limiting_current(
        self,
        Q_mL_s: float,
        d_h_mm: float = 3.75
    ) -> float:
        """
        Compute limiting current density (Equation 7).
        
        i_L = n * F * k_m * c
        
        Args:
            Q_mL_s: Flow rate [mL/s]
            d_h_mm: Hydraulic diameter [mm]
        
        Returns:
            Limiting current density [A/cm²]
        """
        k_m = self.compute_mass_transfer_coefficient(Q_mL_s, d_h_mm)
        
        # Concentration in mol/m³
        c_mol_m3 = self.c * 1000  # M to mol/m³
        
        # Limiting current density
        i_L = F * k_m * c_mol_m3  # A/m²
        i_L_A_cm2 = i_L / 10000  # Convert to A/cm²
        
        return i_L_A_cm2
    
    def compute_activation_loss(
        self,
        i0: float = 0.005,
        alpha: float = 0.5
    ) -> Tuple[float, float]:
        """
        Compute activation overpotential for charge and discharge.
        
        η_act = (RT/αF) * ln(i/i0)
        
        Args:
            i0: Exchange current density [A/cm²]
            alpha: Transfer coefficient
        
        Returns:
            (η_charge, η_discharge) [V]
        """
        RT_alphaF = (R * self.T_K) / (alpha * F)
        
        # Avoid log(0)
        i_safe = max(self.i, 1e-10)
        eta_act = RT_alphaF * np.log(i_safe / i0)
        
        return eta_act, eta_act  # Assume symmetric
    
    def compute_ohmic_resistance(
        self,
        delta_e_mm: float
    ) -> float:
        """
        Compute ohmic resistance.
        
        R_Ω = δ_e / (κ_felt * A) + δ_m / (κ_m * A)
        
        Args:
            delta_e_mm: Electrode thickness [mm]
        
        Returns:
            Ohmic resistance [Ω·cm²]
        """
        delta_e = delta_e_mm * 1e-3  # mm to m
        
        # Felt resistance
        R_felt = delta_e / self.conductivity  # Ω·m²
        
        # Membrane resistance
        R_mem = self.delta_m / self.kappa_m  # Ω·m²
        
        # Total resistance
        R_ohm = R_felt + R_mem  # Ω·m²
        R_ohm_cm2 = R_ohm * 10000  # Convert to Ω·cm²
        
        return R_ohm_cm2
    
    def compute_mass_transfer_loss(
        self,
        Q_mL_s: float
    ) -> Tuple[float, float]:
        """
        Compute mass-transfer overpotential for charge and discharge.
        
        η_mt = -(RT/F) * ln(1 - i/i_L)
        
        Args:
            Q_mL_s: Flow rate [mL/s]
        
        Returns:
            (η_mt_charge, η_mt_discharge) [V]
        """
        i_L = self.compute_limiting_current(Q_mL_s)
        
        RT_F = (R * self.T_K) / F
        
        # Avoid i >= i_L
        ratio = min(self.i / i_L, 0.999)
        eta_mt = -RT_F * np.log(1 - ratio)
        
        return eta_mt, eta_mt  # Assume symmetric
    
    def compute_voltage_efficiency(
        self,
        delta_e_mm: float,
        Q_mL_s: float,
        i0: float = 0.005,
        alpha: float = 0.5
    ) -> float:
        """
        Compute voltage efficiency (Equation 7).
        
        η_V ≈ (E_Nernst - η_dis) / (E_Nernst + η_ch)
        
        Args:
            delta_e_mm: Electrode thickness [mm]
            Q_mL_s: Flow rate [mL/s]
            i0: Exchange current density [A/cm²]
            alpha: Transfer coefficient
        
        Returns:
            Voltage efficiency [dimensionless]
        """
        # Compute losses
        eta_act_ch, eta_act_dis = self.compute_activation_loss(i0, alpha)
        R_ohm = self.compute_ohmic_resistance(delta_e_mm)
        eta_ohm = self.i * R_ohm
        eta_mt_ch, eta_mt_dis = self.compute_mass_transfer_loss(Q_mL_s)
        
        # Total overpotentials
        eta_ch = eta_act_ch + eta_ohm + eta_mt_ch
        eta_dis = eta_act_dis + eta_ohm + eta_mt_dis
        
        # Voltage efficiency
        V_dis = self.E_nernst - eta_dis
        V_ch = self.E_nernst + eta_ch
        eta_V = V_dis / V_ch
        
        return eta_V
    
    def compute_pumping_power(
        self,
        Q_mL_s: float,
        delta_e_mm: float
    ) -> float:
        """
        Compute pumping power.
        
        P_p = Q * Δp
        Δp = f * (ρ * Q²) / (2 * d_h * A_cross²)
        
        Args:
            Q_mL_s: Flow rate [mL/s]
            delta_e_mm: Electrode thickness [mm]
        
        Returns:
            Pumping power [W]
        """
        # Convert units
        Q = Q_mL_s * 1e-6  # mL/s to m³/s
        delta_e = delta_e_mm * 1e-3  # mm to m
        
        # Approximate cross-sectional area
        A_cross = 5e-6  # m²
        
        # Hydraulic diameter
        d_h = 3.75e-3  # m
        
        # Velocity
        u = Q / A_cross
        
        # Reynolds number
        Re = (self.rho * u * d_h) / self.mu
        
        # Friction factor (Darcy-Weisbach, laminar/turbulent)
        if Re < 2300:
            f = 64 / Re  # Laminar
        else:
            f = 0.316 / (Re ** 0.25)  # Turbulent (Blasius)
        
        # Pressure drop
        L = delta_e  # Flow path length ≈ electrode thickness
        delta_p = f * (L / d_h) * (0.5 * self.rho * u**2)
        
        # Pumping power
        P_p = Q * delta_p
        
        return P_p
    
    def objective_function(
        self,
        design_vars: np.ndarray,
        beta: float = 0.5
    ) -> float:
        """
        Multi-objective function combining voltage efficiency and pumping power.
        
        f = -η_V + β * (P_p / P_ref)
        
        Args:
            design_vars: [delta_e_mm, Q_mL_s]
            beta: Trade-off weight between efficiency and pumping
        
        Returns:
            Objective value (to minimize)
        """
        delta_e_mm, Q_mL_s = design_vars
        
        # Compute metrics
        eta_V = self.compute_voltage_efficiency(delta_e_mm, Q_mL_s)
        P_p = self.compute_pumping_power(Q_mL_s, delta_e_mm)
        
        # Reference power (electrochemical power)
        A_electrode = 0.01  # 10 cm² = 0.001 m²
        P_ref = self.i * A_electrode * 10000 * self.E_nernst  # W
        
        # Normalized pumping power
        P_p_norm = P_p / P_ref
        
        # Multi-objective
        obj = -eta_V + beta * P_p_norm
        
        return obj
    
    def check_constraints(
        self,
        delta_e_mm: float,
        Q_mL_s: float
    ) -> Dict[str, bool]:
        """
        Check design constraints.
        
        Args:
            delta_e_mm: Electrode thickness [mm]
            Q_mL_s: Flow rate [mL/s]
        
        Returns:
            Dict of constraint satisfaction
        """
        # Limiting current constraint
        i_L = self.compute_limiting_current(Q_mL_s)
        current_ok = self.i < i_L
        
        # Pressure drop constraint
        P_p = self.compute_pumping_power(Q_mL_s, delta_e_mm)
        Q = Q_mL_s * 1e-6
        delta_p = P_p / Q if Q > 0 else 0
        pressure_ok = delta_p < self.delta_p_max
        
        # Thickness bounds
        thickness_ok = (VRFBParams.Electrode.thickness_min <= delta_e_mm <= 
                       VRFBParams.Electrode.thickness_max)
        
        # Flow rate bounds
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
        """
        Perform multi-objective optimization.
        
        Args:
            beta: Trade-off weight (0=max efficiency, 1=min pumping)
            method: Optimization method ('differential_evolution' or 'minimize')
        
        Returns:
            VRFBOptimizationResult with optimal design
        """
        # Bounds
        bounds = [
            (VRFBParams.Electrode.thickness_min, VRFBParams.Electrode.thickness_max),  # delta_e [mm]
            (VRFBParams.Flow.Q_min, VRFBParams.Flow.Q_max)  # Q [mL/s]
        ]
        
        if self.verbose:
            print(f"\nOptimizing VRFB design...")
            print(f"  Trade-off parameter β: {beta}")
            print(f"  Method: {method}")
        
        # Optimize
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
            x0 = [5.0, 30.0]  # Initial guess
            result = minimize(
                lambda x: self.objective_function(x, beta),
                x0,
                bounds=bounds,
                method='L-BFGS-B'
            )
        
        # Extract results
        delta_e_opt, Q_opt = result.x
        eta_V_opt = self.compute_voltage_efficiency(delta_e_opt, Q_opt)
        P_p_opt = self.compute_pumping_power(Q_opt, delta_e_opt)
        
        # Normalized pumping power
        A_electrode = 0.01
        P_ref = self.i * A_electrode * 10000 * self.E_nernst
        P_p_norm = P_p_opt / P_ref
        
        # Check constraints
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
        """
        Generate Pareto front by varying trade-off parameter.
        
        Args:
            n_points: Number of points on Pareto front
        
        Returns:
            (design_points, pareto_points) arrays
        """
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
        """
        Plot Pareto front with design space visualization.
        
        Args:
            design_points: Array of (δ_e, Q) design points
            pareto_points: Array of (η_V, P_p) performance points
            save_path: Path to save figure (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Pareto front (performance space)
        ax = axes[0]
        scatter = ax.scatter(
            pareto_points[:, 1] * 100,  # P_p in %
            pareto_points[:, 0] * 100,  # η_V in %
            c=design_points[:, 0],      # Color by thickness
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
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Electrode Thickness [mm]', fontsize=10)
        
        # Annotate optimal points
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
        
        # 2. Design space (δ_e vs Q)
        ax = axes[1]
        scatter = ax.scatter(
            design_points[:, 1],  # Q [mL/s]
            design_points[:, 0],  # δ_e [mm]
            c=pareto_points[:, 0] * 100,  # Color by η_V
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
        
        # Add colorbar
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
        """
        Perform sensitivity analysis around a design point.
        
        Args:
            delta_e_mm: Nominal electrode thickness [mm]
            Q_mL_s: Nominal flow rate [mL/s]
            delta_e_range: Range for thickness variation (optional)
            Q_range: Range for flow rate variation (optional)
            n_points: Number of points for each parameter
        
        Returns:
            Dict with sensitivity data
        """
        if delta_e_range is None:
            delta_e_range = (delta_e_mm * 0.7, delta_e_mm * 1.3)
        if Q_range is None:
            Q_range = (Q_mL_s * 0.7, Q_mL_s * 1.3)
        
        # Vary thickness (holding flow constant)
        delta_e_vals = np.linspace(delta_e_range[0], delta_e_range[1], n_points)
        eta_V_vs_delta_e = np.array([
            self.compute_voltage_efficiency(d, Q_mL_s) for d in delta_e_vals
        ])
        P_p_vs_delta_e = np.array([
            self.compute_pumping_power(Q_mL_s, d) for d in delta_e_vals
        ])
        
        # Vary flow (holding thickness constant)
        Q_vals = np.linspace(Q_range[0], Q_range[1], n_points)
        eta_V_vs_Q = np.array([
            self.compute_voltage_efficiency(delta_e_mm, q) for q in Q_vals
        ])
        P_p_vs_Q = np.array([
            self.compute_pumping_power(q, delta_e_mm) for q in Q_vals
        ])
        
        # Compute derivatives (numerical)
        d_eta_d_delta_e = np.gradient(eta_V_vs_delta_e, delta_e_vals)
        d_eta_d_Q = np.gradient(eta_V_vs_Q, Q_vals)
        
        if self.verbose:
            nominal_eta = self.compute_voltage_efficiency(delta_e_mm, Q_mL_s)
            print(f"\n✓ Sensitivity Analysis Complete")
            print(f"  Nominal point: δ_e={delta_e_mm:.2f}mm, Q={Q_mL_s:.2f}mL/s")
            print(f"  Nominal η_V: {nominal_eta*100:.2f}%")
            print(f"\nSensitivity at nominal point:")
            # Find closest indices
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
        """
        Plot sensitivity analysis results.
        
        Args:
            sensitivity_data: Output from sensitivity_analysis()
            save_path: Path to save figure (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. η_V vs δ_e
        ax = axes[0, 0]
        ax.plot(sensitivity_data['delta_e_vals'],
                sensitivity_data['eta_V_vs_delta_e'] * 100,
                'o-', linewidth=2, markersize=6, color='blue')
        ax.set_xlabel('Electrode Thickness [mm]', fontsize=11)
        ax.set_ylabel('Voltage Efficiency [%]', fontsize=11)
        ax.set_title('Efficiency vs Thickness', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 2. η_V vs Q
        ax = axes[0, 1]
        ax.plot(sensitivity_data['Q_vals'],
                sensitivity_data['eta_V_vs_Q'] * 100,
                'o-', linewidth=2, markersize=6, color='green')
        ax.set_xlabel('Flow Rate [mL/s]', fontsize=11)
        ax.set_ylabel('Voltage Efficiency [%]', fontsize=11)
        ax.set_title('Efficiency vs Flow Rate', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 3. P_p vs δ_e
        ax = axes[1, 0]
        ax.plot(sensitivity_data['delta_e_vals'],
                sensitivity_data['P_p_vs_delta_e'] * 1000,
                'o-', linewidth=2, markersize=6, color='red')
        ax.set_xlabel('Electrode Thickness [mm]', fontsize=11)
        ax.set_ylabel('Pumping Power [mW]', fontsize=11)
        ax.set_title('Pumping Power vs Thickness', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 4. P_p vs Q
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
        """
        Generate 2D operating map (contour plot data).
        
        Args:
            delta_e_range: Range of electrode thicknesses [mm]
            Q_range: Range of flow rates [mL/s]
            n_delta: Number of thickness points
            n_Q: Number of flow points
        
        Returns:
            Dict with mesh grids and performance maps
        """
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
        """
        Plot operating map with contours.
        
        Args:
            operating_map: Output from generate_operating_map()
            save_path: Path to save figure (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Voltage efficiency contours
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
        
        # 2. Pumping power contours
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
    """
    Test VRFB optimizer with complete diagnostics.
    """
    print("="*70)
    print("VRFB Multi-Objective Optimizer - Complete Test Suite")
    print("="*70)
    
    # Initialize optimizer
    optimizer = VRFBOptimizer(
        current_density_mA_cm2=150.0,
        temperature_C=25.0,
        SOC=0.5,
        electrolyte_conc_M=2.0,
        verbose=True
    )
    
    # 1. Single-point optimization
    print("\n" + "="*70)
    print("1. SINGLE-POINT OPTIMIZATION")
    print("="*70)
    result = optimizer.optimize(beta=0.5)
    
    # 2. Generate Pareto front
    print("\n" + "="*70)
    print("2. PARETO FRONT GENERATION")
    print("="*70)
    design_pts, pareto_pts = optimizer.generate_pareto_front(n_points=10)
    
    # 3. Plot Pareto front
    print("\n" + "="*70)
    print("3. PARETO FRONT VISUALIZATION")
    print("="*70)
    optimizer.plot_pareto_front(
        design_pts,
        pareto_pts,
        save_path="results/figures/vrfb_pareto_front.png"
    )
    
    # 4. Sensitivity analysis
    print("\n" + "="*70)
    print("4. SENSITIVITY ANALYSIS")
    print("="*70)
    sens_data = optimizer.sensitivity_analysis(
        delta_e_mm=result.optimal_thickness_mm,
        Q_mL_s=result.optimal_flow_rate_mL_s,
        n_points=20
    )
    
    # 5. Plot sensitivity
    print("\n" + "="*70)
    print("5. SENSITIVITY VISUALIZATION")
    print("="*70)
    optimizer.plot_sensitivity(
        sens_data,
        save_path="results/figures/vrfb_sensitivity.png"
    )
    
    # 6. Operating map
    print("\n" + "="*70)
    print("6. OPERATING MAP GENERATION")
    print("="*70)
    op_map = optimizer.generate_operating_map(
        delta_e_range=(3, 10),
        Q_range=(10, 50),
        n_delta=20,
        n_Q=20
    )
    
    # 7. Plot operating map
    print("\n" + "="*70)
    print("7. OPERATING MAP VISUALIZATION")
    print("="*70)
    optimizer.plot_operating_map(
        op_map,
        save_path="results/figures/vrfb_operating_map.png"
    )
    
    # Summary
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
