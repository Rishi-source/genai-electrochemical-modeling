"""
Mechanistic Solver Baseline
Benchmarks the physics-based solvers against ML baselines.

Provides timing and accuracy metrics for:
- PEMFC parameter fitting (0D lumped model)
- VRFB multi-objective optimization

This establishes the "ground truth" baseline for comparison.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.solvers.pemfc_fitter import PEMFCFitter
from src.solvers.vrfb_optimizer import VRFBOptimizer


@dataclass
class MechanisticBenchmarkResult:
    """Results from mechanistic solver benchmarking."""
    # PEMFC results
    pemfc_rmse: float
    pemfc_mae: float
    pemfc_r_squared: float
    pemfc_time: float
    pemfc_iterations: int
    
    # VRFB results
    vrfb_pareto_points: int
    vrfb_best_efficiency: float
    vrfb_time: float
    vrfb_evaluations: int
    
    # Comparison metrics
    pemfc_params: Dict[str, float]
    vrfb_optimal_design: Dict[str, float]


class MechanisticBenchmark:
    """
    Benchmark mechanistic solvers.
    
    Runs physics-based solvers and collects:
    - Accuracy metrics (RMSE, MAE, R²)
    - Computational cost (wall-clock time, iterations)
    - Convergence behavior
    - Parameter values
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize benchmark.
        
        Args:
            verbose: Print progress
        """
        self.verbose = verbose
        
    def benchmark_pemfc(
        self,
        data_path: str = "data/synthetic/pemfc_polarization.csv",
        test_fraction: float = 0.15
    ) -> Dict[str, float]:
        """
        Benchmark PEMFC parameter fitting.
        
        Args:
            data_path: Path to polarization data
            test_fraction: Fraction of data for testing
        
        Returns:
            Benchmark metrics dictionary
        """
        if self.verbose:
            print("\n" + "="*70)
            print("Benchmarking PEMFC Mechanistic Solver")
            print("="*70)
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Split data
        n_test = int(len(df) * test_fraction)
        df_train = df.iloc[:-n_test]
        df_test = df.iloc[-n_test:]
        
        # Initialize fitter
        fitter = PEMFCFitter()
        
        # Benchmark fitting
        start_time = time.time()
        
        i_train = df_train['current_density_A_cm2'].values
        V_train = df_train['voltage_V'].values
        
        result = fitter.fit(i_train, V_train)
        
        fit_time = time.time() - start_time
        
        # Evaluate on test set
        i_test = df_test['current_density_A_cm2'].values
        V_test = df_test['voltage_V'].values
        
        # Use fitted parameters to predict
        params_array = np.array([
            result.i0_A_cm2,
            result.alpha,
            result.R_ohm_cm2,
            result.i_L_A_cm2
        ])
        V_pred = fitter.voltage_model(params_array, i_test)
        
        # Metrics
        residuals = V_test - V_pred
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((V_test - V_test.mean())**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        if self.verbose:
            print(f"\n✓ Fitting complete")
            print(f"  Wall-clock time: {fit_time:.3f} s")
            print(f"  Iterations: {result.convergence_info.get('nit', 'N/A')}")
            print(f"\nTest Set Performance:")
            print(f"  RMSE: {rmse*1000:.2f} mV")
            print(f"  MAE: {mae*1000:.2f} mV")
            print(f"  R²: {r_squared:.4f}")
            print(f"\nFitted Parameters:")
            print(f"  i0: {result.i0_A_cm2:.6e}")
            print(f"  alpha: {result.alpha:.6e}")
            print(f"  R_ohm: {result.R_ohm_cm2:.6e}")
            print(f"  i_L: {result.i_L_A_cm2:.6e}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r_squared': r_squared,
            'time': fit_time,
            'iterations': result.convergence_info.get('nit', 0),
            'params': {
                'i0': result.i0_A_cm2,
                'alpha': result.alpha,
                'R_ohm': result.R_ohm_cm2,
                'i_L': result.i_L_A_cm2
            },
            'n_train': len(df_train),
            'n_test': len(df_test)
        }
    
    def benchmark_vrfb(
        self,
        target_current: float = 200.0,  # mA/cm²
        n_designs: int = 50
    ) -> Dict[str, float]:
        """
        Benchmark VRFB multi-objective optimization.
        
        Args:
            target_current: Target operating current density
            n_designs: Number of design points to evaluate
        
        Returns:
            Benchmark metrics dictionary
        """
        if self.verbose:
            print("\n" + "="*70)
            print("Benchmarking VRFB Mechanistic Solver")
            print("="*70)
        
        # Initialize optimizer
        optimizer = VRFBOptimizer()
        
        # Benchmark optimization
        start_time = time.time()
        
        result = optimizer.optimize(beta=0.5)
        
        opt_time = time.time() - start_time
        
        if self.verbose:
            print(f"\n✓ Optimization complete")
            print(f"  Wall-clock time: {opt_time:.3f} s")
            print(f"  Function evaluations: {len(result.design_points)}")
            print(f"\nOptimal Design:")
            print(f"  Electrode thickness: {result.optimal_thickness_mm:.2f} mm")
            print(f"  Flow rate: {result.optimal_flow_rate_mL_s:.2f} mL/s")
            print(f"\nPerformance:")
            print(f"  Voltage efficiency: {result.voltage_efficiency*100:.2f}%")
            print(f"  Pumping power: {result.pumping_power_W:.4f} W")
        
        return {
            'time': opt_time,
            'evaluations': len(result.design_points),
            'optimal_efficiency': result.voltage_efficiency,
            'optimal_thickness': result.optimal_thickness_mm,
            'optimal_flow_rate': result.optimal_flow_rate_mL_s,
            'pumping_power': result.pumping_power_W,
            'pareto_points': len(result.pareto_front) if hasattr(result, 'pareto_front') else 1
        }
    
    def run_full_benchmark(
        self,
        save_path: Optional[str] = None
    ) -> MechanisticBenchmarkResult:
        """
        Run complete benchmark suite.
        
        Args:
            save_path: Path to save results JSON
        
        Returns:
            Complete benchmark results
        """
        print("\n" + "="*70)
        print("MECHANISTIC SOLVER BENCHMARK")
        print("="*70)
        
        # PEMFC benchmark
        pemfc_results = self.benchmark_pemfc()
        
        # VRFB benchmark
        vrfb_results = self.benchmark_vrfb()
        
        # Summary
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        print(f"\n📊 PEMFC Mechanistic Solver:")
        print(f"  Accuracy: R²={pemfc_results['r_squared']:.4f}, RMSE={pemfc_results['rmse']*1000:.2f}mV")
        print(f"  Speed: {pemfc_results['time']:.3f}s ({pemfc_results['iterations']} iterations)")
        print(f"  Efficiency: {pemfc_results['n_train']/pemfc_results['time']:.1f} samples/second")
        
        print(f"\n📊 VRFB Mechanistic Solver:")
        print(f"  Optimal efficiency: {vrfb_results['optimal_efficiency']*100:.2f}%")
        print(f"  Speed: {vrfb_results['time']:.3f}s ({vrfb_results['evaluations']} evaluations)")
        print(f"  Throughput: {vrfb_results['evaluations']/vrfb_results['time']:.1f} designs/second")
        
        # Create result object
        result = MechanisticBenchmarkResult(
            pemfc_rmse=pemfc_results['rmse'],
            pemfc_mae=pemfc_results['mae'],
            pemfc_r_squared=pemfc_results['r_squared'],
            pemfc_time=pemfc_results['time'],
            pemfc_iterations=pemfc_results['iterations'],
            vrfb_pareto_points=vrfb_results['pareto_points'],
            vrfb_best_efficiency=vrfb_results['optimal_efficiency'],
            vrfb_time=vrfb_results['time'],
            vrfb_evaluations=vrfb_results['evaluations'],
            pemfc_params=pemfc_results['params'],
            vrfb_optimal_design={
                'electrode_thickness_mm': vrfb_results['optimal_thickness'],
                'flow_rate_mL_s': vrfb_results['optimal_flow_rate'],
                'voltage_efficiency': vrfb_results['optimal_efficiency'],
                'pumping_power_W': vrfb_results['pumping_power']
            }
        )
        
        # Save results
        if save_path:
            import json
            results_dict = {
                'pemfc': {
                    'rmse_mV': pemfc_results['rmse'] * 1000,
                    'mae_mV': pemfc_results['mae'] * 1000,
                    'r_squared': pemfc_results['r_squared'],
                    'time_s': pemfc_results['time'],
                    'iterations': pemfc_results['iterations'],
                    'params': {k: float(v) for k, v in pemfc_results['params'].items()}
                },
                'vrfb': {
                    'optimal_efficiency_%': vrfb_results['optimal_efficiency'] * 100,
                    'optimal_thickness_mm': vrfb_results['optimal_thickness'],
                    'optimal_flow_rate_mL_s': vrfb_results['optimal_flow_rate'],
                    'pumping_power_W': vrfb_results['pumping_power'],
                    'time_s': vrfb_results['time'],
                    'evaluations': vrfb_results['evaluations']
                }
            }
            
            with open(save_path, 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            if self.verbose:
                print(f"\n✓ Results saved to {save_path}")
        
        print("="*70)
        
        return result


def main():
    """
    Run mechanistic solver benchmark.
    """
    benchmark = MechanisticBenchmark(verbose=True)
    
    result = benchmark.run_full_benchmark(
        save_path="results/tables/mechanistic_benchmark.json"
    )
    
    print("\n✓ Mechanistic solver benchmark complete!")
    print(f"\nKey Findings:")
    print(f"  • Physics-based models provide ground-truth accuracy")
    print(f"  • No training data required (except for parameter fitting)")
    print(f"  • Fast inference (~{result.pemfc_time + result.vrfb_time:.1f}s total)")
    print(f"  • Fully interpretable physical parameters")


if __name__ == "__main__":
    main()
