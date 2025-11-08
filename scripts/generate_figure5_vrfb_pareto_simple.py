"""
Generate Figure 5: Enhanced VRFB Pareto Front (Using Actual Solver)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.solvers.vrfb_optimizer import VRFBOptimizer

print("Generating enhanced VRFB Pareto front...")

# Initialize optimizer
optimizer = VRFBOptimizer()

# Define design space
electrode_thicknesses = [3, 5, 7, 10]  # mm
flow_rates = np.linspace(15, 45, 15)  # mL/s
current_density = 0.15  # A/cm²

# Generate design points using actual solver
design_points = []
for delta_e in electrode_thicknesses:
    for Q in flow_rates:
        eta_V = optimizer.compute_voltage_efficiency(delta_e, Q)
        P_p = optimizer.compute_pumping_power(Q, delta_e)
        
        if eta_V > 0 and eta_V < 1.0:
            design_points.append({
                'delta_e': delta_e,
                'Q': Q,
                'eta_V': eta_V * 100,  # Convert to percentage
                'P_p': P_p,
                'P_p_norm': P_p / (current_density * 1.26)
            })

print(f"✓ Generated {len(design_points)} valid design points")

# Convert to arrays
delta_e_array = np.array([d['delta_e'] for d in design_points])
Q_array = np.array([d['Q'] for d in design_points])
eta_V_array = np.array([d['eta_V'] for d in design_points])
P_p_array = np.array([d['P_p'] for d in design_points])
P_p_norm_array = np.array([d['P_p_norm'] for d in design_points])

# Create enhanced figure
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[2, 1])

# Main Pareto front
ax_main = fig.add_subplot(gs[0, :])

# Scatter plot colored by electrode thickness
scatter = ax_main.scatter(P_p_norm_array * 100, eta_V_array, 
                          c=delta_e_array, cmap='viridis', s=120, 
                          alpha=0.8, edgecolors='black', linewidth=1.5)

# Highlight optimal points
for thickness in electrode_thicknesses:
    mask = delta_e_array == thickness
    if mask.any():
        idx_max = np.argmax(eta_V_array[mask])
        Q_vals = Q_array[mask]
        eta_vals = eta_V_array[mask]
        P_vals = P_p_norm_array[mask]
        ax_main.scatter(P_vals[idx_max] * 100, eta_vals[idx_max],
                       s=400, marker='*', c='red', edgecolors='darkred',
                       linewidth=2.5, zorder=10)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax_main)
cbar.set_label('Electrode Thickness (mm)', fontsize=13, fontweight='bold')

# Labels
ax_main.set_xlabel('Normalized Pumping Power (%)', fontsize=14, fontweight='bold')
ax_main.set_ylabel('Voltage Efficiency (%)', fontsize=14, fontweight='bold')
ax_main.set_title('VRFB Multi-Objective Optimization: Pareto Front\n(i = 150 mA/cm²)', 
                  fontsize=15, fontweight='bold')
ax_main.grid(True, alpha=0.3, linewidth=0.8)

# Annotations
best_idx = np.argmax(eta_V_array)
ax_main.annotate(f'Optimal: η_V={eta_V_array[best_idx]:.1f}%\nδₑ={delta_e_array[best_idx]:.0f}mm, Q={Q_array[best_idx]:.1f}mL/s',
                xy=(P_p_norm_array[best_idx]*100, eta_V_array[best_idx]),
                xytext=(P_p_norm_array[best_idx]*100 + 0.5, eta_V_array[best_idx] - 2),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                arrowprops=dict(arrowstyle='->', lw=2))

# Contour map
ax_contour = fig.add_subplot(gs[1, 0])

# Create grid
delta_e_grid = np.array(electrode_thicknesses)
Q_grid = flow_rates
Delta_E_mesh, Q_mesh = np.meshgrid(delta_e_grid, Q_grid)
Eta_V_grid = np.zeros_like(Delta_E_mesh)

for i in range(len(Q_grid)):
    for j in range(len(delta_e_grid)):
        eta_v = optimizer.compute_voltage_efficiency(
            Delta_E_mesh[i, j],
            Q_mesh[i, j]
        )
        Eta_V_grid[i, j] = eta_v * 100  # Convert to percentage

# Contour plot
contour = ax_contour.contourf(Delta_E_mesh, Q_mesh, Eta_V_grid, levels=12, cmap='RdYlGn')
contour_lines = ax_contour.contour(Delta_E_mesh, Q_mesh, Eta_V_grid, levels=8, 
                                   colors='black', linewidths=0.8, alpha=0.4)
ax_contour.clabel(contour_lines, inline=True, fontsize=9, fmt='%1.1f%%')

# Mark optimal designs
for thickness in electrode_thicknesses:
    mask = delta_e_array == thickness
    if mask.any():
        optimal_Q = Q_array[mask][np.argmax(eta_V_array[mask])]
        ax_contour.scatter(thickness, optimal_Q, s=200, marker='*', 
                          c='red', edgecolors='darkred', linewidth=2, zorder=10)

ax_contour.set_xlabel('Electrode Thickness (mm)', fontsize=12, fontweight='bold')
ax_contour.set_ylabel('Flow Rate (mL/s)', fontsize=12, fontweight='bold')
ax_contour.set_title('Efficiency Contour Map', fontsize=13, fontweight='bold')
plt.colorbar(contour, ax=ax_contour, label='η_V (%)', pad=0.02)

# Summary table
ax_table = fig.add_subplot(gs[1, 1])
ax_table.axis('off')

table_text = f"""
VRFB Design Space Summary

Configuration:
  • Current: {current_density*1000:.0f} mA/cm²
  • Temperature: 25°C
  • Vanadium: 2M in 4M H₂SO₄

Design Variables:
  • δₑ: {electrode_thicknesses[0]}-{electrode_thicknesses[-1]} mm
  • Q: {flow_rates[0]:.0f}-{flow_rates[-1]:.0f} mL/s

Performance Range:
  • η_V: {eta_V_array.min():.1f}% - {eta_V_array.max():.1f}%
  • P_p: {P_p_array.min()*1000:.1f} - {P_p_array.max()*1000:.1f} mW

Optimal Design:
  • δₑ: {delta_e_array[best_idx]:.0f} mm
  • Q: {Q_array[best_idx]:.1f} mL/s
  • η_V: {eta_V_array[best_idx]:.2f}%
  • P_p: {P_p_array[best_idx]*1000:.2f} mW

Trade-offs:
  ↑ δₑ → ↑ R_Ω → ↓ η_V
  ↑ Q → ↑ k_m → ↑ η_V (but ↑ P_p)

Design Points: {len(design_points)}
"""

ax_table.text(0.05, 0.98, table_text, transform=ax_table.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

plt.tight_layout()
plt.savefig('results/figures/figure5_vrfb_pareto_enhanced.png', dpi=300, bbox_inches='tight')
print("✓ Figure 5 saved: results/figures/figure5_vrfb_pareto_enhanced.png")
plt.close()

# Print summary
print("\n" + "="*70)
print("FIGURE 5: VRFB PARETO FRONT SUMMARY")
print("="*70)
print(f"\nDesign Space: {len(design_points)} points")
print(f"  Electrode thickness: {electrode_thicknesses[0]}-{electrode_thicknesses[-1]} mm")
print(f"  Flow rate: {flow_rates[0]:.1f}-{flow_rates[-1]:.1f} mL/s")
print(f"\nOptimal Performance:")
print(f"  η_V: {eta_V_array.max():.2f}% at δₑ={delta_e_array[best_idx]:.0f}mm, Q={Q_array[best_idx]:.1f}mL/s")
print(f"\nTrade-off Range:")
print(f"  Efficiency: {eta_V_array.min():.1f}% - {eta_V_array.max():.1f}%")
print(f"  Pumping power: {P_p_array.min()*1000:.2f} - {P_p_array.max()*1000:.2f} mW")
print("="*70)
print("\n✓ Enhanced VRFB Pareto front generation complete!")
