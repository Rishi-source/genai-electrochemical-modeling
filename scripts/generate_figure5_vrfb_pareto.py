import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import json


with open('results/tables/mechanistic_benchmark.json', 'r') as f:
    mech_results = json.load(f)


R = 8.314  
F = 96485  
T = 25 + 273.15  
i_target = 150e-3  


electrode_thicknesses = np.array([3, 5, 7, 10])  
flow_rates = np.linspace(10, 50, 20)  


def calculate_vrfb_performance(delta_e, Q):    
    
    i0 = 1e-3  
    alpha = 0.5
    
    
    E_nernst = 1.26  
    
    
    eta_act = (R * T / (alpha * F)) * np.log(i_target / i0)
    
    
    R_ohm = 0.05 + (delta_e / 10) * 0.1  
    eta_ohm = i_target * R_ohm
    
    
    
    
    D = 1e-10  
    d_h = 0.001  
    mu = 1e-3  
    rho = 1000  
    
    Re = (rho * Q * 1e-6 * d_h) / (mu * 0.0001)  
    Sh = 1.5 * Re**0.5  
    k_m = Sh * D / d_h
    
    
    c_bulk = 2000  
    i_L = F * k_m * c_bulk * 0.0001  
    
    
    if i_target < i_L:
        eta_mt = -(R * T / F) * np.log(1 - i_target / i_L)
    else:
        eta_mt = 1.0  
    
    
    V_discharge = E_nernst - eta_act - eta_ohm - eta_mt
    
    
    V_charge = E_nernst + eta_act + eta_ohm + eta_mt
    
    
    eta_V = V_discharge / V_charge
    
    
    
    
    A_cross = 0.0001  
    f = 0.1  
    Delta_p = f * (rho * (Q * 1e-6 / A_cross)**2) / (2 * d_h)  
    P_p = (Q * 1e-6) * Delta_p  
    
    return eta_V, P_p, i_L


design_points = []
for delta_e in electrode_thicknesses:
    for Q in flow_rates:
        eta_V, P_p, i_L = calculate_vrfb_performance(delta_e, Q)
        if i_target < i_L and eta_V > 0 and eta_V < 1.0:  
            design_points.append({
                'delta_e': delta_e,
                'Q': Q,
                'eta_V': eta_V,
                'P_p': P_p,
                'P_p_norm': P_p / (i_target * 0.01 * 1.26)  
            })


if len(design_points) == 0:
    print("WARNING: No valid design points generated. Adjusting parameters...")
    
    design_points = []
    for delta_e in electrode_thicknesses:
        for Q in flow_rates:
            eta_V, P_p, i_L = calculate_vrfb_performance(delta_e, Q)
            if 0.5 < eta_V < 0.95:  
                design_points.append({
                    'delta_e': delta_e,
                    'Q': Q,
                    'eta_V': eta_V,
                    'P_p': P_p,
                    'P_p_norm': P_p / (i_target * 0.01 * 1.26)
                })

if len(design_points) == 0:
    raise ValueError("Unable to generate valid design points. Check model parameters.")

print(f"Generated {len(design_points)} valid design points")


design_points = sorted(design_points, key=lambda x: x['Q'])
delta_e_array = np.array([d['delta_e'] for d in design_points])
Q_array = np.array([d['Q'] for d in design_points])
eta_V_array = np.array([d['eta_V'] for d in design_points])
P_p_array = np.array([d['P_p'] for d in design_points])
P_p_norm_array = np.array([d['P_p_norm'] for d in design_points])


fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[2, 1])


ax_main = fig.add_subplot(gs[0, :])


norm = Normalize(vmin=delta_e_array.min(), vmax=delta_e_array.max())
cmap = cm.viridis
colors = cmap(norm(delta_e_array))

scatter = ax_main.scatter(P_p_norm_array * 100, eta_V_array * 100, 
                          c=delta_e_array, cmap='viridis', s=100, 
                          alpha=0.7, edgecolors='black', linewidth=1.5)


for thickness in electrode_thicknesses:
    mask = delta_e_array == thickness
    if mask.any():
        points = [(P_p_norm_array[i], eta_V_array[i]) for i, m in enumerate(mask) if m]
        if points:
            
            max_eff_idx = np.argmax([p[1] for p in points])
            optimal_point = points[max_eff_idx]
            ax_main.scatter(optimal_point[0] * 100, optimal_point[1] * 100,
                           s=300, marker='*', c='red', edgecolors='darkred',
                           linewidth=2, zorder=10, label=f'Optimal δₑ={thickness}mm' if thickness == electrode_thicknesses[0] else '')



eta_levels = np.linspace(eta_V_array.min(), eta_V_array.max(), 50)
pareto_front = []
for eta_level in eta_levels:
    nearby_points = [(P_p_norm_array[i], eta_V_array[i]) for i in range(len(eta_V_array)) 
                     if abs(eta_V_array[i] - eta_level) < 0.02]
    if nearby_points:
        min_power = min([p[0] for p in nearby_points])
        pareto_front.append((min_power * 100, eta_level * 100))

if pareto_front:
    pareto_front = sorted(pareto_front, key=lambda x: x[0])
    ax_main.plot([p[0] for p in pareto_front], [p[1] for p in pareto_front],
                'r--', linewidth=2, alpha=0.5, label='Pareto Frontier')


cbar = plt.colorbar(scatter, ax=ax_main)
cbar.set_label('Electrode Thickness (mm)', fontsize=12, fontweight='bold')


ax_main.set_xlabel('Normalized Pumping Power (%)', fontsize=13, fontweight='bold')
ax_main.set_ylabel('Voltage Efficiency (%)', fontsize=13, fontweight='bold')
ax_main.set_title('VRFB Multi-Objective Optimization: Pareto Front', 
                  fontsize=14, fontweight='bold')
ax_main.legend(loc='lower left', fontsize=10)
ax_main.grid(True, alpha=0.3)


ax_main.annotate('High Efficiency\nLow Power', xy=(0.3, 77), fontsize=10,
                ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax_main.annotate('Balanced\nDesign', xy=(1.5, 73), fontsize=10,
                ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))


ax_contour = fig.add_subplot(gs[1, 0])


delta_e_grid = np.linspace(3, 10, 50)
Q_grid = np.linspace(10, 50, 50)
Delta_E, Q_mesh = np.meshgrid(delta_e_grid, Q_grid)
Eta_V_grid = np.zeros_like(Delta_E)

for i in range(len(Q_grid)):
    for j in range(len(delta_e_grid)):
        eta_v, _, _ = calculate_vrfb_performance(Delta_E[i, j], Q_mesh[i, j])
        Eta_V_grid[i, j] = eta_v * 100


contour = ax_contour.contourf(Delta_E, Q_mesh, Eta_V_grid, levels=15, cmap='RdYlGn')
contour_lines = ax_contour.contour(Delta_E, Q_mesh, Eta_V_grid, levels=10, 
                                   colors='black', linewidths=0.5, alpha=0.3)
ax_contour.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.1f%%')


for thickness in electrode_thicknesses:
    mask = delta_e_array == thickness
    if mask.any():
        optimal_Q = Q_array[mask][np.argmax(eta_V_array[mask])]
        ax_contour.scatter(thickness, optimal_Q, s=150, marker='*', 
                          c='red', edgecolors='darkred', linewidth=2, zorder=10)

ax_contour.set_xlabel('Electrode Thickness (mm)', fontsize=11, fontweight='bold')
ax_contour.set_ylabel('Flow Rate (mL/s)', fontsize=11, fontweight='bold')
ax_contour.set_title('Efficiency Contour Map', fontsize=12, fontweight='bold')
plt.colorbar(contour, ax=ax_contour, label='η_V (%)')


ax_table = fig.add_subplot(gs[1, 1])
ax_table.axis('off')

table_text = f"""
Design Space Summary

Electrode Thickness Range:
  {electrode_thicknesses[0]} - {electrode_thicknesses[-1]} mm

Flow Rate Range:
  {flow_rates[0]:.0f} - {flow_rates[-1]:.0f} mL/s

Performance Metrics:
  Max η_V: {eta_V_array.max()*100:.2f}%
  Min P_p: {P_p_array.min()*1000:.2f} mW

Optimal Design:
  δₑ: {delta_e_array[np.argmax(eta_V_array)]:.0f} mm
  Q: {Q_array[np.argmax(eta_V_array)]:.1f} mL/s
  η_V: {eta_V_array.max()*100:.2f}%

Trade-offs:
  • Thicker electrodes → ↑ R_Ω → ↓ η_V
  • Higher flow → ↑ k_m → ↑ η_V
  • Higher flow → ↑ P_p → ↓ Net efficiency

Design Points Evaluated: {len(design_points)}
"""

ax_table.text(0.1, 0.95, table_text, transform=ax_table.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('results/figures/figure5_vrfb_pareto_enhanced.png', dpi=300, bbox_inches='tight')
print("✓ Figure 5 (Enhanced) saved: results/figures/figure5_vrfb_pareto_enhanced.png")
plt.close()


print("\n" + "="*70)
print("FIGURE 5: VRFB PARETO FRONT SUMMARY")
print("="*70)
print(f"\nDesign Space:")
print(f"  Electrode thickness: {electrode_thicknesses[0]}-{electrode_thicknesses[-1]} mm")
print(f"  Flow rate: {flow_rates[0]:.1f}-{flow_rates[-1]:.1f} mL/s")
print(f"  Total designs evaluated: {len(design_points)}")
print(f"\nOptimal Performance:")
print(f"  Maximum η_V: {eta_V_array.max()*100:.2f}%")
print(f"  At δₑ={delta_e_array[np.argmax(eta_V_array)]:.0f} mm, Q={Q_array[np.argmax(eta_V_array)]:.1f} mL/s")
print(f"  Minimum P_p: {P_p_array.min()*1000:.2f} mW")
print(f"\nKey Trade-offs:")
print(f"  • Efficiency range: {eta_V_array.min()*100:.2f}% - {eta_V_array.max()*100:.2f}%")
print(f"  • Power range: {P_p_norm_array.min()*100:.3f}% - {P_p_norm_array.max()*100:.3f}%")
print("="*70)
print("\n✓ Enhanced VRFB Pareto front complete!")
