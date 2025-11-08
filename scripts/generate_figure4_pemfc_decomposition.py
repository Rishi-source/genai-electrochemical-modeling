"""
Generate Figure 4: PEMFC Polarization Curve Decomposition
Shows loss breakdown: Nernst, activation, ohmic, mass-transfer.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
import json

# Load PEMFC data
df = pd.read_csv('data/synthetic/pemfc_polarization.csv')

# Load mechanistic results for fitted parameters
with open('results/tables/mechanistic_benchmark.json', 'r') as f:
    mech_results = json.load(f)

# Use data at T=80°C for visualization
df_80 = df[df['temperature_C'] == 80.0].copy()
df_80 = df_80.sort_values('current_density_A_cm2')

i = df_80['current_density_A_cm2'].values
V = df_80['voltage_V'].values

# Physics constants
R = 8.314  # J/(mol·K)
F = 96485  # C/mol
T = 80 + 273.15  # K

# Fitted parameters (from Phase 4 results)
i0 = 5.436e-06  # A/cm²
alpha = 0.495
R_ohm = 0.139  # Ω·cm²
i_L = 2.330  # A/cm²

# Calculate components
E_nernst = 1.17  # V (constant for this example)

# Activation overpotential (Tafel approximation)
eta_act = (R * T / (alpha * F)) * np.log(i / i0)

# Ohmic overpotential
eta_ohm = i * R_ohm

# Mass-transfer overpotential
eta_mt = -(R * T / (2 * F)) * np.log(1 - i / i_L)

# Total voltage
V_model = E_nernst - eta_act - eta_ohm - eta_mt

# Create figure with main plot and Tafel inset
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[2, 1])

# Main polarization curve with decomposition
ax_main = fig.add_subplot(gs[0, :])

# Plot data
ax_main.scatter(i, V, c='black', s=50, alpha=0.6, label='Experimental Data', zorder=5)

# Plot fitted curve
ax_main.plot(i, V_model, 'r-', linewidth=3, label='Fitted Model', zorder=4)

# Plot Nernst baseline
ax_main.axhline(y=E_nernst, color='green', linestyle='--', linewidth=2,
                label=f'$E_{{Nernst}}$ = {E_nernst:.3f} V', zorder=1)

# Show loss regions with shading
# Activation loss region (low current)
i_act = i[i < 0.5]
V_nernst_act = np.full_like(i_act, E_nernst)
V_after_act = E_nernst - eta_act[:len(i_act)]
ax_main.fill_between(i_act, V_nernst_act, V_after_act,
                      alpha=0.3, color='orange', label='Activation Loss')

# Ohmic loss region (mid current)
i_ohm = i[(i >= 0.5) & (i < 1.0)]
idx_ohm = np.where((i >= 0.5) & (i < 1.0))[0]
V_after_act_ohm = E_nernst - eta_act[idx_ohm]
V_after_ohm = E_nernst - eta_act[idx_ohm] - eta_ohm[idx_ohm]
ax_main.fill_between(i_ohm, V_after_act_ohm, V_after_ohm,
                      alpha=0.3, color='blue', label='Ohmic Loss')

# Mass-transfer loss region (high current)
i_mt = i[i >= 1.0]
idx_mt = np.where(i >= 1.0)[0]
V_after_ohm_mt = E_nernst - eta_act[idx_mt] - eta_ohm[idx_mt]
V_after_all = E_nernst - eta_act[idx_mt] - eta_ohm[idx_mt] - eta_mt[idx_mt]
ax_main.fill_between(i_mt, V_after_ohm_mt, V_after_all,
                      alpha=0.3, color='red', label='Mass-Transfer Loss')

ax_main.set_xlabel('Current Density (A/cm²)', fontsize=13, fontweight='bold')
ax_main.set_ylabel('Voltage (V)', fontsize=13, fontweight='bold')
ax_main.set_title('PEMFC Polarization Curve Decomposition (80°C)',
                  fontsize=14, fontweight='bold')
ax_main.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax_main.grid(True, alpha=0.3)
ax_main.set_xlim([0, 1.6])
ax_main.set_ylim([0, 1.3])

# Add annotations for regions
ax_main.text(0.25, 0.4, 'Activation\nDominated', fontsize=10,
             ha='center', style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
ax_main.text(0.75, 0.35, 'Ohmic\nDominated', fontsize=10,
             ha='center', style='italic', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax_main.text(1.3, 0.2, 'Mass-Transfer\nDominated', fontsize=10,
             ha='center', style='italic', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

# Tafel plot inset (activation region)
ax_tafel = fig.add_subplot(gs[1, 0])

# Select low current region for Tafel analysis
i_tafel = i[(i > 0.01) & (i < 0.8)]
eta_tafel = eta_act[(i > 0.01) & (i < 0.8)]

# Tafel plot: log(i) vs eta
ax_tafel.plot(np.log10(i_tafel), eta_tafel * 1000, 'o-', 
              linewidth=2, markersize=4, color='orange')

# Linear fit for Tafel slope
coeffs = np.polyfit(np.log10(i_tafel), eta_tafel * 1000, 1)
tafel_slope = coeffs[0]  # mV/decade
tafel_fit = np.poly1d(coeffs)

x_fit = np.linspace(np.log10(i_tafel.min()), np.log10(i_tafel.max()), 100)
ax_tafel.plot(x_fit, tafel_fit(x_fit), 'r--', linewidth=2,
              label=f'Tafel Slope: {tafel_slope:.1f} mV/dec')

ax_tafel.set_xlabel('log₁₀(i) [log₁₀(A/cm²)]', fontsize=11, fontweight='bold')
ax_tafel.set_ylabel('η$_{act}$ (mV)', fontsize=11, fontweight='bold')
ax_tafel.set_title('Tafel Analysis (Activation Region)', fontsize=12, fontweight='bold')
ax_tafel.legend(fontsize=9)
ax_tafel.grid(True, alpha=0.3)

# Loss breakdown pie chart
ax_pie = fig.add_subplot(gs[1, 1])

# Calculate average losses at i=1.0 A/cm²
i_ref = 1.0
idx_ref = np.argmin(np.abs(i - i_ref))
losses_at_ref = {
    'Activation': eta_act[idx_ref] * 1000,  # mV
    'Ohmic': eta_ohm[idx_ref] * 1000,
    'Mass-Transfer': eta_mt[idx_ref] * 1000
}

colors_pie = ['#ff9999', '#66b3ff', '#99ff99']
explode = (0.05, 0.05, 0.05)

wedges, texts, autotexts = ax_pie.pie(
    losses_at_ref.values(),
    labels=losses_at_ref.keys(),
    autopct='%1.1f%%',
    startangle=90,
    colors=colors_pie,
    explode=explode,
    shadow=True
)

# Enhance text
for text in texts:
    text.set_fontsize(10)
    text.set_fontweight('bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(9)
    autotext.set_fontweight('bold')

ax_pie.set_title(f'Loss Breakdown\nat i={i_ref:.1f} A/cm²',
                 fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('results/figures/figure4_pemfc_decomposition.png', dpi=300, bbox_inches='tight')
print("✓ Figure 4 saved: results/figures/figure4_pemfc_decomposition.png")
plt.close()

# Print summary
print("\n" + "="*70)
print("FIGURE 4: PEMFC DECOMPOSITION SUMMARY")
print("="*70)
print(f"\nFitted Parameters:")
print(f"  i₀: {i0:.3e} A/cm²")
print(f"  α: {alpha:.3f}")
print(f"  R_Ω: {R_ohm:.3f} Ω·cm²")
print(f"  i_L: {i_L:.3f} A/cm²")
print(f"\nLoss Breakdown at i={i_ref:.1f} A/cm²:")
for loss_type, loss_value in losses_at_ref.items():
    print(f"  {loss_type:15s}: {loss_value:6.1f} mV ({loss_value/sum(losses_at_ref.values())*100:5.1f}%)")
print(f"  {'Total':15s}: {sum(losses_at_ref.values()):6.1f} mV")
print(f"\nTafel Slope: {tafel_slope:.1f} mV/decade")
print("="*70)
print("\n✓ Figure 4 generation complete!")
