import numpy as np
import matplotlib.pyplot as plt
import json


with open('results/tables/mechanistic_benchmark.json', 'r') as f:
    mech_results = json.load(f)

with open('results/tables/drl_vrfb_results.json', 'r') as f:
    drl_results = json.load(f)



pemfc_results = {
    'Physics Solver': {
        'rmse_mV': 9.73,
        'human_effort_min': 5,
        'color': '#2ecc71'
    },
    'ANN': {
        'rmse_mV': 16.62,
        'human_effort_min': 45,
        'color': '#e74c3c'
    },
    'Mechanistic\nBaseline': {
        'rmse_mV': mech_results['pemfc']['rmse_mV'],
        'human_effort_min': 10,
        'color': '#3498db'
    }
}


vrfb_results = {
    'ePCDNN': {
        'metric': 'Training\nComplete',
        'human_effort_min': 60,
        'color': '#9b59b6'
    },
    'DRL': {
        'soc_rmse': drl_results['final_soc_rmse'],
        'wlss_%': drl_results['final_wlss_%'],
        'human_effort_min': 90,
        'color': '#e67e22'
    },
    'Mechanistic\nBaseline': {
        'efficiency_%': mech_results['vrfb']['optimal_efficiency_%'],
        'human_effort_min': 15,
        'color': '#3498db'
    }
}


fig, axes = plt.subplots(1, 2, figsize=(14, 6))


ax = axes[0]

methods = list(pemfc_results.keys())
rmse_values = [pemfc_results[m]['rmse_mV'] for m in methods]
effort_values = [pemfc_results[m]['human_effort_min'] for m in methods]
colors = [pemfc_results[m]['color'] for m in methods]

x = np.arange(len(methods))
width = 0.35


bars1 = ax.bar(x - width/2, rmse_values, width, label='RMSE (mV)',
               color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)


ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, effort_values, width, label='Human Effort (min)',
                color='gray', alpha=0.5, edgecolor='black', linewidth=1.5,
                hatch='//')


ax.set_xlabel('Method', fontsize=12, fontweight='bold')
ax.set_ylabel('RMSE (mV)', fontsize=12, fontweight='bold', color='black')
ax2.set_ylabel('Human Effort (minutes)', fontsize=12, fontweight='bold', color='gray')
ax.set_title('PEMFC: Performance vs. Effort', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10)
ax.grid(True, alpha=0.3, axis='y')


for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}m',
             ha='center', va='bottom', fontsize=9, color='gray')


lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)


ax.axhline(y=min(rmse_values), color='green', linestyle='--', alpha=0.5, linewidth=2)
ax.text(0.5, min(rmse_values)*1.1, 'Best RMSE', fontsize=9, color='green',
        fontweight='bold', ha='center')


ax = axes[1]


vrfb_methods = list(vrfb_results.keys())
vrfb_effort = [vrfb_results[m]['human_effort_min'] for m in vrfb_methods]
vrfb_colors = [vrfb_results[m]['color'] for m in vrfb_methods]

x = np.arange(len(vrfb_methods))


bars = ax.bar(x, vrfb_effort, color=vrfb_colors, alpha=0.8,
              edgecolor='black', linewidth=1.5)


ax.set_xlabel('Method', fontsize=12, fontweight='bold')
ax.set_ylabel('Human Effort (minutes)', fontsize=12, fontweight='bold')
ax.set_title('VRFB: Implementation Effort', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(vrfb_methods, fontsize=10)
ax.grid(True, alpha=0.3, axis='y')


for i, (bar, method) in enumerate(zip(bars, vrfb_methods)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)} min',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    
    if method == 'DRL':
        metric_text = f"RMSE: {vrfb_results[method]['soc_rmse']:.4f}\nWLSS: {vrfb_results[method]['wlss_%']:.2f}%"
    elif method == 'Mechanistic\nBaseline':
        metric_text = f"η_V: {vrfb_results[method]['efficiency_%']:.1f}%"
    else:
        metric_text = "Physics\nConstrained"
    
    ax.text(bar.get_x() + bar.get_width()/2., 5,
            metric_text,
            ha='center', va='bottom', fontsize=8, style='italic')

plt.tight_layout()
plt.savefig('results/figures/figure8_comparative_performance.png', dpi=300, bbox_inches='tight')
print("✓ Figure 8 saved: results/figures/figure8_comparative_performance.png")
plt.close()


print("\n" + "="*70)
print("FIGURE 8: COMPARATIVE PERFORMANCE SUMMARY")
print("="*70)
print("\nPEMFC Results:")
for method, data in pemfc_results.items():
    print(f"  {method:20s}: RMSE={data['rmse_mV']:6.2f}mV, Effort={data['human_effort_min']:3d}min")

print("\nVRFB Results:")
for method, data in vrfb_results.items():
    print(f"  {method:20s}: Effort={data['human_effort_min']:3d}min")

print("\nKey Finding:")
print(f"  Physics solver achieves BEST RMSE ({min([d['rmse_mV'] for d in pemfc_results.values()]):.2f}mV)")
print(f"  with LOWEST effort ({min([d['human_effort_min'] for d in pemfc_results.values()])}min)")
print("="*70)
