import json
import numpy as np
import matplotlib.pyplot as plt

with open('results/tables/drl_vrfb_results.json', 'r') as f:
    results = json.load(f)

print("DRL Results Loaded:")
print(f"  SOC RMSE: {results['final_soc_rmse']:.4f}")
print(f"  WLSS: {results['final_wlss_%']:.2f}%")
print(f"  Training time: {results['training_time_s']:.1f}s")
print(f"  Total steps: {results['total_steps']}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
metrics = ['SOC RMSE', 'WLSS (%)']
values = [results['final_soc_rmse'], results['final_wlss_%']]
colors = ['#3498db', '#e74c3c']

bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Value', fontsize=12, fontweight='bold')
ax.set_title('DRL Final Performance Metrics', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.4f}' if value < 1 else f'{value:.2f}',
            ha='center', va='bottom', fontweight='bold')

ax = axes[1]
ax.axis('off')

info_text = f"""
Dueling DQN Training Summary

Architecture:
• Dueling DQN (Value + Advantage)
• State dim: 6 (SOC, V, i, T, Q, δₑ)
• Action dim: 5 (parameter adjustments)

Training Configuration:
• Episodes: 500
• Buffer size: 100,000
• Batch size: 64
• Epsilon decay: 10,000 steps
• Gamma (discount): 0.99

Results:
• Total steps: {results['total_steps']:,}
• Training time: {results['training_time_s']:.1f}s
• Throughput: {results['total_steps']/results['training_time_s']:.0f} steps/s

Optimized Parameters:
• i₀: {results['best_parameters']['i0']:.2e} A/cm²
• α: {results['best_parameters']['alpha']:.3f}

Performance:
• Best SOC RMSE: {results['final_soc_rmse']:.4f}
• Final WLSS: {results['final_wlss_%']:.2f}%
"""

ax.text(0.1, 0.95, info_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('results/figures/drl_vrfb_summary.png', dpi=300, bbox_inches='tight')
print("\n✓ Figure saved: results/figures/drl_vrfb_summary.png")
plt.close()

print("\n✓ DRL visualization complete!")
