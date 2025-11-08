import json
import numpy as np
import matplotlib.pyplot as plt


with open('results/tables/ablation_results.json', 'r') as f:
    data = json.load(f)

results = data['results']
configs = [r['config'] for r in results]
rmse_values = [r['rmse_mV'] for r in results]
violations = [r['constraint_violation_%'] for r in results]
compile_errors = [r['compile_error_%'] for r in results]
effort_reduction = [r['human_effort_reduction_%'] for r in results]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))


x_pos = np.arange(len(configs))
width = 0.6


bars = ax1.bar(x_pos, violations, width, 
               label='Constraint Violations', 
               color='


for i, (bar, val) in enumerate(zip(bars, violations)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')


ax1_twin = ax1.twinx()
line = ax1_twin.plot(x_pos, rmse_values, 'o-', 
                     label='RMSE', color='
                     linewidth=3, markersize=10, markeredgecolor='black', markeredgewidth=1.5)


ax1.set_xlabel('Configuration', fontsize=13, fontweight='bold')
ax1.set_ylabel('Constraint Violation Rate (%)', fontsize=12, fontweight='bold', color='
ax1_twin.set_ylabel('RMSE (mV)', fontsize=12, fontweight='bold', color='
ax1.set_title('Ablation Study: Progressive Component Impact\n(Constraint Violations & RMSE)', 
             fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(configs, fontsize=11, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='
ax1_twin.tick_params(axis='y', labelcolor='
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, max(violations) * 1.15)
ax1_twin.set_ylim(0, max(rmse_values) * 1.2)


lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)



bars2 = ax2.bar(x_pos, compile_errors, width,
               label='Compile Errors', 
               color='


for i, (bar, val) in enumerate(zip(bars2, compile_errors)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')


ax2_twin = ax2.twinx()
line2 = ax2_twin.plot(x_pos, effort_reduction, 's-',
                     label='Human Effort Reduction', color='
                     linewidth=3, markersize=10, markeredgecolor='black', markeredgewidth=1.5)


ax2.set_xlabel('Configuration', fontsize=13, fontweight='bold')
ax2.set_ylabel('Compile Error Rate (%)', fontsize=12, fontweight='bold', color='
ax2_twin.set_ylabel('Human Effort Reduction (%)', fontsize=12, fontweight='bold', color='
ax2.set_title('Ablation Study: System Reliability & Efficiency\n(Compile Errors & Effort Reduction)', 
             fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(configs, fontsize=11, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='
ax2_twin.tick_params(axis='y', labelcolor='
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, max(compile_errors) * 1.15)
ax2_twin.set_ylim(0, max(effort_reduction) * 1.2)


lines3, labels3 = ax2.get_legend_handles_labels()
lines4, labels4 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines3 + lines4, labels3 + labels4, loc='upper left', fontsize=11)


summary_text = f"""Key Improvements (base → full):
• Violations: {violations[0]:.1f}% → {violations[-1]:.1f}% ({(violations[0]-violations[-1])/violations[0]*100:.0f}% ↓)
• Compile errors: {compile_errors[0]:.1f}% → {compile_errors[-1]:.1f}% ({(compile_errors[0]-compile_errors[-1])/compile_errors[0]*100:.0f}% ↓)
• Human effort: +{effort_reduction[-1]:.0f}% reduction
"""

fig.text(0.5, -0.05, summary_text, ha='center', fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8), 
         family='monospace')

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig('results/figures/figure9_ablation_study.png', dpi=300, bbox_inches='tight')
print("✓ Figure 9 saved: results/figures/figure9_ablation_study.png")
plt.close()


print("\n" + "="*70)
print("FIGURE 9: ABLATION STUDY SUMMARY")
print("="*70)
print(f"\nConfigurations tested: {len(configs)}")
print(f"\nProgressive improvements:")
for i, config in enumerate(configs):
    print(f"  {config:12s}: violations={violations[i]:5.1f}%, "
          f"errors={compile_errors[i]:5.1f}%, effort={effort_reduction[i]:4.0f}%")

print(f"\nOverall impact (base → full):")
print(f"  Constraint violations: {violations[0]:.1f}% → {violations[-1]:.1f}% "
      f"({(violations[0]-violations[-1])/violations[0]*100:.0f}% reduction)")
print(f"  Compile errors: {compile_errors[0]:.1f}% → {compile_errors[-1]:.1f}% "
      f"({(compile_errors[0]-compile_errors[-1])/compile_errors[0]*100:.0f}% reduction)")
print(f"  Human effort: 0% → {effort_reduction[-1]:.0f}% reduction")
print("="*70)
print("\n✓ Figure 9 generation complete!")
