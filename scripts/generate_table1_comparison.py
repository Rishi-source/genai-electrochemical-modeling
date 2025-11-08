"""
Generate Table I: Cross-Method Comparison
Comprehensive comparison table for all baseline methods.
"""

import json
import pandas as pd

# Load results
with open('results/tables/mechanistic_benchmark.json', 'r') as f:
    mech_results = json.load(f)

with open('results/tables/drl_vrfb_results.json', 'r') as f:
    drl_results = json.load(f)

# Define comparison table
table_data = {
    'Approach': [
        'Mechanistic Solver',
        'ANN (PEMFC) [2]',
        'ePCDNN (VRFB) [1]',
        'DRL (VRFB) [3]',
        'Digital Twin [5]',
        'LLM Multi-Agent [14]',
        'Proposed LLM-RAG-Physics'
    ],
    'Role': [
        'Ground-truth modeling',
        'Surrogate for V(i)',
        'Physics-informed surrogate',
        'Parameter tuning/control',
        'Plant-wide prediction',
        'Orchestration',
        'Process-calc co-pilot'
    ],
    'Physics Use': [
        'Explicit PDE/DAE',
        'Implicit',
        'Hard/soft constraints',
        'Via simulator',
        'Data-driven',
        'Constraint prompts',
        'Explicit checks + tools'
    ],
    'Selected Metrics': [
        f"PEMFC: R²={mech_results['pemfc']['r_squared']:.4f}, RMSE={mech_results['pemfc']['rmse_mV']:.1f}mV\nVRFB: η_V={mech_results['vrfb']['optimal_efficiency_%']:.1f}%",
        'R²=0.9757, RMSE=16.62mV',
        '~30% error reduction vs. PCDNN',
        f"SOC RMSE={drl_results['final_soc_rmse']:.4f}, WLSS={drl_results['final_wlss_%']:.2f}%",
        'R² up to 0.96 (KNN regression)',
        '3× fewer iterations vs. IPOPT',
        'TBD (Framework ready)'
    ],
    'Interpretability': [
        'High (physical params)',
        'Medium (SHAP)',
        'Med-High',
        'Low-Med',
        'Medium',
        'Medium (rationales)',
        'Med-High (reports)'
    ],
    'Latency': [
        'Med-High',
        'Low (inference)',
        'Medium (training)',
        'High (training)',
        'Low',
        'Low-Med',
        'Low-Med'
    ],
    'Human Effort': [
        'High (setup)',
        'Medium',
        'Med-High',
        'Medium',
        'Medium',
        'Low',
        'Low'
    ]
}

# Create DataFrame
df = pd.DataFrame(table_data)

# Save as CSV
csv_path = 'results/tables/table1_cross_method_comparison.csv'
df.to_csv(csv_path, index=False)
print(f"✓ Table I saved as CSV: {csv_path}")

# Save as LaTeX
latex_path = 'results/tables/table1_cross_method_comparison.tex'
latex_table = df.to_latex(
    index=False,
    column_format='|l|l|l|p{4cm}|l|l|l|',
    caption='Cross-Method Comparison',
    label='tab:cross_method_comparison',
    escape=False
)
with open(latex_path, 'w') as f:
    f.write(latex_table)
print(f"✓ Table I saved as LaTeX: {latex_path}")

# Print formatted table
print("\n" + "="*100)
print("TABLE I: CROSS-METHOD COMPARISON")
print("="*100)
print(df.to_string(index=False))
print("="*100)

# Print summary statistics
print("\n📊 Key Insights:")
print(f"  • Mechanistic solver: Highest interpretability, explicit physics")
print(f"  • ANN: Fast inference, requires training data")
print(f"  • ePCDNN: Physics-constrained, 30% improvement over baseline")
print(f"  • DRL: Sample efficient, SOC RMSE={drl_results['final_soc_rmse']:.4f}")
print(f"  • Proposed framework: Low human effort, physics-aware")

print("\n✓ Table I generation complete!")
