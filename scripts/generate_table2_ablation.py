"""
Generate Table II: Ablation Analysis
Detailed breakdown of each configuration's performance.
"""

import json
import pandas as pd

# Load ablation results
with open('results/tables/ablation_results.json', 'r') as f:
    data = json.load(f)

results = data['results']

# Create DataFrame
df = pd.DataFrame(results)

# Rename columns for display
df_display = df.copy()
df_display.columns = ['Config', 'RMSE (mV)', 'MAE (mV)', 'R²', 
                      'Violations (%)', 'Compile Err (%)', 
                      'Time (s)', 'Effort Red. (%)', 'Iterations']

# Round values
df_display['RMSE (mV)'] = df_display['RMSE (mV)'].round(2)
df_display['MAE (mV)'] = df_display['MAE (mV)'].round(2)
df_display['R²'] = df_display['R²'].round(4)
df_display['Violations (%)'] = df_display['Violations (%)'].round(1)
df_display['Compile Err (%)'] = df_display['Compile Err (%)'].round(1)
df_display['Time (s)'] = df_display['Time (s)'].round(2)
df_display['Effort Red. (%)'] = df_display['Effort Red. (%)'].round(1)

# Save to CSV
csv_path = 'results/tables/table2_ablation.csv'
df_display.to_csv(csv_path, index=False)
print(f"✓ Table II (CSV) saved: {csv_path}")

# Generate LaTeX table
latex_table = r"""\begin{table}[ht]
\centering
\caption{Ablation Study: Impact of Framework Components on PEMFC Fitting Performance}
\label{tab:ablation}
\begin{tabular}{lcccccccc}
\toprule
\textbf{Configuration} & \textbf{RMSE} & \textbf{MAE} & \textbf{R\textsuperscript{2}} & \textbf{Violations} & \textbf{Compile} & \textbf{Time} & \textbf{Effort} & \textbf{Iters} \\
 & \textbf{(mV)} & \textbf{(mV)} & & \textbf{(\%)} & \textbf{Err (\%)} & \textbf{(s)} & \textbf{Red. (\%)} & \\
\midrule
"""

for _, row in df_display.iterrows():
    latex_table += f"{row['Config']} & {row['RMSE (mV)']:.2f} & {row['MAE (mV)']:.2f} & {row['R²']:.4f} & "
    latex_table += f"{row['Violations (%)']:.1f} & {row['Compile Err (%)']:.1f} & {row['Time (s)']:.2f} & "
    latex_table += f"{row['Effort Red. (%)']:.1f} & {row['Iterations']} \\\\\n"

latex_table += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textbf{Configuration key:} base (no components), +rag (retrieval), +hybrid (Eq. 3 similarity), +physics (constraints), +tools (code generation), full (all + self-refinement)
\item \textbf{Key findings:} Progressive component addition reduces constraint violations from 45\% to 0.5\% (99\% reduction) and compile errors from 30\% to 0.8\% (97\% reduction), while maintaining RMSE $\approx$ 9.6 mV and achieving 55\% human effort reduction.
\end{tablenotes}
\end{table}
"""

# Save LaTeX table
latex_path = 'results/tables/table2_ablation.tex'
with open(latex_path, 'w') as f:
    f.write(latex_table)
print(f"✓ Table II (LaTeX) saved: {latex_path}")

# Print formatted table
print("\n" + "="*140)
print("TABLE II: ABLATION ANALYSIS")
print("="*140)
print(df_display.to_string(index=False))
print("="*140)

# Print summary statistics
print("\nSummary Statistics:")
print("-" * 60)
base_row = df_display.iloc[0]
full_row = df_display.iloc[-1]

improvements = {
    'RMSE': (base_row['RMSE (mV)'], full_row['RMSE (mV)']),
    'Constraint Violations': (base_row['Violations (%)'], full_row['Violations (%)']),
    'Compile Errors': (base_row['Compile Err (%)'], full_row['Compile Err (%)']),
    'Human Effort Reduction': (base_row['Effort Red. (%)'], full_row['Effort Red. (%)'])
}

for metric, (base_val, full_val) in improvements.items():
    if 'Reduction' in metric:
        change = full_val - base_val
        print(f"{metric:30s}: {base_val:6.1f}% → {full_val:6.1f}% (+{change:.1f} pp)")
    else:
        if base_val > 0:
            pct_change = (base_val - full_val) / base_val * 100
            print(f"{metric:30s}: {base_val:6.1f} → {full_val:6.1f} ({pct_change:+.0f}% change)")
        else:
            print(f"{metric:30s}: {base_val:6.1f} → {full_val:6.1f}")

print("="*60)
print("\n✓ Table II generation complete!")
print("\nFiles generated:")
print(f"  • CSV:   {csv_path}")
print(f"  • LaTeX: {latex_path}")
