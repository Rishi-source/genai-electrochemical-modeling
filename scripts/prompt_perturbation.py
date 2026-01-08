import os
import sys
import time
from typing import List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_orchestrator.self_correcting_agent import SelfCorrectingAgent

def run_prompt_perturbation():
    print("="*70)
    print("PROMPT PERTURBATION TEST")
    print("="*70)
    
    agent = SelfCorrectingAgent(verbose=False)
    
    # Semantic variations of the same task
    prompts = [
        "Fit PEMFC polarization data at 60C with Pt/C cathode parameters.",
        "Generate MATLAB code to fit PEMFC i-V curve at 60 degrees Celsius for Platinum/Carbon catalyst.",
        "I need a script to extract parameters i0, alpha, R_ohm from PEMFC experimental data at 333K.",
        "Create a parameter estimation routine for Proton Exchange Membrane Fuel Cell operating at 60C.",
        "Perform regression analysis on PEMFC voltage-current data to find kinetic and ohmic parameters (T=60C)."
    ]
    
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: '{prompt}'")
        result = agent.generate_validated_code(
            user_query=prompt,
            task_type="pemfc_fitting",
            use_rag=True
        )
        
        status = "✓ Success" if result.success else "✗ Failed"
        print(f"Status: {status} (Attempts: {result.total_attempts})")
        results.append(result)

    success_count = sum(1 for r in results if r.success)
    print("\n" + "-"*30)
    print("RESULTS SUMMARY")
    print("-"*30)
    print(f"Robustness Score: {success_count}/{len(prompts)} ({success_count/len(prompts)*100:.1f}%)")

if __name__ == "__main__":
    run_prompt_perturbation()
