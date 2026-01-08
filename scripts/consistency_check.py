import os
import sys
import numpy as np
import difflib
import time
from typing import List, Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_orchestrator.self_correcting_agent import SelfCorrectingAgent

def calculate_similarity(code1: str, code2: str) -> float:
    """Calculate similarity ratio between two code strings using SequenceMatcher."""
    return difflib.SequenceMatcher(None, code1, code2).ratio()

def run_consistency_check(n_iterations: int = 10):
    print("="*70)
    print(f"CONSISTENCY CHECK: Running {n_iterations} iterations")
    print("="*70)
    
    agent = SelfCorrectingAgent(verbose=False) # Reduce verbosity for batch run
    
    # Standardized prompt
    query = "Fit PEMFC polarization data at 60C with Pt/C cathode parameters. Output MATLAB code using lsqnonlin."
    
    results = []
    codes = []
    
    start_time = time.time()
    
    for i in range(n_iterations):
        print(f"Iteration {i+1}/{n_iterations}...", end="", flush=True)
        result = agent.generate_validated_code(
            user_query=query,
            task_type="pemfc_fitting",
            use_rag=True # Use RAG to test full pipeline consistency
        )
        
        results.append(result)
        if result.success:
            codes.append(result.final_code)
            print(" Success")
        else:
            print(" Failed")

    total_time = time.time() - start_time
    
    # Metrics
    success_rate = len(codes) / n_iterations
    
    similarity_scores = []
    if len(codes) > 1:
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                sim = calculate_similarity(codes[i], codes[j])
                similarity_scores.append(sim)
        avg_similarity = np.mean(similarity_scores)
    else:
        avg_similarity = 1.0 if len(codes) == 1 else 0.0

    print("\n" + "-"*30)
    print("RESULTS SUMMARY")
    print("-"*30)
    print(f"Total Time: {total_time:.2f}s")
    print(f"Success Rate: {success_rate*100:.1f}% ({len(codes)}/{n_iterations})")
    print(f"Avg Code Similarity: {avg_similarity:.4f}")
    
    # Save samples
    output_dir = "results/consistency_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, code in enumerate(codes):
        with open(os.path.join(output_dir, f"sample_{idx+1}.m"), "w") as f:
            f.write(code)
            
    print(f"\nGenerated samples saved to {output_dir}")

if __name__ == "__main__":
    run_consistency_check()
