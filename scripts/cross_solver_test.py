import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_orchestrator.self_correcting_agent import SelfCorrectingAgent
from src.physics_validator.validator import PhysicsValidator

def run_cross_solver_test():
    print("="*70)
    print("CROSS-SOLVER GENERATION TEST")
    print("="*70)
    
    agent = SelfCorrectingAgent(verbose=False)
    validator = PhysicsValidator(verbose=True)
    
    backends = [
        ("Python", "Generate Python code using scipy.optimize.least_squares to fit PEMFC data."),
        ("MATLAB", "Generate MATLAB code using lsqnonlin to fit PEMFC data.")
    ]
    
    for lang, prompt in backends:
        print(f"\nTesting Backend: {lang}")
        print(f"Prompt: {prompt}")
        
        result = agent.generate_validated_code(
            user_query=prompt,
            task_type="pemfc_fitting",
            use_rag=True
        )
        
        if result.success:
            print(f"✓ {lang} Code Generated Successfully")
            print(f"  Length: {len(result.final_code)} chars")
            
            # Extra validation pass (simulated execution check would go here)
            val_res = validator.validate(result.final_code)
            if val_res.is_valid:
                print(f"  Validator: Valid {lang} syntax/physics")
            else:
                print(f"  Validator: Issues found ({len(val_res.violations)} violations)")
        else:
            print(f"✗ Failed to generate {lang} code")

if __name__ == "__main__":
    run_cross_solver_test()
