import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.llm_orchestrator import SelfCorrectingAgent
from src.rag import ChromaManager


def demo_pemfc_closed_loop():
    print("\n" + "="*80)
    print("DEMO 1: PEMFC Polarization Curve Fitting - Closed Loop Self-Correction")
    print("="*80)
    
    agent = SelfCorrectingAgent(
        rag_manager=None,
        max_iterations=3,
        verbose=True
    )
    
    query = """Generate Python code to fit PEMFC polarization data using:
    - Nernst equation for equilibrium potential
    - Butler-Volmer kinetics for activation losses
    - Ohmic resistance for IR drop
    - Mass transport limitations
    
    Fit parameters: i0, alpha, R_ohm, i_L
    Return fitted parameters and RMSE"""
    
    result = agent.generate_validated_code(
        user_query=query,
        task_type="pemfc_fitting",
        use_rag=False,
        temperature=80,
        p_h2=1.0,
        p_o2=0.21
    )
    
    print("\n" + "-"*80)
    print("RESULTS:")
    print("-"*80)
    summary = result.get_summary()
    
    print(f"Success: {summary['success']}")
    print(f"Total Attempts: {summary['total_attempts']}")
    print(f"Total Time: {summary['total_time']:.2f}s")
    print(f"Total Tokens: {summary['total_tokens']}")
    print(f"Violation History: {summary['violation_history']}")
    
    if result.success:
        print("\n✓ VALID CODE GENERATED:")
        print("-"*80)
        print(result.final_code)
        print("-"*80)
    else:
        print("\n⚠ Failed to generate valid code")
        print("Last attempt violations:")
        for v in result.attempts[-1].validation_result.violations:
            print(f"  - {v}")
    
    return result


def demo_vrfb_closed_loop():
    print("\n" + "="*80)
    print("DEMO 2: VRFB Design Optimization - Closed Loop Self-Correction")
    print("="*80)
    
    agent = SelfCorrectingAgent(
        rag_manager=None,
        max_iterations=3,
        verbose=True
    )
    
    query = """Generate Python code to optimize VRFB design:
    - Design variables: electrode thickness (delta_e), flow rate (Q)
    - Objective: maximize voltage efficiency - beta * pumping_power
    - Constraints: i < i_L, pressure_drop < max_pressure
    
    Use Nernst equation, Butler-Volmer kinetics, and Sherwood correlation
    for mass transfer. Return optimal design parameters."""
    
    result = agent.generate_validated_code(
        user_query=query,
        task_type="vrfb_optimization",
        use_rag=False,
        current_density=200,
        temperature=25
    )
    
    print("\n" + "-"*80)
    print("RESULTS:")
    print("-"*80)
    summary = result.get_summary()
    
    print(f"Success: {summary['success']}")
    print(f"Total Attempts: {summary['total_attempts']}")
    print(f"Total Time: {summary['total_time']:.2f}s")
    print(f"Total Tokens: {summary['total_tokens']}")
    print(f"Violation History: {summary['violation_history']}")
    
    if result.success:
        print("\n✓ VALID CODE GENERATED:")
        print("-"*80)
        print(result.final_code)
        print("-"*80)
    else:
        print("\n⚠ Failed to generate valid code")
    
    return result


def demo_batch_generation():
    print("\n" + "="*80)
    print("DEMO 3: Batch Code Generation with Self-Correction")
    print("="*80)
    
    agent = SelfCorrectingAgent(
        rag_manager=None,
        max_iterations=3,
        verbose=True
    )
    
    queries = [
        "Calculate Nernst potential for hydrogen-air PEMFC at 80°C",
        "Compute activation overpotential using Tafel equation",
        "Calculate ohmic resistance from membrane conductivity"
    ]
    
    results = agent.batch_generate(
        queries=queries,
        task_type="equation_derivation",
        use_rag=False
    )
    
    print("\n" + "-"*80)
    print("BATCH RESULTS:")
    print("-"*80)
    
    stats = agent.get_statistics(results)
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Success Rate: {stats['success_rate']*100:.1f}%")
    print(f"Average Attempts: {stats['average_attempts']:.2f}")
    print(f"First-Attempt Success Rate: {stats['first_attempt_success']*100:.1f}%")
    print(f"Total Time: {stats['total_time']:.2f}s")
    print(f"Total Tokens: {stats['total_tokens']}")
    
    print("\nIndividual Results:")
    for i, (query, result) in enumerate(zip(queries, results)):
        status = "✓" if result.success else "✗"
        print(f"  {status} Query {i+1}: {result.total_attempts} attempts")
    
    return results


def demo_with_rag():
    print("\n" + "="*80)
    print("DEMO 4: Closed Loop with RAG (if ChromaDB available)")
    print("="*80)
    
    try:
        rag_manager = ChromaManager()
        
        if rag_manager.collection.count() == 0:
            print("⚠ No documents in ChromaDB. Adding sample document...")
            rag_manager.add_document(
                doc_id="sample_1",
                text="PEMFC with Pt/C catalyst: i0 = 1e-7 A/cm², alpha = 0.5, T = 80°C",
                metadata={
                    "system": "PEMFC",
                    "catalyst": "Pt/C",
                    "temperature_C": 80,
                    "i0_A_cm2": 1e-7,
                    "alpha": 0.5
                }
            )
        
        agent = SelfCorrectingAgent(
            rag_manager=rag_manager,
            max_iterations=3,
            verbose=True
        )
        
        query = "Generate PEMFC voltage calculation code using Pt/C catalyst parameters"
        
        result = agent.generate_validated_code(
            user_query=query,
            task_type="pemfc_fitting",
            use_rag=True,
            target_conditions={"temperature_C": 80},
            temperature=80,
            p_h2=1.0,
            p_o2=0.21
        )
        
        print("\n" + "-"*80)
        print("RESULTS WITH RAG:")
        print("-"*80)
        summary = result.get_summary()
        print(f"Success: {summary['success']}")
        print(f"Attempts: {summary['total_attempts']}")
        
        return result
        
    except Exception as e:
        print(f"⚠ RAG demo skipped: {e}")
        return None


def main():
    print("\n" + "="*80)
    print("CLOSED-LOOP SELF-CORRECTING CODE GENERATION DEMO")
    print("="*80)
    print("\nThis demonstrates the complete workflow:")
    print("1. User query → RAG retrieval")
    print("2. LLM generates code")
    print("3. Physics validator checks code")
    print("4. IF violations → feedback to LLM → regenerate")
    print("5. Loop until valid or max iterations")
    print("="*80)
    
    all_results = []
    
    result1 = demo_pemfc_closed_loop()
    all_results.append(result1)
    
    result2 = demo_vrfb_closed_loop()
    all_results.append(result2)
    
    batch_results = demo_batch_generation()
    all_results.extend(batch_results)
    
    result_rag = demo_with_rag()
    if result_rag:
        all_results.append(result_rag)
    
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    total_success = sum(r.success for r in all_results)
    total_queries = len(all_results)
    avg_attempts = sum(r.total_attempts for r in all_results) / total_queries
    total_time = sum(r.total_time for r in all_results)
    total_tokens = sum(r.total_tokens for r in all_results)
    
    print(f"Total Queries: {total_queries}")
    print(f"Success Rate: {total_success/total_queries*100:.1f}%")
    print(f"Average Attempts: {avg_attempts:.2f}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Total Tokens: {total_tokens}")
    
    print("\n✓ Demo complete!")
    print("\nKey Takeaways:")
    print("- The closed loop enables iterative refinement")
    print("- Physics validation catches errors automatically")
    print("- Feedback guides LLM to fix violations")
    print("- This matches the paper's claim of 'closed loop for hypothesis")
    print("  generation, simulation, and refinement'")


if __name__ == "__main__":
    main()
