import os
import sys
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.llm_orchestrator.code_generator import CodeGenerator
from src.llm_orchestrator.prompt_templates import get_feedback_prompt
from src.physics_validator.validator import PhysicsValidator, ValidationResult
from src.rag.chroma_manager import ChromaManager


@dataclass
class GenerationAttempt:
    attempt_number: int
    code: str
    validation_result: ValidationResult
    metadata: Dict[str, Any]
    timestamp: float
    
    def is_valid(self) -> bool:
        return self.validation_result.is_valid


@dataclass
class SelfCorrectionResult:
    success: bool
    final_code: str
    attempts: List[GenerationAttempt] = field(default_factory=list)
    total_attempts: int = 0
    total_time: float = 0.0
    total_tokens: int = 0
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "total_attempts": self.total_attempts,
            "total_time": self.total_time,
            "total_tokens": self.total_tokens,
            "final_valid": self.attempts[-1].is_valid() if self.attempts else False,
            "violation_history": [
                len(att.validation_result.violations) for att in self.attempts
            ]
        }


class SelfCorrectingAgent:
    """
    Implements the "LLM-RAG-Physics" orchestration framework described in the paper.
    
    This agent coordinates:
    1. Code Generation (using LLM + RAG)
    2. Physics Validation (checking constraints, units, syntax)
    3. Self-Correction Loop (regenerating code based on validator feedback)
    
    This corresponds to the "Closed-Loop Refinement" block in the system architecture.
    """
    def __init__(
        self,
        rag_manager: Optional[ChromaManager] = None,
        max_iterations: int = 3,
        verbose: bool = True
    ):
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        self.code_generator = CodeGenerator(
            rag_manager=rag_manager,
            verbose=verbose
        )
        self.validator = PhysicsValidator(verbose=verbose)
        
        if self.verbose:
            print("✓ SelfCorrectingAgent initialized")
            print(f"  Max iterations: {max_iterations}")
    
    def generate_validated_code(
        self,
        user_query: str,
        task_type: str = "equation_derivation",
        use_rag: bool = True,
        target_conditions: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> SelfCorrectionResult:
        start_time = time.time()
        result = SelfCorrectionResult(
            success=False,
            final_code="",
            total_attempts=0
        )
        
        if self.verbose:
            print("\n" + "="*70)
            print(f"Self-Correcting Generation: {user_query[:50]}...")
            print("="*70)
        
        for attempt_num in range(1, self.max_iterations + 1):
            if self.verbose:
                print(f"\n--- Attempt {attempt_num}/{self.max_iterations} ---")
            
            if attempt_num == 1:
                code, gen_metadata = self.code_generator.generate_code(
                    user_query=user_query,
                    task_type=task_type,
                    use_rag=use_rag,
                    target_conditions=target_conditions,
                    **kwargs
                )
            else:
                previous_attempt = result.attempts[-1]
                feedback = get_feedback_prompt(
                    previous_attempt.validation_result,
                    attempt_num
                )
                
                code, gen_metadata = self.code_generator.regenerate_with_feedback(
                    original_query=user_query,
                    previous_code=previous_attempt.code,
                    feedback=feedback,
                    task_type=task_type,
                    **kwargs
                )
            
            validation_result = self.validator.validate(
                code,
                context=kwargs
            )
            
            attempt = GenerationAttempt(
                attempt_number=attempt_num,
                code=code,
                validation_result=validation_result,
                metadata=gen_metadata,
                timestamp=time.time()
            )
            result.attempts.append(attempt)
            result.total_tokens += gen_metadata.get("llm_tokens", 0)
            
            if self.verbose:
                summary = validation_result.get_summary()
                print(f"  Validation: {'✓ PASS' if validation_result.is_valid else '✗ FAIL'}")
                print(f"  Violations: {summary['total_violations']}")
                print(f"  Warnings: {summary['warnings']}")
            
            if validation_result.is_valid:
                result.success = True
                result.final_code = code
                if self.verbose:
                    print(f"\n✓ Valid code generated in {attempt_num} attempt(s)")
                break
            else:
                if self.verbose and validation_result.violations:
                    print(f"  Issues found:")
                    for v in validation_result.violations[:3]:
                        print(f"    - {v}")
        
        if not result.success:
            result.final_code = result.attempts[-1].code if result.attempts else ""
            if self.verbose:
                print(f"\n⚠ Failed to generate valid code after {self.max_iterations} attempts")
        
        result.total_attempts = len(result.attempts)
        result.total_time = time.time() - start_time
        
        if self.verbose:
            print(f"\nTotal time: {result.total_time:.2f}s")
            print(f"Total tokens: {result.total_tokens}")
        
        return result
    
    def batch_generate(
        self,
        queries: List[str],
        task_type: str = "equation_derivation",
        **kwargs
    ) -> List[SelfCorrectionResult]:
        results = []
        
        if self.verbose:
            print(f"\nBatch generation: {len(queries)} queries")
        
        for i, query in enumerate(queries):
            if self.verbose:
                print(f"\n[Query {i+1}/{len(queries)}]")
            
            result = self.generate_validated_code(
                user_query=query,
                task_type=task_type,
                **kwargs
            )
            results.append(result)
        
        return results
    
    def get_success_rate(self, results: List[SelfCorrectionResult]) -> float:
        if not results:
            return 0.0
        return sum(r.success for r in results) / len(results)
    
    def get_average_attempts(self, results: List[SelfCorrectionResult]) -> float:
        if not results:
            return 0.0
        return sum(r.total_attempts for r in results) / len(results)
    
    def get_statistics(
        self,
        results: List[SelfCorrectionResult]
    ) -> Dict[str, Any]:
        if not results:
            return {}
        
        return {
            "total_queries": len(results),
            "success_rate": self.get_success_rate(results),
            "average_attempts": self.get_average_attempts(results),
            "total_time": sum(r.total_time for r in results),
            "total_tokens": sum(r.total_tokens for r in results),
            "first_attempt_success": sum(
                r.success and r.total_attempts == 1 for r in results
            ) / len(results)
        }


def main():
    print("="*70)
    print("Testing SelfCorrectingAgent")
    print("="*70)
    
    agent = SelfCorrectingAgent(
        rag_manager=None,
        max_iterations=3,
        verbose=True
    )
    
    print("\n1. Single query with self-correction:")
    result = agent.generate_validated_code(
        user_query="Calculate PEMFC voltage with Butler-Volmer kinetics",
        task_type="pemfc_fitting",
        use_rag=False,
        temperature=80,
        p_h2=1.0,
        p_o2=0.21
    )
    
    print(f"\nResult summary:")
    summary = result.get_summary()
    for key, val in summary.items():
        print(f"  {key}: {val}")
    
    if result.success:
        print(f"\n✓ Final code ({len(result.final_code)} chars):")
        print(result.final_code[:300] + "...")
    
    print("\n2. Batch generation test:")
    queries = [
        "Calculate Nernst potential for PEMFC",
        "Compute activation overpotential using Tafel equation"
    ]
    
    batch_results = agent.batch_generate(
        queries=queries,
        task_type="equation_derivation",
        use_rag=False
    )
    
    print(f"\nBatch statistics:")
    stats = agent.get_statistics(batch_results)
    for key, val in stats.items():
        print(f"  {key}: {val}")
    
    print("\n✓ SelfCorrectingAgent tests complete")


if __name__ == "__main__":
    main()
