import os
import sys
from typing import Dict, Any, Optional, Tuple
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.llm_orchestrator.base_llm import BaseLLM
from src.llm_orchestrator.prompt_templates import (
    get_code_generation_prompt,
    get_system_prompt
)
from src.rag.chroma_manager import ChromaManager


class CodeGenerator:
    def __init__(
        self,
        rag_manager: Optional[ChromaManager] = None,
        verbose: bool = True
    ):
        self.verbose = verbose
        self.rag_manager = rag_manager
        
        system_prompt = get_system_prompt("physics_aware")
        self.llm = BaseLLM(system_prompt=system_prompt, verbose=verbose)
        
        if self.verbose:
            print("✓ CodeGenerator initialized")
    
    def generate_code(
        self,
        user_query: str,
        task_type: str = "equation_derivation",
        use_rag: bool = True,
        target_conditions: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        rag_context = ""
        rag_metadata = {}
        
        if use_rag and self.rag_manager:
            rag_context, rag_metadata = self._retrieve_context(
                user_query,
                target_conditions
            )
        
        prompt = get_code_generation_prompt(
            task_type=task_type,
            rag_context=rag_context,
            user_query=user_query,
            **kwargs
        )
        
        if self.verbose:
            print(f"Generating code for: {user_query[:50]}...")
        
        code, llm_metadata = self.llm.chat_completion(
            prompt,
            temperature=0.3,
            use_history=False
        )
        
        code = self._clean_code(code)
        
        metadata = {
            "task_type": task_type,
            "used_rag": use_rag,
            "rag_documents": len(rag_metadata.get("documents", [])),
            "llm_tokens": llm_metadata["total_tokens"],
            "llm_latency": llm_metadata["latency"]
        }
        
        return code, metadata
    
    def regenerate_with_feedback(
        self,
        original_query: str,
        previous_code: str,
        feedback: str,
        task_type: str = "equation_derivation",
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        refinement_prompt = f"""Previous code attempt:
```python
{previous_code}
```

{feedback}

Generate corrected code that addresses all issues above."""
        
        if self.verbose:
            print(f"Regenerating code with feedback...")
        
        code, llm_metadata = self.llm.chat_completion(
            refinement_prompt,
            temperature=0.2,
            use_history=False
        )
        
        code = self._clean_code(code)
        
        metadata = {
            "task_type": task_type,
            "is_refinement": True,
            "llm_tokens": llm_metadata["total_tokens"],
            "llm_latency": llm_metadata["latency"]
        }
        
        return code, metadata
    
    def _retrieve_context(
        self,
        query: str,
        target_conditions: Optional[Dict[str, float]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        if target_conditions:
            docs, metas, scores = self.rag_manager.hybrid_similarity_search(
                query=query,
                target_conditions=target_conditions,
                n_results=5
            )
        else:
            results = self.rag_manager.cosine_similarity_search(
                query=query,
                n_results=5
            )
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            scores = [1 - d for d in results['distances'][0]]
        
        context_parts = []
        for i, (doc, meta, score) in enumerate(zip(docs, metas, scores)):
            context_parts.append(f"[Source {i+1}, relevance={score:.3f}]\n{doc}")
        
        context = "\n\n".join(context_parts)
        
        metadata = {
            "documents": docs,
            "metadatas": metas,
            "scores": scores
        }
        
        return context, metadata
    
    def _clean_code(self, code: str) -> str:
        code = re.sub(r'^```python\s*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```\s*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n```\s*$', '', code, flags=re.MULTILINE)
        
        code = code.strip()
        
        return code
    
    def get_stats(self) -> Dict[str, Any]:
        return self.llm.get_token_stats()


def main():
    print("="*70)
    print("Testing CodeGenerator")
    print("="*70)
    
    generator = CodeGenerator(verbose=True)
    
    print("\n1. Simple code generation (without RAG):")
    code, metadata = generator.generate_code(
        user_query="Generate code to calculate Nernst potential for PEMFC",
        task_type="equation_derivation",
        use_rag=False,
        temperature=80,
        p_h2=1.0,
        p_o2=0.21
    )
    print(f"Generated {len(code)} characters of code")
    print(f"Metadata: {metadata}")
    print(f"\nCode preview:\n{code[:200]}...")
    
    print("\n2. Code regeneration with feedback:")
    feedback = "Add bounds checking for temperature and pressure"
    new_code, metadata = generator.regenerate_with_feedback(
        original_query="Calculate Nernst potential",
        previous_code=code,
        feedback=feedback
    )
    print(f"Regenerated {len(new_code)} characters of code")
    print(f"Metadata: {metadata}")
    
    print("\n3. Token statistics:")
    stats = generator.get_stats()
    for key, val in stats.items():
        print(f"  {key}: {val}")
    
    print("\n✓ CodeGenerator tests complete")


if __name__ == "__main__":
    main()
