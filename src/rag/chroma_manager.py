"""
ChromaDB Manager
Handles vector database initialization, document indexing, and hybrid retrieval.
"""

import os
import json
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import chromadb
from chromadb.config import Settings
from openai import AzureOpenAI

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.azure_config import EMBEDDING_CONFIG, AZURE_CONFIG


class ChromaManager:
    """
    Manages ChromaDB vector database for RAG pipeline.
    
    Features:
    - Azure OpenAI embeddings (text-embedding-3-large)
    - Hybrid similarity scoring (cosine + Mahalanobis + physics)
    - Metadata filtering for physics constraints
    """
    
    def __init__(
        self,
        persist_directory: str = "data/chromadb",
        collection_name: str = "electrochemical_literature"
    ):
        """
        Initialize ChromaDB manager.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize Azure OpenAI client for embeddings
        self.embedding_client = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=EMBEDDING_CONFIG["azure_endpoint"],
            api_key=EMBEDDING_CONFIG["api_key"]
        )
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"✓ Loaded existing collection: {collection_name}")
            print(f"  Documents: {self.collection.count()}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Electrochemical literature for PEMFC and VRFB"}
            )
            print(f"✓ Created new collection: {collection_name}")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding using Azure OpenAI.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector
        """
        response = self.embedding_client.embeddings.create(
            input=text,
            model=EMBEDDING_CONFIG["deployment_name"]
        )
        return response.data[0].embedding
    
    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Dict[str, Any],
        embedding: Optional[List[float]] = None
    ):
        """
        Add a single document to the collection.
        
        Args:
            doc_id: Unique document ID
            text: Document text
            metadata: Document metadata
            embedding: Precomputed embedding (optional)
        """
        if embedding is None:
            embedding = self.get_embedding(text)
        
        self.collection.add(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata]
        )
    
    def add_documents_batch(
        self,
        doc_ids: List[str],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 100
    ):
        """
        Add multiple documents in batches.
        
        Args:
            doc_ids: List of document IDs
            texts: List of document texts
            metadatas: List of metadata dicts
            batch_size: Batch size for embedding generation
        """
        n_docs = len(doc_ids)
        print(f"Adding {n_docs} documents in batches of {batch_size}...")
        
        for i in range(0, n_docs, batch_size):
            batch_ids = doc_ids[i:i+batch_size]
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            
            # Generate embeddings
            embeddings = []
            for text in batch_texts:
                emb = self.get_embedding(text)
                embeddings.append(emb)
            
            # Add to collection
            self.collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_metadatas
            )
            
            print(f"  Added batch {i//batch_size + 1}/{(n_docs-1)//batch_size + 1}")
        
        print(f"✓ Added {n_docs} documents")
    
    def cosine_similarity_search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform cosine similarity search.
        
        Args:
            query: Query text
            n_results: Number of results
            where: Metadata filter
        
        Returns:
            Search results
        """
        query_embedding = self.get_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        return results
    
    def hybrid_similarity_search(
        self,
        query: str,
        target_conditions: Optional[Dict[str, float]] = None,
        n_results: int = 5,
        lambda_cos: float = 0.5,
        lambda_mah: float = 0.3,
        lambda_phi: float = 0.2,
        where: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """
        Perform hybrid similarity search (Equation 3 from paper).
        
        s_j = λ_cos * cosine_sim - λ_mah * mahalanobis_dist - λ_phi * physics_penalty
        
        Args:
            query: Query text
            target_conditions: Target operating conditions (T, p, etc.)
            n_results: Number of results
            lambda_cos: Weight for cosine similarity
            lambda_mah: Weight for Mahalanobis distance
            lambda_phi: Weight for physics penalty
            where: Metadata filter
        
        Returns:
            Tuple of (document texts, metadatas, hybrid scores)
        """
        # First get cosine similarity results
        results = self.cosine_similarity_search(
            query, 
            n_results=n_results*3,  # Get more to re-rank
            where=where
        )
        
        if not results['documents'][0]:
            return [], [], []
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        cosine_distances = results['distances'][0]
        
        # Convert distances to similarities
        cosine_sims = [1 - d for d in cosine_distances]
        
        # If no target conditions, return cosine results
        if target_conditions is None:
            return (
                documents[:n_results],
                metadatas[:n_results],
                cosine_sims[:n_results]
            )
        
        # Compute hybrid scores
        hybrid_scores = []
        for idx, (doc, meta, cos_sim) in enumerate(zip(documents, metadatas, cosine_sims)):
            # Mahalanobis distance (simplified - using normalized Euclidean)
            mah_dist = self._compute_condition_distance(target_conditions, meta)
            
            # Physics penalty
            phi = self._compute_physics_penalty(target_conditions, meta)
            
            # Hybrid score (Equation 3)
            score = lambda_cos * cos_sim - lambda_mah * mah_dist - lambda_phi * phi
            hybrid_scores.append(score)
        
        # Sort by hybrid score (descending)
        sorted_indices = np.argsort(hybrid_scores)[::-1][:n_results]
        
        return (
            [documents[i] for i in sorted_indices],
            [metadatas[i] for i in sorted_indices],
            [hybrid_scores[i] for i in sorted_indices]
        )
    
    def _compute_condition_distance(
        self,
        target: Dict[str, float],
        metadata: Dict[str, Any]
    ) -> float:
        """
        Compute normalized distance between target and metadata conditions.
        
        Args:
            target: Target conditions (T, p, etc.)
            metadata: Document metadata
        
        Returns:
            Normalized distance [0, 1]
        """
        distances = []
        
        # Temperature distance
        if "temperature_C" in target and "temperature_C" in metadata:
            T_target = target["temperature_C"]
            T_meta = metadata["temperature_C"]
            # Normalize by typical range (0-100°C)
            dist = abs(T_target - T_meta) / 100.0
            distances.append(dist)
        
        # Pressure distance
        if "pressure_atm" in target and "pressure_atm" in metadata:
            p_target = target["pressure_atm"]
            p_meta = metadata["pressure_atm"]
            # Normalize by typical range (0-5 atm)
            dist = abs(p_target - p_meta) / 5.0
            distances.append(dist)
        
        # Average distance
        if distances:
            return np.mean(distances)
        return 0.0
    
    def _compute_physics_penalty(
        self,
        target: Dict[str, float],
        metadata: Dict[str, Any]
    ) -> float:
        """
        Compute physics constraint penalty.
        
        Args:
            target: Target conditions
            metadata: Document metadata
        
        Returns:
            Penalty value [0, 1]
        """
        penalty = 0.0
        
        # Example: Penalize if current density exceeds limiting current
        if "current_density_A_cm2" in target and "i_L_A_cm2" in metadata:
            i = target["current_density_A_cm2"]
            i_L = metadata["i_L_A_cm2"]
            if i >= i_L:
                penalty += 1.0
        
        # Example: Penalize if temperature out of material bounds
        if "temperature_C" in target:
            T = target["temperature_C"]
            T_min = metadata.get("T_min_C", 0)
            T_max = metadata.get("T_max_C", 100)
            if T < T_min or T > T_max:
                penalty += 0.5
        
        return penalty
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Statistics dictionary
        """
        count = self.collection.count()
        
        # Sample a few documents to check metadata
        if count > 0:
            sample = self.collection.get(limit=min(count, 5))
            sample_metadatas = sample['metadatas']
        else:
            sample_metadatas = []
        
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": self.persist_directory,
            "sample_metadatas": sample_metadatas
        }
    
    def reset(self):
        """Reset (delete) the collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Electrochemical literature for PEMFC and VRFB"}
        )
        print(f"✓ Reset collection: {self.collection_name}")


def main():
    """
    Test ChromaDB manager.
    """
    print("Testing ChromaDB Manager...")
    
    # Initialize
    manager = ChromaManager()
    
    # Get stats
    stats = manager.get_stats()
    print(f"\nCollection Stats:")
    print(f"  Name: {stats['collection_name']}")
    print(f"  Documents: {stats['document_count']}")
    print(f"  Directory: {stats['persist_directory']}")
    
    # Test adding a document (if empty)
    if stats['document_count'] == 0:
        print("\nAdding test document...")
        manager.add_document(
            doc_id="test_1",
            text="PEMFC with Pt/C catalyst operates at 80°C with exchange current density of 1e-7 A/cm².",
            metadata={
                "system": "PEMFC",
                "catalyst": "Pt/C",
                "temperature_C": 80,
                "i0_A_cm2": 1e-7,
                "source": "test"
            }
        )
        print("✓ Added test document")
    
    # Test cosine search
    print("\nTesting cosine similarity search...")
    results = manager.cosine_similarity_search(
        query="What is the exchange current density for PEMFC?",
        n_results=3
    )
    print(f"  Found {len(results['documents'][0])} results")
    
    # Test hybrid search
    print("\nTesting hybrid similarity search...")
    docs, metas, scores = manager.hybrid_similarity_search(
        query="PEMFC exchange current density",
        target_conditions={"temperature_C": 70},
        n_results=3
    )
    print(f"  Found {len(docs)} results with hybrid scoring")
    
    print("\n✓ ChromaDB Manager tests complete")


if __name__ == "__main__":
    main()
