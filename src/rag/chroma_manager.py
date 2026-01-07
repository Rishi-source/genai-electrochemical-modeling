import os
import json
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import chromadb
from chromadb.config import Settings
from openai import AzureOpenAI
import pickle

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.azure_config import EMBEDDING_CONFIG, AZURE_CONFIG


class ChromaManager:
    def __init__(
        self,
        persist_directory: str = "data/chromadb",
        collection_name: str = "electrochemical_literature"
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.covariance_path = os.path.join(persist_directory, f"{collection_name}_covariance.pkl")
        self.covariance_matrix = None
        self.condition_features = ['temperature_C', 'pressure_atm', 'current_density_A_cm2']
        
        self.embedding_client = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=EMBEDDING_CONFIG["azure_endpoint"],
            api_key=EMBEDDING_CONFIG["api_key"]
        )
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
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
        
        self._load_or_compute_covariance()
    
    def get_embedding(self, text: str) -> List[float]:
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
        n_docs = len(doc_ids)
        print(f"Adding {n_docs} documents in batches of {batch_size}...")
        
        for i in range(0, n_docs, batch_size):
            batch_ids = doc_ids[i:i+batch_size]
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            
            embeddings = []
            for text in batch_texts:
                emb = self.get_embedding(text)
                embeddings.append(emb)
            
            self.collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_metadatas
            )
            
            print(f"  Added batch {i//batch_size + 1}/{(n_docs-1)//batch_size + 1}")
        
        print(f"✓ Added {n_docs} documents")
        
        self.compute_covariance_matrix()
    
    def cosine_similarity_search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
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
        Performs hybrid similarity search using Equation 3 from the paper.
        
        Score S(d, q) = λ_cos * CosineSim(d, q) - λ_mah * MahalanobisDist(d_cond, q_cond) - λ_phi * PhysicsPenalty(d, q)
        
        Where:
        - CosineSim: Semantic similarity between query and document text
        - MahalanobisDist: Statistical distance between operating conditions (T, P, I)
        - PhysicsPenalty: Hard penalty for violated constraints (e.g., T < T_min)
        
        Args:
            query: Natural language query
            target_conditions: target operating conditions (e.g., {'temperature_C': 80})
            n_results: Number of results to return
            lambda_cos: Weight for cosine similarity (default 0.5)
            lambda_mah: Weight for Mahalanobis distance (default 0.3)
            lambda_phi: Weight for physics penalty (default 0.2)
            where: Optional ChromaDB filter
            
        Returns:
            Tuple of (documents, metadatas, scores)
        """
        results = self.cosine_similarity_search(
            query, 
            n_results=n_results*3,
            where=where
        )
        
        if not results['documents'][0]:
            return [], [], []
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        cosine_distances = results['distances'][0]
        cosine_sims = [1 - d for d in cosine_distances]
        
        if target_conditions is None:
            return (
                documents[:n_results],
                metadatas[:n_results],
                cosine_sims[:n_results]
            )
        
        hybrid_scores = []
        for idx, (doc, meta, cos_sim) in enumerate(zip(documents, metadatas, cosine_sims)):
            mah_dist = self._compute_mahalanobis_distance(target_conditions, meta)
            phi = self._compute_physics_penalty(target_conditions, meta)
            score = lambda_cos * cos_sim - lambda_mah * mah_dist - lambda_phi * phi
            hybrid_scores.append(score)
        
        sorted_indices = np.argsort(hybrid_scores)[::-1][:n_results]
        
        return (
            [documents[i] for i in sorted_indices],
            [metadatas[i] for i in sorted_indices],
            [hybrid_scores[i] for i in sorted_indices]
        )
    
    def _load_or_compute_covariance(self):
        if os.path.exists(self.covariance_path):
            try:
                with open(self.covariance_path, 'rb') as f:
                    self.covariance_matrix = pickle.load(f)
                print(f"✓ Loaded covariance matrix from {self.covariance_path}")
                return
            except Exception as e:
                print(f"⚠ Could not load covariance matrix: {e}")
        
        self.compute_covariance_matrix()
    
    def compute_covariance_matrix(self):
        if self.collection.count() < 10:
            print(f"⚠ Too few documents ({self.collection.count()}) to compute covariance matrix")
            self.covariance_matrix = None
            return
        
        all_data = self.collection.get(include=['metadatas'])
        metadatas = all_data['metadatas']
        
        condition_vectors = []
        for meta in metadatas:
            vector = []
            for feature in self.condition_features:
                vector.append(meta.get(feature, 0.0))
            condition_vectors.append(vector)
        
        condition_vectors = np.array(condition_vectors)
        
        try:
            cov = np.cov(condition_vectors.T)
            epsilon = 1e-6
            cov += epsilon * np.eye(cov.shape[0])
            self.covariance_matrix = cov
            
            os.makedirs(os.path.dirname(self.covariance_path), exist_ok=True)
            with open(self.covariance_path, 'wb') as f:
                pickle.dump(self.covariance_matrix, f)
            
            print(f"✓ Computed and saved covariance matrix: shape {cov.shape}")
        except Exception as e:
            print(f"⚠ Error computing covariance matrix: {e}")
            self.covariance_matrix = None
    
    def _compute_mahalanobis_distance(
        self,
        target: Dict[str, float],
        metadata: Dict[str, Any]
    ) -> float:
        if self.covariance_matrix is None:
            return self._compute_simple_distance(target, metadata)
        
        x_target = np.array([target.get(f, 0.0) for f in self.condition_features])
        x_meta = np.array([metadata.get(f, 0.0) for f in self.condition_features])
        
        diff = x_meta - x_target
        
        try:
            S_inv = np.linalg.inv(self.covariance_matrix)
            mah_dist = np.sqrt(diff.T @ S_inv @ diff)
            normalized_dist = 1 - np.exp(-mah_dist / 10.0)
            return normalized_dist
        except np.linalg.LinAlgError:
            return self._compute_simple_distance(target, metadata)
    
    def _compute_simple_distance(
        self,
        target: Dict[str, float],
        metadata: Dict[str, Any]
    ) -> float:
        distances = []
        
        if "temperature_C" in target and "temperature_C" in metadata:
            T_target = target["temperature_C"]
            T_meta = metadata["temperature_C"]
            dist = abs(T_target - T_meta) / 100.0
            distances.append(dist)
        
        if "pressure_atm" in target and "pressure_atm" in metadata:
            p_target = target["pressure_atm"]
            p_meta = metadata["pressure_atm"]
            dist = abs(p_target - p_meta) / 5.0
            distances.append(dist)
        
        if "current_density_A_cm2" in target and "current_density_A_cm2" in metadata:
            i_target = target["current_density_A_cm2"]
            i_meta = metadata["current_density_A_cm2"]
            dist = abs(i_target - i_meta) / 2.0
            distances.append(dist)
        
        if distances:
            return np.mean(distances)
        return 0.0
    
    def _compute_physics_penalty(
        self,
        target: Dict[str, float],
        metadata: Dict[str, Any]
    ) -> float:
        penalty = 0.0
        if "current_density_A_cm2" in target and "i_L_A_cm2" in metadata:
            i = target["current_density_A_cm2"]
            i_L = metadata["i_L_A_cm2"]
            if i >= i_L:
                penalty += 1.0
        
        if "temperature_C" in target:
            T = target["temperature_C"]
            T_min = metadata.get("T_min_C", 0)
            T_max = metadata.get("T_max_C", 100)
            if T < T_min or T > T_max:
                penalty += 0.5
        
        return penalty
    
    def get_stats(self) -> Dict[str, Any]:
        count = self.collection.count()
        
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
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Electrochemical literature for PEMFC and VRFB"}
        )
        print(f"✓ Reset collection: {self.collection_name}")


def main():
    print("Testing ChromaDB Manager...")
    
    manager = ChromaManager()
    
    stats = manager.get_stats()
    print(f"\nCollection Stats:")
    print(f"  Name: {stats['collection_name']}")
    print(f"  Documents: {stats['document_count']}")
    print(f"  Directory: {stats['persist_directory']}")
    
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
    
    print("\nTesting cosine similarity search...")
    results = manager.cosine_similarity_search(
        query="What is the exchange current density for PEMFC?",
        n_results=3
    )
    print(f"  Found {len(results['documents'][0])} results")
    
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
