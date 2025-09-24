"""
ColBERT Vector Database Evaluator for semantic similarity search.

This module implements vector database functionality using ColBERT
for advanced semantic similarity matching against human evaluation sets.
"""

import numpy as np
import platform
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

from .abstract_vector_evaluator import AbstractVectorEvaluator
from core.data_models import HumanEvalRecord, LLMResponse

# ColBERT imports
try:
    from colbert import Indexer, Searcher
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert.data import Queries, Collection
    COLBERT_AVAILABLE = True
except ImportError:
    print("ColBERT not available, using fallback implementation")
    COLBERT_AVAILABLE = False


class ColBERTVectorDBEvaluator(AbstractVectorEvaluator):
    """Vector database evaluator using ColBERT for semantic similarity"""
    
    def __init__(self, index_path: str = "./colbert_indexes", similarity_threshold: float = 0.7):
        super().__init__(index_path, similarity_threshold)
        self.index_path = Path(index_path)
        self.index_path.mkdir(exist_ok=True)
        self.similarity_threshold = similarity_threshold
        self.human_eval_records: List[HumanEvalRecord] = []
        self.colbert_available = COLBERT_AVAILABLE
        
        # Collection storage
        self.collection_path = self.index_path / "collection.tsv"
        self.index_name = "human_eval_index"
        
        print(f"ColBERT Vector DB initialized at {self.index_path}")
        if not self.colbert_available:
            print("Warning: ColBERT not available - using fallback similarity")
    
    def _create_collection_file(self):
        """Create ColBERT collection file from human evaluation records"""
        if not self.human_eval_records:
            print("No human evaluation records to create collection from")
            return
        
        with open(self.collection_path, 'w', encoding='utf-8') as f:
            for i, record in enumerate(self.human_eval_records):
                # Create searchable text combining prompt and golden output
                # Replace newlines and tabs to avoid TSV format issues
                clean_prompt = record.prompt.replace('\n', ' ').replace('\t', ' ')
                clean_golden = record.golden_output.replace('\n', ' ').replace('\t', ' ')
                combined_text = f"{clean_prompt} [GOLDEN] {clean_golden}"
                f.write(f"{i}\t{combined_text}\n")
        
        print(f"Created collection file with {len(self.human_eval_records)} records")
    
    def _setup_colbert_index(self):
        """Setup ColBERT indexing if available"""
        if not self.colbert_available:
            return False
        
        try:
            # Create collection file
            self._create_collection_file()
            
            # Setup ColBERT configuration
            config = ColBERTConfig(
                nbits=2,  # Use 2-bit compression
                kmeans_niters=4,  # Fewer k-means iterations for speed
                nranks=1,  # Single GPU/CPU
                doc_maxlen=256,  # Max document length
                query_maxlen=64,  # Max query length
            )
            
            # Initialize indexer
            with Run().context(RunConfig(nranks=1, experiment="llm_router")):
                indexer = Indexer(
                    checkpoint="answerdotai/answerai-colbert-small-v1",
                    config=config
                )
                
                # Create index
                indexer.index(
                    name=self.index_name,
                    collection=str(self.collection_path),
                    overwrite=True
                )
            
            print("ColBERT index created successfully")
            return True
            
        except Exception as e:
            print(f"Warning: ColBERT indexing failed: {str(e)[:200]}...")
            return False
    
    def ingest_human_eval_set(self, eval_records: List[HumanEvalRecord]):
        """Ingest human evaluation records into the vector database"""
        self.human_eval_records = eval_records
        print(f"Ingesting {len(eval_records)} human evaluation records...")
        
        # Create collection and try to set up ColBERT index
        success = self._setup_colbert_index()
        
        if success:
            print("Human evaluation set ingested with ColBERT indexing")
        else:
            print("Warning: Using fallback similarity - ColBERT indexing failed")
    
    def _colbert_similarity_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Perform similarity search using ColBERT"""
        try:
            config = ColBERTConfig()
            
            with Run().context(RunConfig(nranks=1, experiment="llm_router")):
                searcher = Searcher(index=self.index_name, config=config)
                results = searcher.search(query, k=top_k)
                
                # Convert to our format (doc_id, score)
                formatted_results = []
                for doc_id, rank, score in results:
                    formatted_results.append((doc_id, float(score)))
                
                return formatted_results
                
        except Exception as e:
            print(f"ColBERT search failed: {e}")
            return []
    
    def _fallback_similarity_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Fallback similarity using token overlap"""
        if not self.human_eval_records:
            return []
        
        query_tokens = set(query.lower().split())
        similarities = []
        
        for i, record in enumerate(self.human_eval_records):
            # Compare with both prompt and golden output
            combined_text = f"{record.prompt} {record.golden_output}".lower()
            doc_tokens = set(combined_text.split())
            
            # Calculate Jaccard similarity
            intersection = len(query_tokens.intersection(doc_tokens))
            union = len(query_tokens.union(doc_tokens))
            
            similarity = intersection / union if union > 0 else 0.0
            similarities.append((i, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def find_similar_examples(self, query: str, top_k: int = 5) -> List[Tuple[HumanEvalRecord, float]]:
        """Find similar examples from the human evaluation set"""
        if not self.human_eval_records:
            return []
        
        # Try ColBERT first, fallback to token overlap
        if self.colbert_available:
            results = self._colbert_similarity_search(query, top_k)
        else:
            results = self._fallback_similarity_search(query, top_k)
        
        # Convert results to evaluation records with scores
        similar_examples = []
        for doc_id, score in results:
            if 0 <= doc_id < len(self.human_eval_records):
                record = self.human_eval_records[doc_id]
                similar_examples.append((record, score))
        
        return similar_examples
    
    def evaluate_response_quality(self, prompt: str, response: str) -> float:
        """Evaluate response quality against human evaluation set"""
        if not self.human_eval_records:
            return 0.5  # Neutral score if no evaluation data
        
        # Find similar examples
        similar_examples = self.find_similar_examples(prompt, top_k=3)
        
        if not similar_examples:
            return 0.5
        
        # Calculate weighted quality score
        total_weight = 0.0
        weighted_score = 0.0
        
        for record, similarity in similar_examples:
            if similarity < 0.1:  # Skip very low similarity matches
                continue
            
            # Compare response with golden output using token overlap
            response_tokens = set(response.lower().split())
            golden_tokens = set(record.golden_output.lower().split())
            
            intersection = len(response_tokens.intersection(golden_tokens))
            union = len(response_tokens.union(golden_tokens))
            
            quality = intersection / union if union > 0 else 0.0
            
            # Weight by similarity to the prompt
            weight = similarity
            weighted_score += quality * weight
            total_weight += weight
        
        # Return weighted average
        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = 0.5
        
        return np.clip(final_score, 0.0, 1.0)
    
    def get_implementation_info(self) -> Dict[str, Any]:
        """Get ColBERT implementation information"""
        return {
            'implementation': 'ColBERT Vector Database',
            'colbert_available': self.colbert_available,
            'index_path': str(self.index_path),
            'similarity_threshold': self.similarity_threshold,
            'platform': platform.system(),
            'num_records': len(self.human_eval_records),
            'index_name': self.index_name,
            'fallback_mode': not self.colbert_available,
            'dependencies': 'colbert-ai, torch, transformers'
        }