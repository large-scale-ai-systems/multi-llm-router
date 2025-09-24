"""
Abstract Vector Evaluator Interface

This module defines the abstract base class for all vector database evaluators,
ensuring consistent API across different implementations (ColBERT, FAISS, etc.)
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from core.data_models import HumanEvalRecord, LLMResponse


class AbstractVectorEvaluator(ABC):
    """Abstract base class for vector database evaluators"""
    
    def __init__(self, index_path: str, similarity_threshold: float = 0.7):
        """
        Initialize vector evaluator
        
        Args:
            index_path: Directory to store vector indexes
            similarity_threshold: Minimum similarity score for matches
        """
        self.index_path = index_path
        self.similarity_threshold = similarity_threshold
        self.human_eval_records: List[HumanEvalRecord] = []
    
    @abstractmethod
    def ingest_human_eval_set(self, records: List[HumanEvalRecord]) -> bool:
        """
        Ingest human evaluation records into the vector database
        
        Args:
            records: List of human evaluation records
            
        Returns:
            bool: True if ingestion successful, False otherwise
        """
        pass
    
    @abstractmethod
    def find_similar_examples(self, query: str, top_k: int = 5) -> List[Tuple['HumanEvalRecord', float]]:
        """
        Find similar examples to the given query
        
        Args:
            query: Query text to search for
            top_k: Number of top similar examples to return
            
        Returns:
            List of tuples (HumanEvalRecord, similarity_score)
        """
        pass
    
    @abstractmethod
    def evaluate_response_quality(self, response: LLMResponse, category: str = None) -> float:
        """
        Evaluate response quality against human evaluation standards
        
        Args:
            response: LLM response to evaluate
            category: Optional category filter for evaluation
            
        Returns:
            float: Quality score between 0.0 and 1.0
        """
        pass
    
    @abstractmethod
    def get_implementation_info(self) -> Dict[str, Any]:
        """
        Get information about the current implementation
        
        Returns:
            Dictionary with implementation details
        """
        pass
    
    def get_similarity_threshold(self) -> float:
        """Get current similarity threshold"""
        return self.similarity_threshold
    
    def set_similarity_threshold(self, threshold: float) -> None:
        """Set similarity threshold"""
        if 0.0 <= threshold <= 1.0:
            self.similarity_threshold = threshold
        else:
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
    
    def get_record_count(self) -> int:
        """Get number of ingested records"""
        return len(self.human_eval_records)
    
    def get_records_by_category(self, category: str) -> List[HumanEvalRecord]:
        """Get all records for a specific category"""
        return [record for record in self.human_eval_records 
                if record.category.lower() == category.lower()]