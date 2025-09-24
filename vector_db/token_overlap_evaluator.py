"""
Token Overlap Vector Evaluator

Simple fallback implementation using token overlap similarity.
Works on all platforms without external dependencies.
"""

import re
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set
from collections import Counter

from .abstract_vector_evaluator import AbstractVectorEvaluator
from core.data_models import HumanEvalRecord, LLMResponse


class TokenOverlapEvaluator(AbstractVectorEvaluator):
    """Simple token overlap-based similarity evaluator as fallback"""
    
    def __init__(self, index_path: str, similarity_threshold: float = 0.7):
        """
        Initialize Token Overlap Evaluator
        
        Args:
            index_path: Directory path (not used but maintained for interface consistency)
            similarity_threshold: Minimum similarity score for matches
        """
        super().__init__(index_path, similarity_threshold)
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Tokenization settings
        self.use_stemming = False  # Keep simple to avoid NLTK dependency
        self.min_token_length = 2
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'would', 'you', 'your'
        }
        
        print(f"Token Overlap Evaluator initialized at {self.index_path}")
        print("Using token overlap similarity as fallback implementation")
    
    def _tokenize(self, text: str) -> Set[str]:
        """
        Tokenize text into normalized tokens
        
        Args:
            text: Input text
            
        Returns:
            Set of normalized tokens
        """
        # Convert to lowercase and extract alphanumeric tokens
        text = text.lower()
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text)
        
        # Filter tokens
        filtered_tokens = set()
        for token in tokens:
            if (len(token) >= self.min_token_length and 
                token not in self.stop_words and
                not token.isdigit()):
                filtered_tokens.add(token)
        
        return filtered_tokens
    
    def _calculate_jaccard_similarity(self, tokens1: Set[str], tokens2: Set[str]) -> float:
        """
        Calculate Jaccard similarity between two token sets
        
        Args:
            tokens1: First token set
            tokens2: Second token set
            
        Returns:
            Jaccard similarity score (0.0 to 1.0)
        """
        if not tokens1 and not tokens2:
            return 1.0  # Both empty
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_cosine_similarity(self, tokens1: Set[str], tokens2: Set[str]) -> float:
        """
        Calculate cosine similarity using token counts
        
        Args:
            tokens1: First token set
            tokens2: Second token set
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if not tokens1 and not tokens2:
            return 1.0
        
        # Convert to counters for cosine similarity
        all_tokens = tokens1.union(tokens2)
        if not all_tokens:
            return 0.0
        
        # Create vectors
        vec1 = [1 if token in tokens1 else 0 for token in all_tokens]
        vec2 = [1 if token in tokens2 else 0 for token in all_tokens]
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 * magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Combined similarity score (0.0 to 1.0)
        """
        tokens1 = self._tokenize(text1)
        tokens2 = self._tokenize(text2)
        
        # Calculate both similarities and take weighted average
        jaccard_sim = self._calculate_jaccard_similarity(tokens1, tokens2)
        cosine_sim = self._calculate_cosine_similarity(tokens1, tokens2)
        
        # Weight cosine similarity higher as it's generally more robust
        combined_sim = 0.3 * jaccard_sim + 0.7 * cosine_sim
        
        return combined_sim
    
    def ingest_human_eval_set(self, records: List[HumanEvalRecord]) -> bool:
        """
        Ingest human evaluation records (store in memory)
        
        Args:
            records: List of human evaluation records
            
        Returns:
            bool: Always True for token overlap method
        """
        if not records:
            print("Warning: No evaluation records provided for ingestion")
            return False
        
        self.human_eval_records = records
        print(f"Successfully ingested {len(records)} records for token overlap similarity")
        return True
    
    def find_similar_examples(self, query: str, top_k: int = 5) -> List[Tuple['HumanEvalRecord', float]]:
        """
        Find similar examples using token overlap
        
        Args:
            query: Query text to search for
            top_k: Number of top similar examples to return
            
        Returns:
            List of tuples (HumanEvalRecord, similarity_score)
        """
        if not self.human_eval_records:
            return []
        
        similarities = []
        
        for record in self.human_eval_records:
            # Combine prompt and golden output for comprehensive matching
            record_text = f"{record.prompt} {record.golden_output}"
            
            # Calculate similarity
            similarity = self._calculate_similarity(query, record_text)
            
            if similarity >= self.similarity_threshold:
                similarities.append((record, similarity))
        
        # Sort by similarity (descending) and take top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def evaluate_response_quality(self, response: LLMResponse, category: str = None) -> float:
        """
        Evaluate response quality using token overlap
        
        Args:
            response: LLM response to evaluate
            category: Optional category filter
            
        Returns:
            float: Quality score between 0.0 and 1.0
        """
        if not self.human_eval_records:
            return 0.0
        
        # Filter records by category if specified
        target_records = (self.get_records_by_category(category) 
                         if category else self.human_eval_records)
        
        if not target_records:
            return 0.0
        
        # Find similar examples
        similar_examples = self.find_similar_examples(response.content, top_k=3)
        
        if not similar_examples:
            return 0.0
        
        # Calculate weighted average similarity
        total_similarity = sum(score for _, score in similar_examples)
        avg_similarity = total_similarity / len(similar_examples)
        
        return avg_similarity
    
    def get_implementation_info(self) -> Dict[str, Any]:
        """Get token overlap implementation information"""
        return {
            'implementation': 'Token Overlap Similarity',
            'similarity_methods': ['Jaccard', 'Cosine'],
            'tokenization': 'Regex-based alphanumeric',
            'stop_words': len(self.stop_words),
            'min_token_length': self.min_token_length,
            'dependencies': 'None (Python standard library)',
            'num_records': len(self.human_eval_records),
            'similarity_threshold': self.similarity_threshold,
            'platform_independent': True
        }