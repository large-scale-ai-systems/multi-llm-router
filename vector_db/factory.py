"""
Vector Evaluator Factory

Factory pattern implementation for automatic selection and creation of 
vector database evaluators based on platform capabilities and availability.
"""

import platform
import sys
from typing import Dict, Any, Optional, Type
from pathlib import Path

from .abstract_vector_evaluator import AbstractVectorEvaluator
from core.data_models import HumanEvalRecord, LLMResponse


class VectorEvaluatorFactory:
    """Factory for creating appropriate vector evaluator implementations"""
    
    # Registry of available implementations
    _implementations = {}
    _fallback_order = []
    
    @classmethod
    def register_implementation(cls, name: str, implementation_class: Type[AbstractVectorEvaluator], 
                              platforms: list = None, priority: int = 100):
        """
        Register a vector evaluator implementation
        
        Args:
            name: Implementation name
            implementation_class: Class implementing AbstractVectorEvaluator
            platforms: List of supported platforms (None = all platforms)
            priority: Priority for auto-selection (lower = higher priority)
        """
        cls._implementations[name] = {
            'class': implementation_class,
            'platforms': platforms or ['Windows', 'Linux', 'Darwin'],
            'priority': priority,
            'available': None  # Lazy evaluation
        }
    
    @classmethod
    def _check_implementation_availability(cls, name: str) -> bool:
        """
        Check if an implementation is available on current platform
        
        Args:
            name: Implementation name
            
        Returns:
            bool: True if available
        """
        impl_info = cls._implementations.get(name)
        if not impl_info:
            return False
        
        # Check if availability already determined
        if impl_info['available'] is not None:
            return impl_info['available']
        
        # Check platform compatibility
        current_platform = platform.system()
        if current_platform not in impl_info['platforms']:
            impl_info['available'] = False
            return False
        
        # Try to import and instantiate
        try:
            implementation_class = impl_info['class']
            
            # Check specific availability based on implementation
            if name == 'faiss':
                # Check FAISS dependencies
                try:
                    import faiss
                    import torch
                    from sentence_transformers import SentenceTransformer
                    impl_info['available'] = True
                    return True
                except ImportError:
                    impl_info['available'] = False
                    return False
            
            elif name == 'colbert':
                # Check ColBERT dependencies and platform
                try:
                    from colbert import Indexer, Searcher
                    from colbert.infra import Run, RunConfig, ColBERTConfig
                    # ColBERT works better on Linux
                    impl_info['available'] = current_platform == 'Linux'
                    return impl_info['available']
                except ImportError:
                    impl_info['available'] = False
                    return False
            
            elif name == 'chroma':
                # Check ChromaDB dependencies
                try:
                    import chromadb
                    impl_info['available'] = True
                    return True
                except ImportError:
                    impl_info['available'] = False
                    return False
            
            else:
                # Default: try to instantiate with minimal parameters
                temp_path = Path.cwd() / 'temp_test_index'
                temp_path.mkdir(exist_ok=True)
                try:
                    instance = implementation_class(str(temp_path))
                    impl_info['available'] = True
                    temp_path.rmdir()  # Clean up
                    return True
                except Exception:
                    impl_info['available'] = False
                    if temp_path.exists():
                        temp_path.rmdir()
                    return False
        
        except Exception as e:
            print(f"Warning: Error checking {name} implementation availability: {e}")
            impl_info['available'] = False
            return False
    
    @classmethod
    def get_available_implementations(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get all available implementations on current platform
        
        Returns:
            Dictionary mapping implementation names to their info
        """
        available = {}
        for name, info in cls._implementations.items():
            if cls._check_implementation_availability(name):
                available[name] = {
                    'class': info['class'].__name__,
                    'platforms': info['platforms'],
                    'priority': info['priority'],
                    'available': True
                }
        return available
    
    @classmethod
    def get_best_implementation(cls) -> Optional[str]:
        """
        Get the best available implementation based on platform and priority
        
        Returns:
            Implementation name or None if none available
        """
        available_impls = []
        
        for name, info in cls._implementations.items():
            if cls._check_implementation_availability(name):
                available_impls.append((name, info['priority']))
        
        if not available_impls:
            return None
        
        # Sort by priority (lower number = higher priority)
        available_impls.sort(key=lambda x: x[1])
        return available_impls[0][0]
    
    @classmethod
    def create_evaluator(cls, implementation: str = None, 
                        index_path: str = "./vector_indexes",
                        similarity_threshold: float = 0.7,
                        **kwargs) -> AbstractVectorEvaluator:
        """
        Create a vector evaluator instance
        
        Args:
            implementation: Specific implementation to use (None for auto-select)
            index_path: Path to store vector indexes
            similarity_threshold: Similarity threshold for matching
            **kwargs: Additional arguments for specific implementations
            
        Returns:
            AbstractVectorEvaluator instance
            
        Raises:
            RuntimeError: If no suitable implementation found
        """
        if implementation is None:
            implementation = cls.get_best_implementation()
        
        if implementation is None:
            raise RuntimeError("No suitable vector evaluator implementation found")
        
        if not cls._check_implementation_availability(implementation):
            raise RuntimeError(f"Implementation '{implementation}' not available on current platform")
        
        implementation_class = cls._implementations[implementation]['class']
        
        print(f"Creating vector evaluator implementation: {implementation}")
        print(f"Index storage path: {index_path}")
        
        # Auto-adjust similarity threshold based on implementation type
        adjusted_threshold = similarity_threshold
        if implementation == 'token_overlap':
            # Token overlap produces lower similarity scores
            adjusted_threshold = min(similarity_threshold, 0.2)
            if similarity_threshold > 0.3:
                print(f"Auto-adjusting similarity threshold from {similarity_threshold} to {adjusted_threshold} for token overlap")
        elif implementation in ['faiss', 'colbert']:
            # Semantic similarity can use higher thresholds
            adjusted_threshold = max(similarity_threshold, 0.5)
        
        print(f"Similarity threshold: {adjusted_threshold}")
        
        # Create instance with appropriate parameters
        try:
            return implementation_class(
                index_path=index_path,
                similarity_threshold=adjusted_threshold,
                **kwargs
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create {implementation} evaluator: {e}")
    
    @classmethod
    def get_platform_info(cls) -> Dict[str, Any]:
        """
        Get detailed platform information for debugging
        
        Returns:
            Platform information dictionary
        """
        return {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': sys.version,
            'python_implementation': platform.python_implementation(),
            'available_implementations': list(cls.get_available_implementations().keys()),
            'best_implementation': cls.get_best_implementation()
        }


# Auto-register implementations when module is imported
def _register_default_implementations():
    """Register default implementations with the factory"""
    
    # FAISS implementation (Windows + Linux, high priority)
    try:
        from .faiss_evaluator import FAISSVectorEvaluator
        VectorEvaluatorFactory.register_implementation(
            'faiss', 
            FAISSVectorEvaluator, 
            platforms=['Windows', 'Linux', 'Darwin'],
            priority=10  # High priority
        )
    except ImportError:
        pass
    
    # ColBERT implementation (Linux preferred, medium priority)  
    try:
        from .colbert_evaluator import ColBERTVectorDBEvaluator
        VectorEvaluatorFactory.register_implementation(
            'colbert',
            ColBERTVectorDBEvaluator,
            platforms=['Linux', 'Darwin'],  # Prefer Unix systems
            priority=20  # Medium priority, but Linux-specific
        )
    except ImportError:
        pass
    
    # Token overlap fallback (all platforms, lowest priority)
    try:
        from .token_overlap_evaluator import TokenOverlapEvaluator  # We'll create this next
        VectorEvaluatorFactory.register_implementation(
            'token_overlap',
            TokenOverlapEvaluator,
            platforms=['Windows', 'Linux', 'Darwin'],
            priority=100  # Lowest priority fallback
        )
    except ImportError:
        pass


# Register implementations on module import
_register_default_implementations()


# Convenience function for easy usage
def create_vector_evaluator(implementation: str = None, **kwargs) -> AbstractVectorEvaluator:
    """
    Convenience function to create a vector evaluator
    
    Args:
        implementation: Implementation name or None for auto-select
        **kwargs: Arguments passed to evaluator constructor
        
    Returns:
        AbstractVectorEvaluator instance
    """
    return VectorEvaluatorFactory.create_evaluator(implementation, **kwargs)