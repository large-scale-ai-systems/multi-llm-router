"""
FAISS Vector Database Evaluator

High-performance vector database implementation using FAISS and Sentence Transformers.
Optimized for Windows compatibility and fast similarity search.
"""

import os
import pickle
import platform
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

# FAISS and Sentence Transformers imports
try:
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError as e:
    print(f"FAISS dependencies not available: {e}")
    FAISS_AVAILABLE = False
    # Create dummy objects to prevent NameError during class definition
    faiss = None

from .abstract_vector_evaluator import AbstractVectorEvaluator
from core.data_models import HumanEvalRecord, LLMResponse


class FAISSVectorEvaluator(AbstractVectorEvaluator):
    """FAISS-based vector database evaluator with sentence transformers"""
    
    def __init__(self, index_path: str, similarity_threshold: float = 0.7,
                 model_name: str = 'all-MiniLM-L6-v2', 
                 index_type: str = 'auto'):
        """
        Initialize FAISS Vector Database Evaluator
        
        Args:
            index_path: Directory to store FAISS indexes
            similarity_threshold: Minimum similarity score for matches
            model_name: Sentence transformer model name
            index_type: FAISS index type ('flat', 'ivf', 'hnsw', 'auto')
        """
        super().__init__(index_path, similarity_threshold)
        
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS and sentence-transformers are required but not installed")
        
        self.index_dir = Path(index_path)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.index_type = index_type
        
        # Performance optimization settings
        self.batch_size = 32
        self.normalize_embeddings = True
        self.use_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
        
        # Initialize model and index
        self._load_model()
        self.faiss_index = None
        self.record_ids = []  # Map FAISS index positions to record IDs
        
        # Try to load existing index from disk
        if self.index_dir.exists():
            print(f"Attempting to load existing FAISS index from {self.index_dir}")
            if self.load_index():
                print(f"Existing FAISS index loaded with {len(self.record_ids)} records")
            else:
                print("No existing index found or load failed - will create new index when data is ingested")
        else:
            print(f"Index directory {self.index_dir} does not exist - will be created when data is ingested")
        
        print(f"FAISS Vector DB initialized at {self.index_dir}")
        print(f"Using model: {self.model_name}, GPU acceleration: {self.use_gpu}")
    
    def _load_model(self):
        """Load sentence transformer model with optimizations"""
        print(f"Loading sentence transformer model: {self.model_name}")
        
        device = 'cuda' if self.use_gpu else 'cpu'
        self.model = SentenceTransformer(self.model_name, device=device)
        
        # Performance optimizations
        if self.use_gpu:
            self.model.to('cuda')
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model embedding dimension: {self.embedding_dim}")
    
    def _create_faiss_index(self, num_vectors: int):
        """
        Create optimized FAISS index based on dataset size
        
        Args:
            num_vectors: Number of vectors to index
            
        Returns:
            Configured FAISS index
        """
        if self.index_type == 'auto':
            # Auto-select index type based on dataset size
            if num_vectors < 1000:
                index_type = 'flat'
            elif num_vectors < 100000:
                index_type = 'ivf'
            else:
                index_type = 'hnsw'
        else:
            index_type = self.index_type
        
        print(f"Creating FAISS index type: {index_type} for {num_vectors} vectors")
        
        if index_type == 'flat':
            # Exact search - best for small datasets
            index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for cosine similarity
        
        elif index_type == 'ivf':
            # Inverted File Index - good balance of speed and accuracy
            n_centroids = min(int(np.sqrt(num_vectors)), 4096)
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, n_centroids)
            
        elif index_type == 'hnsw':
            # Hierarchical NSW - best for large datasets
            m = 16  # Number of bi-directional links for each node
            index = faiss.IndexHNSWFlat(self.embedding_dim, m)
            index.hnsw.efConstruction = 200  # Higher = better accuracy, slower indexing
            index.hnsw.efSearch = 100  # Higher = better accuracy, slower search
        
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # GPU optimization if available
        if self.use_gpu and hasattr(faiss, 'StandardGpuResources'):
            try:
                gpu_resources = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)
                print("FAISS GPU acceleration enabled")
            except Exception as e:
                print(f"GPU acceleration failed, fallback to CPU: {e}")
        
        return index
    
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings with optimization
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Normalized embeddings array
        """
        # Batch encoding for efficiency
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )
        
        return embeddings.astype(np.float32)  # FAISS prefers float32
    
    def ingest_human_eval_set(self, records: List[HumanEvalRecord]) -> bool:
        """
        Ingest human evaluation records with optimized indexing
        
        Args:
            records: List of human evaluation records
            
        Returns:
            bool: True if successful
        """
        if not records:
            print("Warning: No records to ingest")
            return False
        
        print(f"Ingesting {len(records)} human evaluation records into FAISS index")
        self.human_eval_records = records
        self.record_ids = []
        
        # Prepare texts for embedding
        texts = []
        for record in records:
            # Combine prompt and golden output for comprehensive similarity matching
            combined_text = f"{record.prompt} [GOLDEN] {record.golden_output}"
            texts.append(combined_text)
            self.record_ids.append(record.id)
        
        # Generate embeddings
        print("Generating vector embeddings for records")
        embeddings = self._encode_texts(texts)
        
        # Create and populate FAISS index
        self.faiss_index = self._create_faiss_index(len(embeddings))
        
        # Train index if necessary (for IVF)
        if hasattr(self.faiss_index, 'train') and not self.faiss_index.is_trained:
            print("Training FAISS index with vectors")
            self.faiss_index.train(embeddings)
        
        # Add vectors to index
        self.faiss_index.add(embeddings)
        
        # Save index and metadata
        self._save_index()
        
        print(f"Successfully indexed {len(records)} records in FAISS")
        return True
    
    def find_similar_examples(self, query: str, top_k: int = 5) -> List[Tuple['HumanEvalRecord', float]]:
        """
        Find similar examples using FAISS search
        
        Args:
            query: Query text to search for
            top_k: Number of top similar examples to return
            
        Returns:
            List of tuples (HumanEvalRecord, similarity_score)
        """
        if self.faiss_index is None or not self.record_ids:
            print("Warning: FAISS index not initialized or no records available")
            return []
        
        # Encode query
        query_embedding = self._encode_texts([query])
        
        # Search with FAISS
        try:
            # Adjust search parameters for HNSW
            if hasattr(self.faiss_index, 'hnsw'):
                original_ef = self.faiss_index.hnsw.efSearch
                self.faiss_index.hnsw.efSearch = max(top_k * 2, 50)  # Improve recall
            
            scores, indices = self.faiss_index.search(query_embedding, top_k)
            
            # Restore original parameter
            if hasattr(self.faiss_index, 'hnsw'):
                self.faiss_index.hnsw.efSearch = original_ef
            
        except Exception as e:
            print(f"FAISS similarity search failed: {e}")
            return []
        
        # Process results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1 and score >= self.similarity_threshold:  # Valid index and above threshold
                record_id = self.record_ids[idx]
                # Find the full record object
                record = next((r for r in self.human_eval_records if r.id == record_id), None)
                if record:
                    # Convert inner product back to cosine similarity (for normalized vectors)
                    similarity_score = float(score)
                    results.append((record, similarity_score))
        
        return results
    
    def evaluate_response_quality(self, response: LLMResponse, category: str = None) -> float:
        """
        Evaluate response quality using FAISS similarity search
        
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
        
        # Calculate weighted average of similarity scores
        total_weight = 0.0
        weighted_score = 0.0
        
        for record_id, similarity_score in similar_examples:
            weight = similarity_score  # Use similarity as weight
            weighted_score += similarity_score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def get_implementation_info(self) -> Dict[str, Any]:
        """Get FAISS implementation information"""
        return {
            'implementation': 'FAISS + Sentence Transformers',
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'index_type': getattr(self.faiss_index, '__class__.__name__', 'Unknown'),
            'gpu_enabled': self.use_gpu,
            'platform': platform.system(),
            'num_records': len(self.human_eval_records),
            'faiss_available': FAISS_AVAILABLE,
            'similarity_threshold': self.similarity_threshold
        }
    
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            index_path = self.index_dir / 'faiss.index'
            faiss.write_index(self.faiss_index, str(index_path))
            
            # Save metadata
            metadata = {
                'record_ids': self.record_ids,
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim,
                'similarity_threshold': self.similarity_threshold
            }
            metadata_path = self.index_dir / 'metadata.pkl'
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"FAISS index and metadata saved to {self.index_dir}")
        except Exception as e:
            print(f"Warning: Failed to save FAISS index: {e}")
    
    def load_index(self) -> bool:
        """Load FAISS index and metadata from disk"""
        try:
            index_path = self.index_dir / 'faiss.index'
            metadata_path = self.index_dir / 'metadata.pkl'
            
            if not index_path.exists() or not metadata_path.exists():
                return False
            
            # Load FAISS index
            self.faiss_index = faiss.read_index(str(index_path))
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.record_ids = metadata['record_ids']
            self.model_name = metadata['model_name']
            self.embedding_dim = metadata['embedding_dim']
            self.similarity_threshold = metadata['similarity_threshold']
            
            print(f"FAISS index loaded successfully from {self.index_dir}")
            return True
        except Exception as e:
            print(f"Warning: Failed to load FAISS index: {e}")
            return False