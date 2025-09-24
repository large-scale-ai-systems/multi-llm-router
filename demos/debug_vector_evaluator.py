#!/usr/bin/env python3
"""
Debug script to test vector evaluator initialization and functionality
"""

import sys
sys.path.append('.')

from main import create_router

def debug_vector_evaluator():
    """Debug vector evaluator to find initialization issues"""
    
    print("=== Vector Evaluator Debug ===")
    
    # Create router in mock mode for faster testing
    print("Creating router...")
    router = create_router(use_real_providers=False)
    
    evaluator = router.vector_db_evaluator
    print(f"Evaluator type: {type(evaluator).__name__}")
    print(f"Number of records: {len(evaluator.human_eval_records)}")
    print(f"Similarity threshold: {evaluator.similarity_threshold}")
    
    # Test various queries with different similarity thresholds
    test_queries = [
        "machine learning",
        "python function", 
        "algorithm implementation",
        "data analysis",
        "neural network"
    ]
    
    print("\n=== Testing with default threshold (0.7) ===")
    for query in test_queries:
        similar = evaluator.find_similar_examples(query, top_k=3)
        print(f"'{query}': {len(similar)} matches")
        if similar:
            break
    
    # Lower the threshold to see what scores are actually generated
    print(f"\n=== Testing with lower thresholds ===")
    thresholds = [0.5, 0.3, 0.1, 0.05]
    
    for threshold in thresholds:
        print(f"\nThreshold: {threshold}")
        old_threshold = evaluator.similarity_threshold
        evaluator.similarity_threshold = threshold
        
        similar = evaluator.find_similar_examples("machine learning", top_k=5)
        print(f"  Matches: {len(similar)}")
        
        for record_id, score in similar[:3]:
            print(f"    Record {record_id}: {score:.3f}")
            # Get the actual record
            record = next((r for r in evaluator.human_eval_records if r.id == record_id), None)
            if record:
                prompt_preview = record.prompt[:60] + "..." if len(record.prompt) > 60 else record.prompt
                print(f"      -> {prompt_preview}")
        
        # Restore threshold
        evaluator.similarity_threshold = old_threshold
        
        if similar:
            break  # Found working threshold
    
    print(f"\n=== Recommendations ===")
    if similar:
        max_score = max(score for _, score in similar)
        print(f"✓ Vector evaluator is working!")
        print(f"✓ Maximum similarity score found: {max_score:.3f}")
        print(f"💡 Consider lowering similarity threshold from 0.7 to {max_score * 0.8:.2f}")
    else:
        print("❌ No similar examples found even with very low threshold")
        print("❌ There may be an issue with the similarity calculation")

if __name__ == "__main__":
    debug_vector_evaluator()