"""
Demonstration o    print(f"Platform: {platform_info['platform']}")
    print(f"Python: {platform_info['python_implementation']} {platform_info['python_version'].split()[0]}")
    print(f"Architecture: {platform_info['architecture']}")AIS    print(f"Available Implementations: {list(available_impls.keys())}") Vect    print(f"Creating Vector Evaluator using factory pattern...")r Eval    print(f"Loading Human Evaluation Data from CSV...")ator    print(f"Ingesting evaluation data into vector database...")Factory     print(f"Similarity Search Demonstration...")ystem

This scrip    print(f"Quality Evaluation Demonstration...") demonstrates the c    print(f"Implementation Details:")mple    print(f"Demo completed successfully")e Windows-compatible vector search system
with automatic platform detection and FAISS implementation.
"""

from vector_db.factory import create_vector_evaluator, VectorEvaluatorFactory
from main import create_human_eval_set
from core.data_models import LLMResponse


def demo_vector_evaluator_system():
    """Demonstrate the complete vector evaluator system"""
    
    print("Vector Evaluator System Demo")
    print("=" * 50)
    
    # Step 1: Show platform detection
    platform_info = VectorEvaluatorFactory.get_platform_info()
    print(f"Platform: {platform_info['platform']}")
    print(f"Python: {platform_info['python_implementation']} {platform_info['python_version'].split()[0]}")
    print(f"Architecture: {platform_info['architecture']}")
    
    # Step 2: Show available implementations
    available_impls = VectorEvaluatorFactory.get_available_implementations()
    print(f"\nAvailable Implementations: {list(available_impls.keys())}")
    
    best_impl = VectorEvaluatorFactory.get_best_implementation()
    print(f"Best Implementation: {best_impl}")
    
    # Step 3: Create evaluator using factory
    print(f"\nCreating Vector Evaluator using factory pattern...")
    evaluator = create_vector_evaluator(
        index_path="./demo_indexes",
        similarity_threshold=0.3  # Lower threshold for demo
    )
    
    # Step 4: Load evaluation data
    print(f"\nLoading Human Evaluation Data from CSV...")
    eval_set = create_human_eval_set()
    print(f"Loaded {len(eval_set)} evaluation records")
    
    # Step 5: Ingest data
    print(f"\nIngesting evaluation data into vector database...")
    success = evaluator.ingest_human_eval_set(eval_set)
    if success:
        print("Data ingestion successful")
    else:
        print("Error: Data ingestion failed")
        return
    
    # Step 6: Demo similarity search
    print(f"\nSimilarity Search Demonstration...")
    
    test_queries = [
        "How to write a Python function",
        "Machine learning algorithms",  
        "Create a REST API",
        "Explain recursion in programming"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = evaluator.find_similar_examples(query, top_k=2)
        
        if results:
            for i, (record_id, score) in enumerate(results):
                print(f"  {i+1}. Record ID: {record_id}, Similarity: {score:.3f}")
        else:
            print("  No similar examples found")
    
    # Step 7: Demo quality evaluation
    print(f"\nQuality Evaluation Demonstration...")
    
    test_responses = [
        ("Good response", "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"),
        ("Poor response", "I don't know anything about programming"),
        ("Moderate response", "You can use a loop or recursion to implement fibonacci")
    ]
    
    for desc, content in test_responses:
        response = LLMResponse(
            content=content,
            generation_time=1.0,
            token_count=len(content.split()),
            llm_id="demo_llm"
        )
        
        quality_score = evaluator.evaluate_response_quality(response)
        print(f"  {desc}: Quality Score = {quality_score:.3f}")
    
    # Step 8: Show implementation info
    print(f"\nImplementation Details:")
    info = evaluator.get_implementation_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print(f"\nDemo completed successfully")
    print("=" * 50)


if __name__ == "__main__":
    demo_vector_evaluator_system()