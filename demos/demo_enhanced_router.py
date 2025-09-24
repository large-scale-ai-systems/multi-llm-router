#!/usr/bin/env python3
"""
Enhanced Vector Router Demonstration

This script demonstrates the advanced features of the VectorDBEnhancedRouter
including predictive LLM selection, context-aware routing, and multi-objective optimization.
"""

from routers.vector_enhanced_router import VectorDBEnhancedRouter
from main import create_human_eval_set, smart_ingest_evaluation_set
import json


def demonstrate_enhanced_router():
    """Demonstrate advanced features of the enhanced vector router"""
    
    print("=== Enhanced Vector Router Demonstration ===")
    print()
    
    # Configure LLMs with different characteristics including cost factors
    llm_configs = {
        "gpt4_turbo": {
            "base_response_time": 2.0,
            "quality_factor": 0.9,
            "error_rate": 0.02,
            "cost_factor": 3.0  # Higher cost
        },
        "claude3_sonnet": {
            "base_response_time": 1.5,
            "quality_factor": 0.85,
            "error_rate": 0.015,
            "cost_factor": 2.5
        },
        "gemini_pro": {
            "base_response_time": 1.2,
            "quality_factor": 0.8,
            "error_rate": 0.025,
            "cost_factor": 2.0
        },
        "llama2_70b": {
            "base_response_time": 3.0,
            "quality_factor": 0.75,
            "error_rate": 0.03,
            "cost_factor": 1.0  # Lowest cost
        }
    }
    
    # Initialize enhanced router
    router = VectorDBEnhancedRouter(
        llm_configs=llm_configs,
        base_allocation={
            "gpt4_turbo": 0.3,
            "claude3_sonnet": 0.3, 
            "gemini_pro": 0.25,
            "llama2_70b": 0.15
        },
        vector_db_config={
            "index_path": "./vector_indexes",
            "similarity_threshold": 0.7
        }
    )
    
    # Load and ingest evaluation data
    print("Loading human evaluation set...")
    eval_set = create_human_eval_set()
    smart_ingest_evaluation_set(router.vector_db_evaluator, eval_set)
    print()
    
    # Test queries with different categories
    test_scenarios = [
        {
            "prompt": "Explain machine learning algorithms in detail",
            "expected_category": "technical_explanation"
        },
        {
            "prompt": "Write a Python function for binary search",
            "expected_category": "programming"
        },
        {
            "prompt": "What are best practices for handling missing data in datasets?",
            "expected_category": "data_science"
        },
        {
            "prompt": "How do REST APIs work with HTTP methods?",
            "expected_category": "web_development"
        }
    ]
    
    print("=== Testing Predictive LLM Selection ===")
    for i, scenario in enumerate(test_scenarios, 1):
        prompt = scenario["prompt"]
        print(f"\n--- Scenario {i}: {prompt[:50]}... ---")
        
        # Show quality predictions for all LLMs
        print("Quality Predictions:")
        for llm_id in router.llm_ids:
            predicted_quality = router.predict_llm_quality(prompt, llm_id)
            print(f"  {llm_id}: {predicted_quality:.3f}")
        
        # Route with enhanced selection
        response, quality_score = router.route_request_with_evaluation(prompt)
        print(f"Selected LLM: {response.llm_id}")
        print(f"Actual Quality: {quality_score:.3f}")
        print(f"Response Time: {response.generation_time:.2f}s")
    
    print("\n=== Multi-Objective Optimization Demo ===")
    
    # Test different optimization weights
    optimization_scenarios = [
        {"quality": 0.8, "speed": 0.2, "cost": 0.0, "name": "Quality-Focused"},
        {"quality": 0.2, "speed": 0.8, "cost": 0.0, "name": "Speed-Focused"},
        {"quality": 0.3, "speed": 0.3, "cost": 0.4, "name": "Cost-Conscious"},
        {"quality": 0.4, "speed": 0.3, "cost": 0.3, "name": "Balanced"}
    ]
    
    test_prompt = "Create a Python web scraping script using BeautifulSoup"
    
    for scenario in optimization_scenarios:
        print(f"\n--- {scenario['name']} Optimization ---")
        
        # Update optimization weights
        router.update_optimization_weights(
            quality_weight=scenario['quality'],
            speed_weight=scenario['speed'],
            cost_weight=scenario['cost']
        )
        
        # Test selection
        selected_llm = router.select_llm_with_vector_context(test_prompt)
        print(f"Selected LLM: {selected_llm}")
        print(f"Optimization weights: Quality={scenario['quality']}, Speed={scenario['speed']}, Cost={scenario['cost']}")
    
    print("\n=== Category Performance Analysis ===")
    
    # Analyze performance by category after some routing
    categories = ["technical_explanation", "programming", "data_science", "web_development"]
    
    for category in categories:
        recommendations = router.get_category_recommendations(category)
        print(f"\n--- {category.title()} Category ---")
        print(f"Recommended LLM: {recommendations.get('best_llm', 'No data')}")
        
        if recommendations['performance_ranking']:
            print("Performance Ranking:")
            for rank_info in recommendations['performance_ranking'][:3]:  # Top 3
                print(f"  {rank_info['llm_id']}: Quality={rank_info['avg_quality']:.3f}, "
                      f"Consistency={rank_info['consistency']:.3f}, "
                      f"Samples={rank_info['sample_count']}")
        
        print("Insights:")
        for insight in recommendations['insights']:
            print(f"  • {insight}")
    
    print("\n=== Enhanced Statistics ===")
    
    # Get comprehensive statistics
    stats = router.get_enhanced_stats()
    
    # Print key metrics
    print("Vector Database Stats:")
    vdb_stats = stats['vector_db_stats']
    print(f"  Implementation: {vdb_stats['implementation']}")
    print(f"  Evaluation Records: {vdb_stats['total_eval_records']}")
    print(f"  Similarity Threshold: {vdb_stats['similarity_threshold']}")
    
    if 'predictive_routing_stats' in stats:
        pred_stats = stats['predictive_routing_stats']
        print(f"\nPredictive Routing Stats:")
        print(f"  Cached Predictions: {pred_stats['quality_predictions_cached']}")
        print(f"  Successful Patterns: {pred_stats['successful_patterns_learned']}")
        print(f"  Cache Efficiency: {pred_stats['cache_efficiency']:.2f}")
        print(f"  Categories with Data: {len(pred_stats['category_insights'])}")
    
    if 'optimization_insights' in stats:
        opt_stats = stats['optimization_insights']
        print(f"\nOptimization Insights:")
        print(f"  Overall Quality Score: {opt_stats['overall_quality_score']:.3f}")
        print(f"  Overall Speed Factor: {opt_stats['overall_speed_factor']:.3f}")
        print(f"  Quality-Speed Tradeoff: {opt_stats['quality_speed_tradeoff']:.3f}")
    
    print("\n=== Demonstration Complete ===")
    print("The enhanced router successfully demonstrates:")
    print("Predictive LLM quality scoring")
    print("Context-aware routing based on prompt similarity")
    print("Multi-objective optimization (quality/speed/cost)")
    print("Category-specific performance tracking")
    print("Advanced caching and learning mechanisms")
    print("Comprehensive analytics and insights")


if __name__ == "__main__":
    demonstrate_enhanced_router()