"""
Demo: Multi-LLM Comparison System
Showcases the enhanced router capabilities with comprehensive LLM comparison
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from routers.vector_enhanced_router import VectorDBEnhancedRouter
from core.data_models import LLMResponse

def demo_multi_llm_comparison():
    """Demonstrate multi-LLM comparison functionality"""
    print("Multi-LLM Comparison System Demo")
    print("="*50)
    
    # Create enhanced router
    llm_configs = {
        'gpt-4o': {'model': 'gpt-4o', 'provider': 'openai'},
        'gpt-4o-mini': {'model': 'gpt-4o-mini', 'provider': 'openai'},
        'claude-3.5-sonnet': {'model': 'claude-3.5-sonnet', 'provider': 'anthropic'}
    }
    
    base_allocation = {'gpt-4o': 0.4, 'gpt-4o-mini': 0.3, 'claude-3.5-sonnet': 0.3}
    
    router = VectorDBEnhancedRouter(
        llm_configs=llm_configs,
        base_allocation=base_allocation,
        use_real_providers=False  # Using mock for demo
    )
    
    print(f"✓ Router initialized with {len(router.llm_configs)} LLMs")
    print(f"✓ Vector evaluator ready with {len(router.vector_db_evaluator.human_eval_records)} evaluation records")
    
    # Demo prompts with different characteristics
    demo_prompts = [
        {
            'prompt': 'What is machine learning?',
            'context': 'Simple question - should demonstrate fast vs comprehensive trade-offs'
        },
        {
            'prompt': 'Analyze the economic, environmental, and social implications of transitioning to renewable energy sources. Provide detailed recommendations for policymakers considering implementation strategies.',
            'context': 'Complex analysis - should trigger comprehensive mode'
        },
        {
            'prompt': 'Compare Python and JavaScript for web development',
            'context': 'Comparison task - ideal for multi-LLM evaluation'
        }
    ]
    
    for i, demo_case in enumerate(demo_prompts, 1):
        print(f"\n{'='*20} Demo Case {i} {'='*20}")
        print(f"Context: {demo_case['context']}")
        print(f"Prompt: {demo_case['prompt']}")
        
        # Step 1: Get routing recommendations
        print(f"\nRouting Analysis:")
        recommendations = router.get_routing_mode_recommendations(demo_case['prompt'])
        print(f"Recommended mode: {recommendations['recommended_mode']}")
        print(f"Reasoning: {'; '.join(recommendations['analysis']['reasoning'])}")
        
        # Step 2: Compare single vs multi-LLM approaches
        print(f"\n🔄 Comparing Approaches:")
        
        # Single LLM mode
        print("1. Single LLM Selection Mode:")
        single_response, single_quality = router.route_request_with_evaluation(demo_case['prompt'])
        print(f"   Selected: {single_response.llm_id}")
        print(f"   Quality: {single_quality:.3f}")
        print(f"   Response preview: {single_response.content[:100]}...")
        
        # Multi-LLM mode
        print("\n2. Multi-LLM Comparison Mode:")
        multi_response, multi_quality, multi_details = router.route_request_with_multi_llm_evaluation(demo_case['prompt'])
        print(f"   Selected: {multi_response.llm_id}")
        print(f"   Quality: {multi_quality:.3f}")
        print(f"   Responses compared: {len(multi_details['all_responses'])}")
        print(f"   Response preview: {multi_response.content[:100]}...")
        
        # Show comparison details
        print(f"\n📈 Detailed Comparison:")
        for llm_name, evaluation in multi_details['all_evaluations'].items():
            print(f"   {llm_name}:")
            print(f"     Golden similarity: {evaluation['golden_similarity']:.3f}")
            print(f"     Relative rank: {evaluation['relative_rank']:.3f}")
            print(f"     Uniqueness: {evaluation.get('uniqueness', 0):.3f}")
            print(f"     Time efficiency: {evaluation['time_efficiency']:.3f}")
        
        # Show selection reasoning
        print(f"\nSelection Reasoning:")
        if 'comparative_analysis' in multi_details:
            analysis = multi_details['comparative_analysis']
            print(f"Best performers by category:")
            for category, info in analysis.get('best_in_category', {}).items():
                print(f"   {category}: {info['llm']} ({info['score']:.3f})")
            
            print(f"Recommendations:")
            for rec in analysis.get('recommendations', []):
                print(f"   • {rec}")

def demo_learning_adaptation():
    """Demonstrate learning and adaptation capabilities"""
    print(f"\n{'='*20} Learning & Adaptation Demo {'='*20}")
    
    llm_configs = {
        'gpt-4o': {'model': 'gpt-4o', 'provider': 'openai'},
        'gpt-4o-mini': {'model': 'gpt-4o-mini', 'provider': 'openai'},
        'claude-3.5-sonnet': {'model': 'claude-3.5-sonnet', 'provider': 'anthropic'}
    }
    
    base_allocation = {'gpt-4o': 0.4, 'gpt-4o-mini': 0.3, 'claude-3.5-sonnet': 0.3}
    
    router = VectorDBEnhancedRouter(
        llm_configs=llm_configs,
        base_allocation=base_allocation,
        use_real_providers=False
    )
    
    print("📚 Training the system with multiple evaluations...")
    
    # Simulate learning with various prompts
    training_prompts = [
        "Explain cloud computing benefits",
        "What is artificial intelligence?",
        "How does blockchain work?",
        "Describe cybersecurity best practices",
        "What are the advantages of microservices?",
        "Explain data analytics importance",
        "How does DevOps improve software delivery?",
        "What is edge computing?"
    ]
    
    print(f"Initial allocations:")
    for llm_id, allocation in router.current_allocation.items():
        print(f"   {llm_id}: {allocation:.3f}")
    
    # Run training evaluations
    for i, prompt in enumerate(training_prompts, 1):
        print(f"\nTraining iteration {i}: {prompt}")
        try:
            response, quality, details = router.route_request_with_multi_llm_evaluation(prompt)
            print(f"   Selected: {response.llm_id} (quality: {quality:.3f})")
        except Exception as e:
            print(f"   Error: {e}")
    
    # Show performance insights
    print(f"\nPerformance Insights:")
    insights = router.get_multi_llm_performance_insights()
    
    if 'llm_performance' in insights:
        for llm_id, performance in insights['llm_performance'].items():
            print(f"   {llm_id}:")
            avg_quality = performance.get('avg_golden_similarity', 0)
            samples = performance.get('samples_golden_similarity', 0)
            print(f"     Average quality: {avg_quality:.3f} ({samples} samples)")
    
    # Show recommendations
    if 'recommendations' in insights:
        print(f"\n💡 System Recommendations:")
        for rec in insights['recommendations']:
            print(f"   • {rec}")
    
    # Demonstrate adaptation
    print(f"\n🔄 Adapting allocations based on performance...")
    router.adapt_allocation_from_multi_llm_performance()
    
    print(f"Updated allocations:")
    for llm_id, allocation in router.current_allocation.items():
        print(f"   {llm_id}: {allocation:.3f}")

def demo_adaptive_routing():
    """Demonstrate adaptive routing with different modes"""
    print(f"\n{'='*20} Adaptive Routing Demo {'='*20}")
    
    llm_configs = {
        'gpt-4o': {'model': 'gpt-4o', 'provider': 'openai'},
        'gpt-4o-mini': {'model': 'gpt-4o-mini', 'provider': 'openai'},
        'claude-3.5-sonnet': {'model': 'claude-3.5-sonnet', 'provider': 'anthropic'}
    }
    
    base_allocation = {'gpt-4o': 0.4, 'gpt-4o-mini': 0.3, 'claude-3.5-sonnet': 0.3}
    
    router = VectorDBEnhancedRouter(
        llm_configs=llm_configs,
        base_allocation=base_allocation,
        use_real_providers=False
    )
    
    # Test different routing modes
    test_prompt = "Explain the impact of AI on future job markets"
    
    modes = ['auto', 'fast', 'comprehensive']
    
    for mode in modes:
        print(f"\nTesting mode: {mode}")
        try:
            response, quality, details = router.route_request_adaptive(test_prompt, mode=mode)
            print(f"   Mode used: {details['mode']}")
            print(f"   Selected LLM: {response.llm_id}")
            print(f"   Quality: {quality:.3f}")
            print(f"   Reason: {details.get('reason', 'Multi-LLM comparison')}")
        except Exception as e:
            print(f"   Error: {e}")

def main():
    """Run the complete demo"""
    print("🌟 Multi-LLM Comparison System - Complete Demo")
    print("="*60)
    print("This demo showcases the enhanced router with:")
    print("• Multi-LLM response generation and comparison")
    print("• Intelligent best response selection")
    print("• Learning and adaptation from performance data")
    print("• Adaptive routing modes (fast vs comprehensive)")
    print("• Performance insights and recommendations")
    
    try:
        demo_multi_llm_comparison()
        demo_learning_adaptation()
        demo_adaptive_routing()
        
        print(f"\n{'='*60}")
        print("Demo completed successfully!")
        print("The Multi-LLM Comparison System demonstrates:")
        print("✓ Comprehensive LLM response comparison")
        print("✓ Intelligent selection based on multiple criteria")
        print("✓ Continuous learning and adaptation")
        print("✓ Flexible routing modes for different use cases")
        print("✓ Detailed performance insights and recommendations")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    print(f"\nDemo {'completed successfully' if success else 'failed'}!")