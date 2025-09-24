"""
Demo script for LLM Provider Integration.

This script demonstrates how to use the new LLM p    print("\nAdding providers to manager:")ovider system
with real integrations for GPT-4o, Claude Sonnet, and Gemini.
"""

import os
import time
from typing import Dict, Any

from llm_providers import (
    create_provider,
    ProviderFactory, 
    ProviderManager,
    LLMRequest,
    get_available_providers,
    get_provider_models
)


def demo_basic_usage():
    """Demonstrate basic provider usage"""
    print("=" * 60)
    print("LLM Provider Integration Demo")
    print("=" * 60)
    
    # Show available providers and models
    print("\nAvailable Providers:")
    providers = get_available_providers()
    for provider in providers:
        models = get_provider_models(provider)
        print(f"  • {provider.title()}: {', '.join(models[:2])} (+ {len(models)-2 if len(models) > 2 else 0} more)")
    
    print("\n" + "="*60)
    print("NOTE: This demo uses placeholder implementations.")
    print("To use real APIs, you need to:")
    print("1. Install required packages: pip install openai anthropic google-generativeai")
    print("2. Set API keys as environment variables")
    print("3. Uncomment API calls in provider implementations")
    print("="*60)


def demo_provider_creation():
    """Demonstrate provider creation with different methods"""
    print("\n🏭 Provider Creation Demo")
    print("-" * 40)
    
    # Demo API keys (placeholder)
    demo_keys = {
        "openai": "sk-demo-key-openai-placeholder-12345",
        "anthropic": "sk-demo-key-anthropic-placeholder-12345", 
        "google": "demo-key-google-placeholder-12345"
    }
    
    # Method 1: Using factory function
    print("\n1️⃣ Using factory function:")
    try:
        gpt_provider = create_provider("openai", demo_keys["openai"], "gpt-4o")
        print(f"   Created: {gpt_provider}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Method 2: Using ProviderFactory class
    print("\n2. Using ProviderFactory class:")
    try:
        claude_provider = ProviderFactory.create_provider("anthropic", demo_keys["anthropic"])
        print(f"   Created: {claude_provider}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Method 3: Using specific factory methods
    print("\n3. Using specific factory methods:")
    try:
        gemini_provider = ProviderFactory.create_google_provider(
            demo_keys["google"], 
            "gemini-1.5-flash",
            timeout=30
        )
        print(f"   Created: {gemini_provider}")
    except Exception as e:
        print(f"   Error: {e}")
def demo_provider_manager():
    """Demonstrate provider management capabilities"""
    print("\n👨‍💼 Provider Manager Demo")
    print("-" * 40)
    
    # Demo API keys
    demo_keys = {
        "openai": "sk-demo-key-openai-placeholder-12345",
        "anthropic": "sk-demo-key-anthropic-placeholder-12345",
        "google": "demo-key-google-placeholder-12345"
    }
    
    # Create manager and add providers
    manager = ProviderManager()
    
    print("\n📝 Adding providers to manager:")
    try:
        # Add multiple providers
        manager.add_provider("gpt4o", "openai", demo_keys["openai"], "gpt-4o")
        manager.add_provider("claude", "anthropic", demo_keys["anthropic"])
        manager.add_provider("gemini", "google", demo_keys["google"], "gemini-1.5-pro")
        
        print(f"   Added providers: {', '.join(manager.list_providers())}")
        
        # Health check all providers
        print("\n🏥 Health check results:")
        health_results = manager.health_check_all()
        for name, health in health_results.items():
            status = health.get("status", "unknown")
            provider_type = health.get("provider", "unknown")
            print(f"   • {name} ({provider_type}): {status.upper()}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")


def demo_request_generation():
    """Demonstrate request generation and response handling"""
    print("\n💬 Request Generation Demo")
    print("-" * 40)
    
    # Create a sample request
    sample_request = LLMRequest(
        prompt="Explain the concept of machine learning in simple terms.",
        max_tokens=150,
        temperature=0.7,
        system_prompt="You are a helpful AI assistant focused on clear explanations."
    )
    
    print(f"\n📤 Sample Request:")
    print(f"   Prompt: {sample_request.prompt[:50]}...")
    print(f"   Max Tokens: {sample_request.max_tokens}")
    print(f"   Temperature: {sample_request.temperature}")
    
    # Demo provider creation and response generation
    demo_keys = {"openai": "sk-demo-key-openai-placeholder-12345"}
    
    try:
        # Create provider
        provider = create_provider("openai", demo_keys["openai"], "gpt-4o")
        print(f"\nUsing provider: {provider}")
        
        # Generate response (this will use mock implementation)
        print("\nGenerating response...")
        start_time = time.time()
        response = provider.generate_response(sample_request)
        elapsed = time.time() - start_time
        
        print(f"\n📥 Response received:")
        print(f"   Content: {response.content[:100]}...")
        print(f"   Model: {response.model_name}")
        print(f"   Provider: {response.provider}")
        print(f"   Generation Time: {response.generation_time:.2f}s")
        print(f"   Tokens: {response.token_count}")
        print(f"   Cost: ${response.total_cost:.4f}")
        print(f"   Finish Reason: {response.finish_reason}")
        
        # Show provider stats
        print(f"\nProvider Statistics:")
        stats = provider.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"   Error generating response: {e}")


def demo_model_information():
    """Demonstrate model information retrieval"""
    print("\n📋 Model Information Demo")
    print("-" * 40)
    
    demo_keys = {
        "openai": "sk-demo-key-openai-placeholder-12345",
        "anthropic": "sk-demo-key-anthropic-placeholder-12345",
        "google": "demo-key-google-placeholder-12345"
    }
    
    providers_to_test = [
        ("openai", "gpt-4o"),
        ("anthropic", "claude-3-5-sonnet-20241022"),
        ("google", "gemini-1.5-pro")
    ]
    
    for provider_type, model_name in providers_to_test:
        try:
            provider = create_provider(provider_type, demo_keys[provider_type], model_name)
            model_info = provider.get_model_info()
            
            print(f"\n{model_info.name} ({model_info.provider}):")
            print(f"   Context Window: {model_info.context_window:,} tokens")
            print(f"   Max Output: {model_info.max_tokens:,} tokens")
            print(f"   Input Cost: ${model_info.input_cost_per_token*1000:.3f}/1K tokens")
            print(f"   Output Cost: ${model_info.output_cost_per_token*1000:.3f}/1K tokens")
            print(f"   Capabilities: {', '.join(model_info.capabilities[:3])} (+{len(model_info.capabilities)-3 if len(model_info.capabilities) > 3 else 0})")
            
        except Exception as e:
            print(f"   Error getting model info for {model_name}: {e}")


def demo_cost_estimation():
    """Demonstrate cost estimation capabilities"""
    print("\n💰 Cost Estimation Demo") 
    print("-" * 40)
    
    # Sample requests of different sizes
    test_requests = [
        LLMRequest(prompt="Short question about AI?", max_tokens=50),
        LLMRequest(prompt="Write a detailed essay about machine learning. " * 20, max_tokens=500),
        LLMRequest(prompt="Create a comprehensive analysis. " * 50, max_tokens=2000)
    ]
    
    demo_key = "sk-demo-key-openai-placeholder-12345"
    
    try:
        provider = create_provider("openai", demo_key, "gpt-4o")
        
        print(f"\n💲 Cost estimates for {provider.model_name}:")
        for i, request in enumerate(test_requests, 1):
            input_cost, total_cost = provider.estimate_cost(request)
            print(f"   Request {i}: ~${total_cost:.4f} (input: ${input_cost:.4f})")
            
    except Exception as e:
        print(f"   Error estimating costs: {e}")


def main():
    """Main demo function"""
    print("Starting LLM Provider Integration Demo...")
    
    # Run all demos
    demo_basic_usage()
    demo_provider_creation()
    demo_provider_manager()
    demo_request_generation()
    demo_model_information()
    demo_cost_estimation()
    
    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("\nNext Steps:")
    print("1. Set up real API keys in environment variables:")
    print("   - OPENAI_API_KEY")
    print("   - ANTHROPIC_API_KEY") 
    print("   - GOOGLE_API_KEY")
    print("2. Install required packages:")
    print("   pip install openai anthropic google-generativeai")
    print("3. Uncomment real API calls in provider implementations")
    print("4. Integrate with your existing router system")
    print("="*60)


if __name__ == "__main__":
    main()