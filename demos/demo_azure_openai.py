#!/usr/bin/env python3
"""
Demo script showing Azure OpenAI GPT-4o provider capabilities.

This demonstrates the provider interface and functionality for development purposes.
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_providers.openai.gpt4o_provider import AzureOpenAIProvider, get_openai_models


def demo_azure_openai_interface():
    """Demonstrate the Azure OpenAI provider interface."""
    
    print("Azure OpenAI GPT-4o Provider Interface Demo")
    print("=" * 55)
    
    try:
        print("Available Models:")
        models = get_openai_models()
        for i, (model_name, config) in enumerate(models.items(), 1):
            print(f"   {i}. {model_name}")
            print(f"      Context Window: {config['context_window']:,} tokens")
            print(f"      Max Tokens: {config['max_tokens']:,}")
            print(f"      Input Cost: ${config['input_cost_per_million']}/1M tokens")
            print(f"      Output Cost: ${config['output_cost_per_million']}/1M tokens")
            print(f"      Capabilities: {', '.join(config['capabilities'])}")
            print()
        
        print("Provider Initialization:")
        print("   - Loads Azure OpenAI credentials from config.ini")
        print("   - Supports GPT-4o and GPT-4o-mini models")
        print("   - Implements full LLMProvider interface")
        print("   - Automatic cost calculation and token tracking")
        print("   - Uses Azure OpenAI Service endpoints")
        print()
        
        print("Key Features:")
        print("   - Azure OpenAI Service integration")
        print("   - Multi-model support (GPT-4o, GPT-4o-mini)")
        print("   - Configurable via INI files")
        print("   - Comprehensive error handling")
        print("   - Cost estimation and tracking")
        print("   - Health checks and connection testing")
        print("   - System prompt support")
        print("   - Vision capabilities (GPT-4o)")
        print("   - JSON mode and function calling")
        print()
        
        print("🔌 Connection Requirements:")
        print("   1. Valid Azure subscription with OpenAI resource")
        print("   2. Azure OpenAI resource with GPT-4o deployment")
        print("   3. API key and endpoint from Azure OpenAI resource")
        print("   4. Proper deployment name configuration")
        print()
        
        print("💼 Usage Example:")
        print("   ```python")
        print("   from llm_providers.openai.gpt4o_provider import AzureOpenAIProvider")
        print("   from llm_providers.base_provider import LLMRequest")
        print()
        print("   # Initialize provider (uses config.ini)")
        print("   provider = AzureOpenAIProvider(api_key='azure', model_name='gpt-4o')")
        print()
        print("   # Create request")
        print("   request = LLMRequest(")
        print("       prompt='Explain machine learning in simple terms.',")
        print("       max_tokens=150,")
        print("       temperature=0.7")
        print("   )")
        print()
        print("   # Generate response")
        print("   response = provider.generate_response(request)")
        print("   print(f'Response: {response.content}')")
        print("   print(f'Cost: ${response.total_cost:.4f}')")
        print("   ```")
        print()
        
        print("Configuration Structure:")
        print("   ```ini")
        print("   [LLM_CONFIG]")
        print("   # Azure OpenAI Configuration")
        print("   azure_openai_api_key = your_api_key_here")
        print("   azure_openai_endpoint = https://your-resource.openai.azure.com/")
        print("   azure_openai_api_version = 2024-02-01")
        print("   azure_openai_deployment_name = gpt-4o")
        print("   gpt4o_max_tokens = 4096")
        print("   gpt4o_temperature = 0.7")
        print("   gpt4o_timeout = 30")
        print("   ```")
        print()
        
        print("Advanced Features:")
        print("   • System prompts for context")
        print("   • Vision input support (images with GPT-4o)")
        print("   • JSON mode for structured output")
        print("   • Function calling capabilities")
        print("   • Streaming responses (via OpenAI client)")
        print("   • Custom stop sequences")
        print("   • Temperature and top_p control")
        print()
        
        print("Provider implementation is complete and ready for production!")
        print("   Next steps: Configure Azure OpenAI credentials and test with real deployment")
        
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_azure_openai_interface()