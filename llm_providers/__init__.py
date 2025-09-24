"""
LLM Providers Package.

This package provides a unified interface for integrating multiple LLM providers
including OpenAI, Anthropic, and Google with the routing system.
"""

# Base classes and types
from .base_provider import (
    BaseLLMProvider,
    LLMRequest,
    LLMResponse,
    ModelInfo,
    LLMProviderType,
    LLMProviderError,
    RateLimitError,
    AuthenticationError,
    ModelNotFoundError
)

# Provider implementations
from .openai import AzureOpenAIProvider, create_openai_provider, get_openai_models
from .anthropic import AnthropicProvider, create_anthropic_provider, get_anthropic_models
from .google import GoogleProvider, create_google_provider, get_google_models

# Factory and management
from .factory import (
    ProviderFactory,
    ProviderManager,
    create_provider,
    get_available_providers,
    get_provider_models
)

__version__ = "1.0.0"

__all__ = [
    # Base classes and types
    "BaseLLMProvider",
    "LLMRequest", 
    "LLMResponse",
    "ModelInfo",
    "LLMProviderType",
    "LLMProviderError",
    "RateLimitError",
    "AuthenticationError", 
    "ModelNotFoundError",
    
    # Provider implementations
    "AzureOpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    
    # Factory functions
    "create_openai_provider",
    "create_anthropic_provider",
    "create_google_provider",
    
    # Model information
    "get_openai_models",
    "get_anthropic_models", 
    "get_google_models",
    
    # Factory and management
    "ProviderFactory",
    "ProviderManager",
    "create_provider",
    "get_available_providers",
    "get_provider_models"
]

# Package metadata
SUPPORTED_PROVIDERS = ["openai", "anthropic", "google"]
SUPPORTED_MODELS = {
    "openai": ["gpt-4o", "gpt-4o-mini"],
    "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
    "google": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]
}