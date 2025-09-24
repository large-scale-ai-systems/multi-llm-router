"""
LLM Provider Factory for easy instantiation and management.

This module provides factory methods and utilities for creating and managing
LLM provider instances across different providers.
"""

from typing import Dict, Any, Optional, List, Union
from enum import Enum

from .base_provider import BaseLLMProvider, LLMProviderType, LLMProviderError
from .openai import AzureOpenAIProvider, create_openai_provider
from .anthropic import AnthropicProvider, create_anthropic_provider  
from .google import GoogleProvider, create_google_provider


class ProviderFactory:
    """
    Factory class for creating LLM providers.
    
    Provides a unified interface for instantiating different LLM providers
    with consistent configuration and error handling.
    """
    
    # Registry of available providers
    PROVIDERS = {
        LLMProviderType.OPENAI: {
            "class": AzureOpenAIProvider,
            "factory": create_openai_provider,
            "default_models": ["gpt-4o", "gpt-4o-mini"]
        },
        LLMProviderType.ANTHROPIC: {
            "class": AnthropicProvider,
            "factory": create_anthropic_provider,
            "default_models": ["claude-3-5-sonnet-20241022", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
        },
        LLMProviderType.GOOGLE: {
            "class": GoogleProvider,
            "factory": create_google_provider,
            "default_models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]
        }
    }
    
    @classmethod
    def create_provider(
        cls,
        provider_type: Union[str, LLMProviderType],
        api_key: str,
        model_name: Optional[str] = None,
        **kwargs
    ) -> BaseLLMProvider:
        """
        Create a provider instance.
        
        Args:
            provider_type: Type of provider ("openai", "anthropic", "google" or enum)
            api_key: API key for the provider
            model_name: Model to use (uses default if not specified)
            **kwargs: Additional provider-specific configuration
            
        Returns:
            BaseLLMProvider: Configured provider instance
            
        Raises:
            LLMProviderError: If provider type is not supported
        """
        # Convert string to enum if needed
        if isinstance(provider_type, str):
            try:
                provider_type = LLMProviderType(provider_type.lower())
            except ValueError:
                raise LLMProviderError(
                    f"Unsupported provider type: {provider_type}. "
                    f"Available: {[p.value for p in LLMProviderType]}",
                    provider=provider_type
                )
        
        if provider_type not in cls.PROVIDERS:
            raise LLMProviderError(
                f"Provider {provider_type.value} not registered",
                provider=provider_type.value
            )
        
        provider_info = cls.PROVIDERS[provider_type]
        
        # Use default model if none specified
        if not model_name:
            model_name = provider_info["default_models"][0]
        
        # Create provider using factory method
        return provider_info["factory"](api_key, model_name, **kwargs)
    
    @classmethod
    def create_openai_provider(cls, api_key: str = "azure", model_name: str = "gpt-4o", **kwargs) -> AzureOpenAIProvider:
        """Create OpenAI provider with specific configuration."""
        return create_openai_provider(api_key, model_name, **kwargs)
    
    @classmethod
    def create_anthropic_provider(cls, api_key: str, model_name: str = "claude-3-5-sonnet-20241022", **kwargs) -> AnthropicProvider:
        """Create Anthropic provider with specific configuration."""
        return create_anthropic_provider(api_key, model_name, **kwargs)
    
    @classmethod
    def create_google_provider(cls, api_key: str, model_name: str = "gemini-1.5-pro", **kwargs) -> GoogleProvider:
        """Create Google provider with specific configuration."""
        return create_google_provider(api_key, model_name, **kwargs)
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """
        Get list of available provider types.
        
        Returns:
            List[str]: Available provider names
        """
        return [provider.value for provider in cls.PROVIDERS.keys()]
    
    @classmethod
    def get_provider_models(cls, provider_type: Union[str, LLMProviderType]) -> List[str]:
        """
        Get available models for a provider.
        
        Args:
            provider_type: Provider type
            
        Returns:
            List[str]: Available model names
            
        Raises:
            LLMProviderError: If provider not found
        """
        if isinstance(provider_type, str):
            try:
                provider_type = LLMProviderType(provider_type.lower())
            except ValueError:
                raise LLMProviderError(
                    f"Unknown provider: {provider_type}",
                    provider=provider_type
                )
        
        if provider_type not in cls.PROVIDERS:
            raise LLMProviderError(
                f"Provider {provider_type.value} not registered",
                provider=provider_type.value
            )
        
        return cls.PROVIDERS[provider_type]["default_models"].copy()
    
    @classmethod
    def get_provider_info(cls, provider_type: Union[str, LLMProviderType]) -> Dict[str, Any]:
        """
        Get information about a provider.
        
        Args:
            provider_type: Provider type
            
        Returns:
            Dict[str, Any]: Provider information
        """
        if isinstance(provider_type, str):
            provider_type = LLMProviderType(provider_type.lower())
        
        if provider_type not in cls.PROVIDERS:
            raise LLMProviderError(
                f"Provider {provider_type.value} not registered",
                provider=provider_type.value
            )
        
        provider_info = cls.PROVIDERS[provider_type]
        return {
            "type": provider_type.value,
            "class_name": provider_info["class"].__name__,
            "default_models": provider_info["default_models"],
            "total_models": len(provider_info["default_models"])
        }


class ProviderManager:
    """
    Manager class for handling multiple provider instances.
    
    Provides utilities for managing multiple providers, health checks,
    and coordinated operations across providers.
    """
    
    def __init__(self):
        """Initialize the provider manager."""
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.provider_configs: Dict[str, Dict[str, Any]] = {}
    
    def add_provider(
        self,
        name: str,
        provider_type: Union[str, LLMProviderType],
        api_key: str,
        model_name: Optional[str] = None,
        **kwargs
    ) -> BaseLLMProvider:
        """
        Add a provider instance to the manager.
        
        Args:
            name: Unique name for this provider instance
            provider_type: Type of provider
            api_key: API key
            model_name: Model name
            **kwargs: Additional configuration
            
        Returns:
            BaseLLMProvider: Created provider instance
        """
        provider = ProviderFactory.create_provider(
            provider_type, api_key, model_name, **kwargs
        )
        
        self.providers[name] = provider
        self.provider_configs[name] = {
            "provider_type": provider_type,
            "model_name": model_name or "default",
            "config": kwargs
        }
        
        return provider
    
    def get_provider(self, name: str) -> Optional[BaseLLMProvider]:
        """Get a provider by name."""
        return self.providers.get(name)
    
    def remove_provider(self, name: str) -> bool:
        """Remove a provider from the manager."""
        if name in self.providers:
            del self.providers[name]
            del self.provider_configs[name]
            return True
        return False
    
    def list_providers(self) -> List[str]:
        """Get list of managed provider names."""
        return list(self.providers.keys())
    
    def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Perform health check on all managed providers.
        
        Returns:
            Dict[str, Dict[str, Any]]: Health status for each provider
        """
        results = {}
        for name, provider in self.providers.items():
            try:
                results[name] = provider.health_check()
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "error": str(e),
                    "provider": provider.get_provider_type().value
                }
        return results
    
    def get_stats_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Get usage statistics for all providers.
        
        Returns:
            Dict[str, Dict[str, Any]]: Statistics for each provider
        """
        return {name: provider.get_stats() for name, provider in self.providers.items()}
    
    def reset_stats_all(self) -> None:
        """Reset statistics for all providers."""
        for provider in self.providers.values():
            provider.reset_stats()


# Convenience functions
def create_provider(provider_type: str, api_key: str, model_name: str = None, **kwargs) -> BaseLLMProvider:
    """
    Convenience function to create a provider.
    
    Args:
        provider_type: Provider type ("openai", "anthropic", "google")
        api_key: API key
        model_name: Model name
        **kwargs: Additional configuration
        
    Returns:
        BaseLLMProvider: Provider instance
    """
    return ProviderFactory.create_provider(provider_type, api_key, model_name, **kwargs)


def get_available_providers() -> List[str]:
    """Get list of available provider types."""
    return ProviderFactory.get_available_providers()


def get_provider_models(provider_type: str) -> List[str]:
    """Get available models for a provider type."""
    return ProviderFactory.get_provider_models(provider_type)