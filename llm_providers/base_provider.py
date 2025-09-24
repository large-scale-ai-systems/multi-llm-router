"""
Abstract Base Class for LLM Provider Integration.

This module defines the base interface that all LLM providers must implement
for consistent integration with the routing system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import time
from enum import Enum


class LLMProviderType(Enum):
    """Enumeration of supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


@dataclass
class LLMRequest:
    """Standardized request format for all LLM providers"""
    prompt: str
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """Standardized response format from all LLM providers"""
    content: str
    model_name: str
    provider: str
    generation_time: float
    token_count: int
    total_cost: Optional[float] = None
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary format"""
        return {
            "content": self.content,
            "model_name": self.model_name,
            "provider": self.provider,
            "generation_time": self.generation_time,
            "token_count": self.token_count,
            "total_cost": self.total_cost,
            "finish_reason": self.finish_reason,
            "metadata": self.metadata or {}
        }


@dataclass
class ModelInfo:
    """Information about a specific LLM model"""
    name: str
    provider: str
    max_tokens: int
    input_cost_per_token: float
    output_cost_per_token: float
    context_window: int
    capabilities: List[str]
    description: str


class LLMProviderError(Exception):
    """Base exception for LLM provider errors"""
    def __init__(self, message: str, provider: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.provider = provider
        self.error_code = error_code


class RateLimitError(LLMProviderError):
    """Raised when rate limits are exceeded"""
    pass


class AuthenticationError(LLMProviderError):
    """Raised when authentication fails"""
    pass


class ModelNotFoundError(LLMProviderError):
    """Raised when requested model is not available"""
    pass


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    
    All concrete implementations must inherit from this class and implement
    the abstract methods to ensure consistent behavior across providers.
    """
    
    def __init__(self, api_key: str, model_name: str, **kwargs):
        """
        Initialize the LLM provider.
        
        Args:
            api_key: API key for the provider
            model_name: Name of the model to use
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.model_name = model_name
        self.config = kwargs
        self._request_count = 0
        self._total_tokens = 0
        self._total_cost = 0.0
    
    @abstractmethod
    def generate_response(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            request: Standardized LLM request object
            
        Returns:
            LLMResponse: Standardized response object
            
        Raises:
            LLMProviderError: For provider-specific errors
            RateLimitError: When rate limits are exceeded
            AuthenticationError: When authentication fails
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """
        Get information about the current model.
        
        Returns:
            ModelInfo: Detailed model information
        """
        pass
    
    @abstractmethod
    def validate_api_key(self) -> bool:
        """
        Validate the API key with the provider.
        
        Returns:
            bool: True if API key is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def estimate_cost(self, request: LLMRequest) -> Tuple[float, float]:
        """
        Estimate the cost of a request.
        
        Args:
            request: LLM request to estimate cost for
            
        Returns:
            Tuple[float, float]: (input_cost, estimated_total_cost)
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Get list of available models from the provider.
        
        Returns:
            List[str]: List of available model names
        """
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the provider.
        
        Returns:
            Dict[str, Any]: Health status information
        """
        pass
    
    # Common utility methods (implemented in base class)
    
    def get_provider_type(self) -> LLMProviderType:
        """Get the provider type enum"""
        # This will be overridden in concrete implementations
        raise NotImplementedError("Subclasses must implement get_provider_type")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics for this provider instance.
        
        Returns:
            Dict[str, Any]: Statistics including request count, tokens, cost
        """
        return {
            "provider": self.get_provider_type().value,
            "model_name": self.model_name,
            "request_count": self._request_count,
            "total_tokens": self._total_tokens,
            "total_cost": self._total_cost,
            "avg_tokens_per_request": (
                self._total_tokens / self._request_count 
                if self._request_count > 0 else 0
            ),
            "avg_cost_per_request": (
                self._total_cost / self._request_count 
                if self._request_count > 0 else 0
            )
        }
    
    def reset_stats(self) -> None:
        """Reset usage statistics"""
        self._request_count = 0
        self._total_tokens = 0
        self._total_cost = 0.0
    
    def _update_stats(self, response: LLMResponse) -> None:
        """Update internal statistics after a successful request"""
        self._request_count += 1
        self._total_tokens += response.token_count
        if response.total_cost:
            self._total_cost += response.total_cost
    
    def _handle_provider_error(self, error: Exception) -> LLMProviderError:
        """
        Convert provider-specific errors to standardized errors.
        
        This method should be overridden in concrete implementations
        to handle provider-specific error types.
        
        Args:
            error: Original exception from the provider
            
        Returns:
            LLMProviderError: Standardized error
        """
        return LLMProviderError(
            message=str(error),
            provider=self.get_provider_type().value
        )
    
    def __str__(self) -> str:
        """String representation of the provider"""
        return f"{self.get_provider_type().value.title()}Provider(model={self.model_name})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the provider"""
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.model_name}', "
            f"requests={self._request_count}, "
            f"tokens={self._total_tokens}"
            f")"
        )