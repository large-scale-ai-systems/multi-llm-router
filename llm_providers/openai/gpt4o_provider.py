"""
Azure OpenAI GPT-4o Provider Implementation.

This module provides integration with Azure OpenAI's GPT-4o model through Azure OpenAI Service.
"""

import time
import configparser
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json

try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None

from ..base_provider import (
    BaseLLMProvider, LLMRequest, LLMResponse, ModelInfo, LLMProviderType,
    LLMProviderError, RateLimitError, AuthenticationError, ModelNotFoundError
)


class AzureOpenAIProvider(BaseLLMProvider):
    """
    Azure OpenAI GPT-4o provider implementation.
    
    Handles communication with Azure OpenAI Service for GPT-4o model.
    """
    
    # Configuration schema for automatic parsing and validation
    CONFIG_SCHEMA = {
        # Common Azure OpenAI settings (required for all models)
        'azure_openai_api_key': {
            'required': True,
            'type': str,
            'description': 'Azure OpenAI API Key',
            'validation': lambda x: x and len(x.strip()) > 0
        },
        'azure_openai_endpoint': {
            'required': True,
            'type': str,
            'description': 'Azure OpenAI Service Endpoint URL',
            'validation': lambda x: x and x.startswith('https://')
        },
        'azure_openai_api_version': {
            'required': False,
            'type': str,
            'default': '2024-02-01',
            'description': 'Azure OpenAI API Version'
        },
        
        # Model-specific settings (dynamic keys based on model)
        '{model_key}_deployment_name': {
            'required': True,
            'type': str,
            'description': 'Azure deployment name for the model',
            'example': 'recorder-gpt-4o'
        },
        '{model_key}_max_tokens': {
            'required': False,
            'type': int,
            'default': 'model_config.max_tokens',
            'description': 'Maximum tokens for response',
            'validation': lambda x: 1 <= int(x) <= 128000
        },
        '{model_key}_temperature': {
            'required': False,
            'type': float,
            'default': 0.7,
            'description': 'Response temperature (0.0-2.0)',
            'validation': lambda x: 0.0 <= float(x) <= 2.0
        },
        '{model_key}_timeout': {
            'required': False,
            'type': int,
            'default': 30,
            'description': 'Request timeout in seconds',
            'validation': lambda x: 1 <= int(x) <= 300
        }
    }
    
    # Model configuration
    SUPPORTED_MODELS = {
        "gpt-4o": {
            "max_tokens": 128000,
            "context_window": 128000,
            "input_cost_per_million": 5.00,   # $5 per 1M input tokens
            "output_cost_per_million": 15.00, # $15 per 1M output tokens
            "capabilities": ["text", "vision", "json_mode", "function_calling"],
            "token_param": "max_tokens"  # Uses standard max_tokens parameter
        },
        "o4-mini": {
            "max_tokens": 128000,
            "context_window": 128000,
            "input_cost_per_million": 0.15,   # $0.15 per 1M input tokens
            "output_cost_per_million": 0.60,  # $0.60 per 1M output tokens
            "capabilities": ["text", "vision", "json_mode", "function_calling"],
            "token_param": "max_completion_tokens",  # Uses max_completion_tokens parameter
            "fixed_temperature": 1.0,  # Only supports default temperature of 1.0
            "restricted_params": ["top_p"]  # Parameters not supported by this model
        },
        "gpt-4o-mini": {
            "max_tokens": 128000,
            "context_window": 128000,
            "input_cost_per_million": 0.15,   # $0.15 per 1M input tokens
            "output_cost_per_million": 0.60,  # $0.60 per 1M output tokens
            "capabilities": ["text", "vision", "json_mode", "function_calling"],
            "token_param": "max_completion_tokens",  # Uses max_completion_tokens parameter
            "fixed_temperature": 1.0,  # Only supports default temperature of 1.0
            "restricted_params": ["top_p"]  # Parameters not supported by this model
        }
    }
    
    def __init__(self, api_key: str, model_name: str, **kwargs):
        """
        Initialize Azure OpenAI provider.
        
        Args:
            api_key: Azure OpenAI API key (MANDATORY)
            model_name: Model to use (MANDATORY)
            **kwargs: Additional configuration options including config_file
                
        Raises:
            LLMProviderError: If configuration or model is invalid
            ValueError: If required parameters are missing
        """
        # Validate mandatory parameters
        if not api_key or not api_key.strip():
            raise ValueError("api_key is required and cannot be empty")
        if not model_name or not model_name.strip():
            raise ValueError("model_name is required and cannot be empty")
        
        # Initialize parent class first
        super().__init__(api_key, model_name, **kwargs)
        
        # Validate model
        if model_name not in self.SUPPORTED_MODELS:
            raise ModelNotFoundError(
                f"Model {model_name} not supported. Available: {list(self.SUPPORTED_MODELS.keys())}",
                "azure_openai"
            )
        
        self.model_config = self.SUPPORTED_MODELS[model_name]
        
        # Load configuration
        config_file = kwargs.get("config_file", "config.ini")
        self.config = configparser.ConfigParser()
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise LLMProviderError(f"Configuration file not found: {config_file}", "azure_openai")
        
        self.config.read(config_path)
        
        # Azure OpenAI configuration with dynamic model-specific parsing
        try:
            llm_config = self.config['PROVIDER_CONFIGS']
            
            # Extract model key for dynamic configuration (e.g., "gpt4o", "gpt4o-mini")
            model_key = self._extract_model_key(model_name)
            
            # Validate and parse configuration
            config_data = self._validate_configuration(llm_config, model_key)
            
            # Set configuration attributes from validated data
            self.azure_api_key = config_data['azure_openai_api_key']
            self.azure_endpoint = config_data['azure_openai_endpoint']
            self.api_version = config_data['azure_openai_api_version']
            self.deployment_name = config_data[f'{model_key}_deployment_name']
            self.max_tokens_default = config_data[f'{model_key}_max_tokens']
            
            # Handle temperature default - use fixed temperature if model requires it
            if "fixed_temperature" in self.model_config:
                self.temperature_default = self.model_config["fixed_temperature"]
                print(f"Using fixed temperature {self.temperature_default} for {model_name}")
            else:
                self.temperature_default = config_data[f'{model_key}_temperature']
                
            self.timeout = config_data[f'{model_key}_timeout']
            
        except (KeyError, ValueError) as e:
            raise LLMProviderError(f"Invalid Azure OpenAI configuration: {e}", "azure_openai")
        
        # Initialize Azure OpenAI client
        if AzureOpenAI is None:
            raise LLMProviderError(
                "Azure OpenAI library not available. Please install: pip install openai",
                "azure_openai"
            )
            
        try:
            self.client = AzureOpenAI(
                api_key=self.azure_api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version,
                timeout=self.timeout
            )
        except Exception as e:
            raise LLMProviderError(f"Failed to initialize Azure OpenAI client: {e}", "azure_openai")
        
        # Initialize tracking attributes
        self.failed_requests = 0
        
        print(f"Azure OpenAI Provider initialized with model: {model_name} (deployment: {self.deployment_name})")
        
    def _extract_model_key(self, model_name: str) -> str:
        """Extract configuration key from model name for dynamic config parsing.
        
        Args:
            model_name: Full model name (e.g., "gpt-4o", "o4-mini")
            
        Returns:
            str: Configuration key (e.g., "gpt4o", "gpt4o-mini")
        """
        # Map model names to configuration keys
        model_key_mapping = {
            "gpt-4o": "gpt4o",
            "o4-mini": "gpt4o-mini",
            "gpt-4o-mini": "gpt4o-mini"
        }
        
        return model_key_mapping.get(model_name, model_name.replace('-', '').lower())
    
    def _validate_configuration(self, llm_config: configparser.SectionProxy, model_key: str) -> dict:
        """Validate and parse model-specific configuration using schema-based validation.
        
        Args:
            llm_config: Configuration section
            model_key: Model configuration key
            
        Returns:
            dict: Parsed and validated configuration
            
        Raises:
            LLMProviderError: If configuration is invalid or missing required fields
        """
        config_data = {}
        errors = []
        
        # Process each field in the configuration schema
        for field_template, schema in self.CONFIG_SCHEMA.items():
            # Replace {model_key} placeholder with actual model key
            field_name = field_template.replace('{model_key}', model_key)
            
            # Get value from configuration
            raw_value = llm_config.get(field_name)
            
            # Handle required fields
            if schema['required']:
                if not raw_value or not str(raw_value).strip():
                    errors.append(f"Missing required field: {schema['description']} ({field_name})")
                    continue
            
            # Use default if value is missing and field is optional
            if not raw_value and not schema['required']:
                if 'default' in schema:
                    default_val = schema['default']
                    # Handle special default values
                    if default_val == 'model_config.max_tokens':
                        raw_value = str(self.model_config['max_tokens'])
                    else:
                        raw_value = str(default_val)
                else:
                    continue  # Skip optional fields without defaults
            
            # Type conversion
            try:
                if schema['type'] == int:
                    value = int(raw_value)
                elif schema['type'] == float:
                    value = float(raw_value)
                else:
                    value = str(raw_value).strip()
                
                # Apply validation if provided
                if 'validation' in schema:
                    if not schema['validation'](value):
                        errors.append(f"Invalid value for {field_name}: {raw_value}")
                        continue
                
                config_data[field_name] = value
                
            except (ValueError, TypeError) as e:
                errors.append(f"Invalid {schema['type'].__name__} value for {field_name}: {raw_value}")
        
        if errors:
            error_msg = f"Azure OpenAI configuration errors for model {self.model_name}:\\n" + "\\n".join(errors)
            raise LLMProviderError(error_msg, "azure_openai")
        
        return config_data
    
    @classmethod
    def generate_config_documentation(cls, model_key: str = "gpt4o") -> str:
        """Generate configuration documentation from schema.
        
        Args:
            model_key: Model key to use in examples
            
        Returns:
            str: Configuration documentation
        """
        doc_lines = []
        doc_lines.append(f"# Azure OpenAI Provider Configuration")
        doc_lines.append(f"# Configuration for {cls.__name__}")
        doc_lines.append("")
        doc_lines.append("[PROVIDER_CONFIGS]")
        doc_lines.append("")
        
        # Group fields by type
        common_fields = []
        model_fields = []
        
        for field_template, schema in cls.CONFIG_SCHEMA.items():
            if '{model_key}' in field_template:
                model_fields.append((field_template, schema))
            else:
                common_fields.append((field_template, schema))
        
        # Document common fields
        if common_fields:
            doc_lines.append("# Common Azure OpenAI settings")
            for field_name, schema in common_fields:
                doc_lines.append(f"# {schema['description']}")
                if 'example' in schema:
                    doc_lines.append(f"# Example: {schema['example']}")
                required_text = " (REQUIRED)" if schema['required'] else ""
                default_text = f" (default: {schema.get('default', 'none')})" if not schema['required'] else ""
                doc_lines.append(f"{field_name} = your_value_here{required_text}{default_text}")
                doc_lines.append("")
        
        # Document model-specific fields
        if model_fields:
            doc_lines.append(f"# Model-specific settings (replace {{{model_key}}} with your model key)")
            for field_template, schema in model_fields:
                field_name = field_template.replace('{model_key}', model_key)
                doc_lines.append(f"# {schema['description']}")
                if 'example' in schema:
                    doc_lines.append(f"# Example: {schema['example']}")
                required_text = " (REQUIRED)" if schema['required'] else ""
                default_text = f" (default: {schema.get('default', 'none')})" if not schema['required'] else ""
                doc_lines.append(f"{field_name} = your_value_here{required_text}{default_text}")
                doc_lines.append("")
        
        return "\\n".join(doc_lines)
    
    def get_provider_type(self) -> LLMProviderType:
        """Get the provider type"""
        return LLMProviderType.OPENAI  # Using OPENAI type for compatibility
    
    def generate_response(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response using Azure OpenAI GPT-4o.
        
        Args:
            request: Standardized LLM request
            
        Returns:
            LLMResponse: Generated response
            
        Raises:
            LLMProviderError: For API errors
            RateLimitError: When rate limited
            AuthenticationError: For auth failures
        """
        start_time = time.time()
        self._request_count += 1
        
        try:
            # Prepare messages for Azure OpenAI
            messages = []
            
            # Add system message if provided
            if request.system_prompt:
                messages.append({
                    "role": "system",
                    "content": request.system_prompt
                })
            
            # Add user message
            messages.append({
                "role": "user", 
                "content": request.prompt
            })
            
            # Prepare base API call parameters
            api_params = {
                "model": self.deployment_name,  # Use deployment name for Azure
                "messages": messages,
            }
            
            # Add parameters that are not restricted for this model
            restricted_params = self.model_config.get("restricted_params", [])
            
            if "top_p" not in restricted_params and request.top_p is not None:
                api_params["top_p"] = request.top_p
            elif "top_p" in restricted_params and request.top_p is not None:
                print(f"Warning: {self.model_name} does not support top_p parameter. Ignoring value: {request.top_p}")
            
            if "stop" not in restricted_params and request.stop_sequences:
                api_params["stop"] = request.stop_sequences
            elif "stop" in restricted_params and request.stop_sequences:
                print(f"Warning: {self.model_name} does not support stop sequences. Ignoring: {request.stop_sequences}")
            
            # Handle temperature parameter - some models only support fixed temperature
            if "fixed_temperature" in self.model_config:
                # Model only supports fixed temperature (like GPT-4o-mini)
                fixed_temp = self.model_config["fixed_temperature"]
                api_params["temperature"] = fixed_temp
                
                # Log warning if user requested different temperature
                requested_temp = request.temperature if request.temperature is not None else self.temperature_default
                if requested_temp != fixed_temp:
                    print(f"Warning: {self.model_name} only supports temperature={fixed_temp}. "
                          f"Ignoring requested temperature={requested_temp}")
            else:
                # Model supports custom temperature
                api_params["temperature"] = request.temperature if request.temperature is not None else self.temperature_default
            
            # Handle max_tokens vs max_completion_tokens based on model configuration
            max_tokens_value = request.max_tokens or self.max_tokens_default
            token_param = self.model_config.get("token_param", "max_tokens")
            api_params[token_param] = max_tokens_value
            
            # Make the request to Azure OpenAI
            response = self.client.chat.completions.create(**api_params)
            
            # Parse the response
            generation_time = time.time() - start_time
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            
            # Extract usage information
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else input_tokens + output_tokens
            
            # Calculate cost
            input_cost = (input_tokens / 1_000_000) * self.model_config["input_cost_per_million"]
            output_cost = (output_tokens / 1_000_000) * self.model_config["output_cost_per_million"]
            total_cost = input_cost + output_cost
            
            # Update internal statistics
            self._total_tokens += total_tokens
            self._total_cost += total_cost
            
            llm_response = LLMResponse(
                content=content,
                model_name=self.model_name,
                provider="azure_openai",
                generation_time=generation_time,
                token_count=total_tokens,
                total_cost=total_cost,
                finish_reason=finish_reason,
                metadata={
                    "azure_deployment": self.deployment_name,
                    "azure_endpoint": self.azure_endpoint,
                    "api_version": self.api_version,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "model_id": response.model
                }
            )
            
            # Update internal stats
            self._update_stats(llm_response)
            return llm_response
            
        except Exception as e:
            self.failed_requests += 1
            
            # Handle specific Azure OpenAI errors
            error_message = str(e)
            
            if "rate_limit" in error_message.lower() or "429" in error_message:
                raise RateLimitError(f"Azure OpenAI rate limit exceeded: {error_message}", "azure_openai")
            elif "unauthorized" in error_message.lower() or "401" in error_message:
                raise AuthenticationError(f"Azure OpenAI authentication failed: {error_message}", "azure_openai")
            elif "not found" in error_message.lower() or "404" in error_message:
                raise ModelNotFoundError(f"Azure OpenAI model not found: {error_message}", "azure_openai")
            else:
                raise LLMProviderError(f"Azure OpenAI error: {error_message}", "azure_openai")
            
            # Create mock response
            response_content = self._generate_mock_response(request)
            generation_time = time.time() - start_time
            
            # Estimate token count (rough approximation)
            input_tokens = len(request.prompt.split()) * 1.3  # Rough tokenization
            output_tokens = len(response_content.split()) * 1.3
            total_tokens = int(input_tokens + output_tokens)
            
            # Calculate cost
            model_config = self.SUPPORTED_MODELS[self.model_name]
            input_cost = (input_tokens / 1_000_000) * model_config["input_cost_per_million"]
            output_cost = (output_tokens / 1_000_000) * model_config["output_cost_per_million"]
            total_cost = input_cost + output_cost
            
            response = LLMResponse(
                content=response_content,
                model_name=self.model_name,
                provider="openai",
                generation_time=generation_time,
                token_count=total_tokens,
                total_cost=total_cost,
                finish_reason="stop",
                metadata={
                    "input_tokens": int(input_tokens),
                    "output_tokens": int(output_tokens),
                    "input_cost": input_cost,
                    "output_cost": output_cost,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens
                }
            )
            
            self._update_stats(response)
            return response
            
        except Exception as e:
            # TODO: Handle specific OpenAI exceptions
            # - openai.RateLimitError -> RateLimitError
            # - openai.AuthenticationError -> AuthenticationError
            # - openai.NotFoundError -> ModelNotFoundError
            raise self._handle_openai_error(e)
    
    def get_model_info(self) -> ModelInfo:
        """
        Get information about the current model.
        
        Returns:
            ModelInfo: Detailed model information
        """
        model_config = self.SUPPORTED_MODELS[self.model_name]
        
        return ModelInfo(
            name=self.model_name,
            provider="openai",
            max_tokens=model_config["max_tokens"],
            input_cost_per_token=model_config["input_cost_per_million"] / 1_000_000,
            output_cost_per_token=model_config["output_cost_per_million"] / 1_000_000,
            context_window=model_config["context_window"],
            capabilities=model_config["capabilities"],
            description=f"OpenAI {self.model_name} - Advanced language model with vision and function calling capabilities"
        )
    
    def validate_api_key(self) -> bool:
        """
        Validate the OpenAI API key.
        
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # TODO: Implement actual API key validation
            # response = self.client.models.list()
            # return True if response else False
            
            # Placeholder validation
            return len(self.api_key) > 20 and self.api_key.startswith("sk-")
            
        except Exception:
            return False
    
    def estimate_cost(self, request: LLMRequest) -> Tuple[float, float]:
        """
        Estimate the cost of a request.
        
        Args:
            request: Request to estimate cost for
            
        Returns:
            Tuple[float, float]: (input_cost, estimated_total_cost)
        """
        model_config = self.SUPPORTED_MODELS[self.model_name]
        
        # Rough token estimation
        input_tokens = len(request.prompt.split()) * 1.3
        estimated_output_tokens = min(
            request.max_tokens or 1000,
            model_config["max_tokens"] // 4  # Assume 25% of context for output
        )
        
        input_cost = (input_tokens / 1_000_000) * model_config["input_cost_per_million"]
        output_cost = (estimated_output_tokens / 1_000_000) * model_config["output_cost_per_million"]
        total_cost = input_cost + output_cost
        
        return input_cost, total_cost
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available OpenAI models.
        
        Returns:
            List[str]: Available model names
        """
        return list(self.SUPPORTED_MODELS.keys())
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on OpenAI service.
        
        Returns:
            Dict[str, Any]: Health status
        """
        try:
            # TODO: Implement actual health check
            # response = self.client.models.retrieve(self.model_name)
            
            # Placeholder health check
            is_healthy = self.validate_api_key()
            
            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "provider": "openai",
                "model": self.model_name,
                "api_key_valid": is_healthy,
                "base_url": self.base_url,
                "timestamp": time.time(),
                "capabilities": self.SUPPORTED_MODELS[self.model_name]["capabilities"]
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "openai", 
                "model": self.model_name,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _generate_mock_response(self, request: LLMRequest) -> str:
        """Generate a mock response for testing purposes"""
        prompt_preview = request.prompt[:100] + "..." if len(request.prompt) > 100 else request.prompt
        
        mock_responses = [
            f"GPT-4o Response: I understand you're asking about '{prompt_preview}'. This is a comprehensive analysis based on my training data and reasoning capabilities.",
            f"Based on your prompt '{prompt_preview}', I can provide the following insights and recommendations...",
            f"Thank you for your question about '{prompt_preview}'. Let me break this down systematically...",
            f"GPT-4o Analysis: Your query '{prompt_preview}' touches on several important aspects that I'd like to address..."
        ]
        
        import random
        base_response = random.choice(mock_responses)
        
        # Add some variability based on temperature
        if request.temperature > 0.7:
            base_response += " [High creativity mode engaged]"
        elif request.temperature < 0.3:
            base_response += " [Precise analytical mode]"
        
        return base_response
    
    def _handle_openai_error(self, error: Exception) -> LLMProviderError:
        """
        Handle OpenAI-specific errors.
        
        Args:
            error: Original OpenAI error
            
        Returns:
            LLMProviderError: Standardized error
        """
        error_message = str(error).lower()
        
        # TODO: Handle specific OpenAI error types
        if "rate limit" in error_message or "quota exceeded" in error_message:
            return RateLimitError(str(error), "openai")
        elif "authentication" in error_message or "api key" in error_message:
            return AuthenticationError(str(error), "openai")
        elif "model not found" in error_message:
            return ModelNotFoundError(str(error), "openai")
        else:
            return LLMProviderError(str(error), "openai")


# Utility functions for Azure OpenAI integration
def create_openai_provider(api_key: str, model_name: str, **kwargs) -> AzureOpenAIProvider:
    """
    Factory function to create Azure OpenAI provider.
    
    Args:
        api_key: Azure API key (MANDATORY - no defaults)
        model_name: Model to use (MANDATORY - no defaults)
        **kwargs: Additional configuration
        
    Returns:
        AzureOpenAIProvider: Configured provider instance
        
    Raises:
        ValueError: If required parameters are missing or empty
    """
    if not api_key or not api_key.strip():
        raise ValueError("api_key is required and cannot be empty")
    if not model_name or not model_name.strip():
        raise ValueError("model_name is required and cannot be empty")
    
    return AzureOpenAIProvider(api_key, model_name, **kwargs)


def get_openai_models() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all supported Azure OpenAI models.
    
    Returns:
        Dict: Model information
    """
    return AzureOpenAIProvider.SUPPORTED_MODELS.copy()