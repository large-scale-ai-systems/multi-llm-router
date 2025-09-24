"""
Anthropic Provider for Amazon Bedrock Integration.

This module provides an implementation of the LLM provider interface
for Anthropic's Claude models accessed via Amazon Bedrock.
"""

import json
import time
import asyncio
import configparser
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, BotoCoreError

from ..base_provider import (
    BaseLLMProvider, LLMRequest, LLMResponse, ModelInfo, LLMProviderType,
    LLMProviderError, RateLimitError, AuthenticationError, ModelNotFoundError
)


# Supported Claude models via Bedrock
SUPPORTED_MODELS = {
    "claude-sonnet-4": {
        "model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "context_window": 200000,
        "max_tokens": 8192,
        "input_cost_per_million": 3.00,
        "output_cost_per_million": 15.00,
        "capabilities": ["text", "vision", "function_calling", "reasoning"]
    },
    "claude-3.5-sonnet": {
        "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "context_window": 200000,
        "max_tokens": 8192,
        "input_cost_per_million": 3.00,
        "output_cost_per_million": 15.00,
        "capabilities": ["text", "vision", "function_calling", "reasoning"]
    },
    "claude-3-sonnet": {
        "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
        "context_window": 200000,
        "max_tokens": 4096,
        "input_cost_per_million": 3.00,
        "output_cost_per_million": 15.00,
        "capabilities": ["text", "vision"]
    },
    "claude-3-haiku": {
        "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
        "context_window": 200000,
        "max_tokens": 4096,
        "input_cost_per_million": 0.25,
        "output_cost_per_million": 1.25,
        "capabilities": ["text", "vision"]
    }
}


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Provider implementation using Amazon Bedrock.
    
    This provider integrates with Anthropic's Claude models through
    Amazon Bedrock managed service, providing enterprise-grade
    authentication and access controls.
    """
    
    def __init__(self, api_key: str, model_name: str, **kwargs):
        """
        Initialize the Anthropic provider.
        
        Args:
            api_key: AWS access key ID (MANDATORY)
            model_name: Name of the Claude model to use (MANDATORY)
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
        
        if model_name not in SUPPORTED_MODELS:
            raise LLMProviderError(f"Unsupported model: {model_name}", "anthropic")
        
        self.model_config = SUPPORTED_MODELS[model_name]
        self.model_id = self.model_config["model_id"]
        
        # Load configuration
        config_file = kwargs.get("config_file", "config.ini")
        self.config = configparser.ConfigParser()
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise LLMProviderError(f"Configuration file not found: {config_file}", "anthropic")
        
        self.config.read(config_path)
        
        # AWS Bedrock configuration
        try:
            aws_config = self.config['AWS_BEDROCK']
            provider_config = self.config['PROVIDER_CONFIGS']
            
            # AWS credentials - these are the actual API keys (not the placeholder parameter)
            self.aws_access_key_id = aws_config.get('aws_access_key_id')
            self.aws_secret_access_key = aws_config.get('aws_secret_access_key')
            self.aws_region = aws_config.get('aws_region', 'us-east-1')
            
            # Provider-specific configuration
            self.model_id = provider_config.get('claude_model_id', self.model_config['model_id'])
            self.max_tokens_default = int(provider_config.get('claude_max_tokens', str(self.model_config['max_tokens'])))
            self.temperature_default = float(provider_config.get('claude_temperature', '1.0'))
            self.top_p_default = float(provider_config.get('claude_top_p', '0.999'))
            self.timeout = int(provider_config.get('claude_timeout', '30'))
            
        except (KeyError, ValueError) as e:
            raise LLMProviderError(f"Invalid AWS Bedrock or provider AWS Bedrock or provider configuration: {e}", "anthropic")
        
        # Validate required AWS credentials
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            raise LLMProviderError(
                "AWS credentials not found in config. Please ensure aws_access_key_id and aws_secret_access_key are set in [AWS_BEDROCK] section.", 
                "anthropic"
            )
        
        # Initialize Bedrock client
        try:
            self.bedrock_client = boto3.client(
                'bedrock-runtime',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region
            )
            
            # Initialize tracking attributes
            self.failed_requests = 0
            
            print(f"Anthropic Provider initialized with model: {model_name} (model_id: {self.model_id})")
        except Exception as e:
            raise LLMProviderError(f"Failed to initialize Bedrock client: {e}", "anthropic")
    
    def get_provider_type(self) -> LLMProviderType:
        """Get the provider type."""
        return LLMProviderType.ANTHROPIC
    
    def generate_response(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response from Claude via Bedrock.
        
        Args:
            request (LLMRequest): The request to process
            
        Returns:
            LLMResponse: The generated response
            
        Raises:
            LLMProviderError: If the request fails
        """
        start_time = time.time()
        self._request_count += 1
        
        try:
            # Prepare the payload for Claude via Bedrock
            messages = []
            
            # Handle the prompt (convert to messages format)
            messages.append({
                "role": "user",
                "content": request.prompt
            })
            
            # Build the payload
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": request.max_tokens or self.max_tokens_default,
                "messages": messages
            }
            
            # Add optional parameters
            if request.temperature is not None:
                payload["temperature"] = request.temperature
            else:
                payload["temperature"] = self.temperature_default
                
            if request.system_prompt:
                payload["system"] = request.system_prompt
                
            if request.top_p is not None:
                payload["top_p"] = request.top_p
            else:
                payload["top_p"] = self.top_p_default
            
            # Make the request to Bedrock
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(payload)
            )
            
            # Parse the response
            response_body = json.loads(response['body'].read())
            
            # Extract content from Claude's response format
            content = response_body.get('content', [])
            if content and isinstance(content, list):
                text_content = content[0].get('text', '')
            else:
                text_content = str(content)
            
            # Calculate metrics
            generation_time = time.time() - start_time
            input_tokens = response_body.get('usage', {}).get('input_tokens', 0)
            output_tokens = response_body.get('usage', {}).get('output_tokens', 0)
            total_tokens = input_tokens + output_tokens
            
            # Calculate cost
            input_cost = (input_tokens / 1_000_000) * self.model_config["input_cost_per_million"]
            output_cost = (output_tokens / 1_000_000) * self.model_config["output_cost_per_million"]
            total_cost = input_cost + output_cost
            
            # Update internal statistics
            self._total_tokens += total_tokens
            self._total_cost += total_cost
            
            llm_response = LLMResponse(
                content=text_content,
                model_name=self.model_name,
                provider="anthropic",
                generation_time=generation_time,
                token_count=total_tokens,
                total_cost=total_cost,
                finish_reason=response_body.get('stop_reason'),
                metadata={
                    "bedrock_request_id": response['ResponseMetadata'].get('RequestId'),
                    "model_id": self.model_id,
                    "aws_region": self.aws_region,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
            )
            
            # Update internal stats
            self._update_stats(llm_response)
            return llm_response
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            
            if error_code == 'ThrottlingException':
                raise RateLimitError(f"Request throttled: {error_message}", "anthropic")
            elif error_code == 'ValidationException':
                raise LLMProviderError(f"Invalid request: {error_message}", "anthropic", error_code)
            elif error_code == 'AccessDeniedException':
                raise AuthenticationError(f"Access denied: {error_message}", "anthropic")
            else:
                raise LLMProviderError(f"AWS error ({error_code}): {error_message}", "anthropic", error_code)
                
        except BotoCoreError as e:
            raise LLMProviderError(f"AWS service error: {e}", "anthropic")
            
        except json.JSONDecodeError as e:
            raise LLMProviderError(f"Failed to parse response: {e}", "anthropic")
            
        except Exception as e:
            raise LLMProviderError(f"Unexpected error: {e}", "anthropic")
    
    def get_model_info(self) -> ModelInfo:
        """
        Get information about the current model.
        
        Returns:
            ModelInfo: Detailed model information
        """
        config = self.model_config
        return ModelInfo(
            name=self.model_name,
            provider="anthropic",
            max_tokens=config["max_tokens"],
            input_cost_per_token=config["input_cost_per_million"] / 1_000_000,
            output_cost_per_token=config["output_cost_per_million"] / 1_000_000,
            context_window=config["context_window"],
            capabilities=config["capabilities"],
            description=f"Claude {self.model_name} via Amazon Bedrock"
        )
    
    def validate_api_key(self) -> bool:
        """
        Validate AWS credentials and Bedrock access.
        
        Returns:
            bool: True if credentials are valid and have Bedrock access
        """
        return self.test_connection()
    
    def estimate_cost(self, request: LLMRequest) -> Tuple[float, float]:
        """
        Estimate the cost of a request.
        
        Args:
            request: LLM request to estimate cost for
            
        Returns:
            Tuple[float, float]: (input_cost, estimated_total_cost)
        """
        # Rough estimation based on prompt length
        # This is an approximation since exact tokenization would require the actual tokenizer
        estimated_input_tokens = len(request.prompt.split()) * 1.3  # Rough approximation
        max_output_tokens = request.max_tokens or self.max_tokens_default
        
        input_cost = (estimated_input_tokens / 1_000_000) * self.model_config["input_cost_per_million"]
        max_output_cost = (max_output_tokens / 1_000_000) * self.model_config["output_cost_per_million"]
        estimated_total_cost = input_cost + max_output_cost
        
        return input_cost, estimated_total_cost
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models from the provider.
        
        Returns:
            List[str]: List of available model names
        """
        return list(SUPPORTED_MODELS.keys())
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the provider.
        
        Returns:
            Dict[str, Any]: Health status information
        """
        try:
            # Test connection with minimal request
            test_request = LLMRequest(
                prompt="Hi",
                max_tokens=10,
                temperature=0.1
            )
            
            response = self.generate_response(test_request)
            
            return {
                "status": "healthy",
                "provider": "anthropic",
                "model": self.model_name,
                "bedrock_model_id": self.model_id,
                "aws_region": self.aws_region,
                "response_time": response.generation_time,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "anthropic",
                "model": self.model_name,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def generate_response_sync(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response synchronously.
        
        Args:
            request (LLMRequest): The request to process
            
        Returns:
            LLMResponse: The generated response
            
        Raises:
            LLMProviderError: If the request fails
        """
        start_time = time.time()
        self.total_requests += 1
        
        try:
            # Prepare the payload for Claude via Bedrock
            messages = []
            
            # Handle different message formats
            if isinstance(request.messages, str):
                messages.append({
                    "role": "user",
                    "content": request.messages
                })
            elif isinstance(request.messages, list):
                messages = request.messages
            else:
                raise LLMProviderError("Messages must be string or list")
            
            # Build the payload
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": request.max_tokens or self.max_tokens,
                "messages": messages
            }
            
            # Add optional parameters
            if request.temperature is not None:
                payload["temperature"] = request.temperature
            elif self.temperature:
                payload["temperature"] = self.temperature
                
            if request.system_prompt:
                payload["system"] = request.system_prompt
                
            if self.top_p:
                payload["top_p"] = self.top_p
            
            # Make the request to Bedrock
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(payload)
            )
            
            # Parse the response
            response_body = json.loads(response['body'].read())
            
            # Extract content from Claude's response format
            content = response_body.get('content', [])
            if content and isinstance(content, list):
                text_content = content[0].get('text', '')
            else:
                text_content = str(content)
            
            # Calculate metrics
            response_time = time.time() - start_time
            input_tokens = response_body.get('usage', {}).get('input_tokens', 0)
            output_tokens = response_body.get('usage', {}).get('output_tokens', 0)
            total_tokens = input_tokens + output_tokens
            
            # Calculate cost
            input_cost = (input_tokens / 1_000_000) * self.model_config["input_cost_per_million"]
            output_cost = (output_tokens / 1_000_000) * self.model_config["output_cost_per_million"]
            total_cost = input_cost + output_cost
            
            # Update metrics
            self.successful_requests += 1
            self.total_tokens_used += total_tokens
            self.total_cost += total_cost
            self.total_response_time += response_time
            
            return LLMResponse(
                content=text_content,
                model=self.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                response_time=response_time,
                cost=total_cost,
                metadata={
                    "bedrock_request_id": response['ResponseMetadata'].get('RequestId'),
                    "model_id": self.model_id,
                    "aws_region": self.aws_region
                }
            )
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            
            self.failed_requests += 1
            self.logger.error(f"AWS ClientError: {error_code} - {error_message}")
            
            if error_code == 'ThrottlingException':
                raise LLMProviderError(f"Request throttled: {error_message}")
            elif error_code == 'ValidationException':
                raise LLMProviderError(f"Invalid request: {error_message}")
            elif error_code == 'AccessDeniedException':
                raise LLMProviderError(f"Access denied: {error_message}")
            else:
                raise LLMProviderError(f"AWS error ({error_code}): {error_message}")
                
        except BotoCoreError as e:
            self.failed_requests += 1
            self.logger.error(f"AWS BotoCoreError: {e}")
            raise LLMProviderError(f"AWS service error: {e}")
            
        except json.JSONDecodeError as e:
            self.failed_requests += 1
            self.logger.error(f"JSON decode error: {e}")
            raise LLMProviderError(f"Failed to parse response: {e}")
            
        except Exception as e:
            self.failed_requests += 1
            self.logger.error(f"Unexpected error in generate_response_sync: {e}")
            raise LLMProviderError(f"Unexpected error: {e}")
    def test_connection(self) -> bool:
        """
        Test connection to Bedrock service.
        
        Returns:
            bool: True if connection is successful
        """
        try:
            # Test with a minimal request
            test_payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "messages": [
                    {
                        "role": "user",
                        "content": "Hi"
                    }
                ]
            }
            
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(test_payload)
            )
            
            return response['ResponseMetadata']['HTTPStatusCode'] == 200
            
        except Exception as e:
            return False

    def __str__(self) -> str:
        """String representation of the provider."""
        return f"AnthropicProvider(model={self.model_id}, region={self.aws_region})"


# Factory functions for compatibility with existing code
def create_anthropic_provider(api_key: str, model_name: str, **kwargs) -> AnthropicProvider:
    """
    Create an Anthropic provider instance.
    
    Args:
        api_key: AWS access key ID (MANDATORY - no defaults)
        model_name: Claude model to use (MANDATORY - no defaults)
        **kwargs: Additional configuration including config_file path
        
    Returns:
        AnthropicProvider: Configured Anthropic provider instance
        
    Raises:
        ValueError: If required parameters are missing or empty
    """
    if not api_key or not api_key.strip():
        raise ValueError("api_key is required and cannot be empty")
    if not model_name or not model_name.strip():
        raise ValueError("model_name is required and cannot be empty")
    
    return AnthropicProvider(api_key=api_key, model_name=model_name, **kwargs)


def get_anthropic_models() -> list[str]:
    """Get list of supported Anthropic models."""
    return list(SUPPORTED_MODELS.keys())