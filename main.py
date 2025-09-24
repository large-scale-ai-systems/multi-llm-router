#!/usr/bin/env python3
"""
MultiLLMRouter - Control Theory Based LLM Routing System

This system implements intelligent LLM routing using PID controllers 
for Proportional, Integral and Derivative control theory.

The PID components operate as follows: the proportional component provides direct 
response to current deviation from target allocation, the integral component 
accumulates historical routing decisions through persistent bias updates, and 
the derivative component is implicitly handled through adaptive learning rate 
schedules that respond to allocation or routing changes.

Enhanced with Vector Database evalation for real-time quality assessment.
"""

import time
import csv
import configparser
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import our modular components
from core.data_models import HumanEvalRecord
from routers.vector_enhanced_router import VectorDBEnhancedRouter

# Import provider factory for real provider integration
from llm_providers.factory import ProviderFactory
from llm_providers.base_provider import LLMProviderType


def create_human_eval_set() -> List[HumanEvalRecord]:
    """Load human evaluation set from CSV file"""
    csv_file = Path("data/human_eval_set.csv")
    
    if not csv_file.exists():
        raise FileNotFoundError(f"Required CSV file {csv_file} not found. Please ensure data/human_eval_set.csv exists.")
    
    eval_records = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                record = HumanEvalRecord(
                    id=row['id'],
                    prompt=row['prompt'],
                    golden_output=row['golden_output'],
                    category=row['category'],
                    difficulty=row['difficulty'],
                    source=row['source']
                )
                eval_records.append(record)
        
        print(f"Loaded {len(eval_records)} human evaluation records from {csv_file}")
        return eval_records
        
    except Exception as e:
        raise Exception(f"Failed to parse CSV file {csv_file}: {e}. Please check the file format and contents.") from e


def smart_ingest_evaluation_set(vector_evaluator, eval_set, force_rebuild=False):
    """
    Smart ingestion that only rebuilds index when necessary.
    
    Args:
        vector_evaluator: The vector evaluator instance
        eval_set: List of HumanEvalRecord objects
        force_rebuild: Force index rebuild even if it exists
    """
    import os
    import pickle
    from pathlib import Path
    
    index_path = Path(vector_evaluator.index_path)
    metadata_file = index_path / "metadata.pkl"
    
    # Check if index exists and is valid
    index_exists = False
    current_record_ids = {record.id for record in eval_set}
    
    if not force_rebuild and metadata_file.exists():
        try:
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
                indexed_record_ids = set(metadata.get('record_ids', []))
                
            # Check if we have new records to add
            new_records = [r for r in eval_set if r.id not in indexed_record_ids]
            missing_records = indexed_record_ids - current_record_ids
            
            if not new_records and not missing_records:
                print(f"Index is up-to-date with {len(current_record_ids)} records")
                # Just load existing records into evaluator
                vector_evaluator.human_eval_records = eval_set
                return
            elif new_records and not missing_records:
                print(f"Found {len(new_records)} new records to add to existing index")
                # For incremental updates, just do full rebuild
                # Different evaluator types handle indexing differently
                print(f"Rebuilding index with {len(eval_set)} total records...")
                vector_evaluator.ingest_human_eval_set(eval_set)
                return
            else:
                print(f"Index structure changed: {len(new_records)} new, {len(missing_records)} removed. Rebuilding...")
                
        except Exception as e:
            print(f"Warning: Could not read index metadata: {e}. Rebuilding index...")
    
    # Full rebuild needed
    print(f"Building vector index for {len(eval_set)} evaluation records...")
    vector_evaluator.ingest_human_eval_set(eval_set)
    
    # Save metadata for future smart loading
    try:
        index_path.mkdir(parents=True, exist_ok=True)
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'record_ids': list(current_record_ids),
                'total_records': len(eval_set),
                'implementation': vector_evaluator.get_implementation_info()['implementation']
            }, f)
    except Exception as e:
        print(f"Warning: Could not save index metadata: {e}")


def create_real_providers(config: configparser.ConfigParser) -> Dict[str, Any]:
    """
    Create real provider instances from PROVIDER_CONFIGS section.
    
    Args:
        config: ConfigParser object with loaded configuration
        
    Returns:
        Dictionary mapping provider names to provider instances
        
    Raises:
        LLMProviderError: If provider creation fails or required configuration is missing
        ValueError: If mandatory configuration sections/keys are missing
    """
    providers = {}
    errors = []
    
    # Validate required configuration sections exist
    if 'PROVIDER_CONFIGS' not in config:
        raise ValueError("PROVIDER_CONFIGS section not found in config.ini. This section is mandatory for real provider initialization.")
    
    provider_config = config['PROVIDER_CONFIGS']
    
    # Create Azure OpenAI Providers - Support multiple models/deployments
    azure_api_key = provider_config.get('azure_openai_api_key')
    if azure_api_key:
        if not azure_api_key.strip():
            errors.append("Azure OpenAI API key is empty in PROVIDER_CONFIGS section")
        else:
            try:
                # Validate common Azure OpenAI configuration
                required_azure_keys = ['azure_openai_endpoint']
                missing_keys = [key for key in required_azure_keys if not provider_config.get(key, '').strip()]
                if missing_keys:
                    errors.append(f"Missing required Azure OpenAI configuration keys: {missing_keys}")
                else:
                    # Get list of OpenAI models to create
                    openai_models = provider_config.get('openai_models', '').strip()
                    if openai_models:
                        model_list = [model.strip() for model in openai_models.split(',')]
                        
                        for model_key in model_list:
                            # Get model-specific configuration
                            deployment_name = provider_config.get(f'{model_key}_deployment_name')
                            model_name = provider_config.get(f'{model_key}_model_name', model_key.replace('-', '_'))
                            
                            if deployment_name and deployment_name.strip():
                                try:
                                    azure_provider = ProviderFactory.create_openai_provider(
                                        api_key=azure_api_key,
                                        model_name=model_name,
                                        config_file="config.ini"
                                    )
                                    # Create unique provider name for each model
                                    provider_name = f"azure-openai-{model_key}"
                                    providers[provider_name] = azure_provider
                                    print(f"Created Azure OpenAI provider: {provider_name} (deployment: {deployment_name})")
                                except Exception as e:
                                    errors.append(f"Failed to create Azure OpenAI provider for {model_key}: {e}")
                            else:
                                errors.append(f"Missing deployment name for OpenAI model: {model_key}")
                    else:
                        # Fallback for backward compatibility - single model
                        deployment_name = provider_config.get('azure_openai_deployment_name')
                        if deployment_name and deployment_name.strip():
                            azure_provider = ProviderFactory.create_openai_provider(
                                api_key=azure_api_key,
                                model_name=deployment_name,
                                config_file="config.ini"
                            )
                            providers['azure-openai-gpt4o'] = azure_provider
                            print("Created Azure OpenAI provider successfully (legacy mode)")
                        else:
                            errors.append("Missing azure_openai_deployment_name for legacy single-model configuration")
            except Exception as e:
                errors.append(f"Failed to create Azure OpenAI providers: {e}")
    
    # Create Bedrock Anthropic Provider - MANDATORY if claude config and AWS section exist
    claude_model_id = provider_config.get('claude_model_id')
    if claude_model_id and config.has_section('AWS_BEDROCK'):
        if not claude_model_id.strip():
            errors.append("Claude model ID is empty in PROVIDER_CONFIGS section")
        else:
            try:
                # Validate required AWS Bedrock configuration
                aws_config = config['AWS_BEDROCK']
                required_aws_keys = ['aws_access_key_id', 'aws_secret_access_key', 'aws_region']
                missing_aws_keys = [key for key in required_aws_keys if not aws_config.get(key, '').strip()]
                if missing_aws_keys:
                    errors.append(f"Missing required AWS Bedrock configuration keys: {missing_aws_keys}")
                else:
                    # Extract model name from model_id (MANDATORY, no defaults)
                    if 'claude-sonnet-4' in claude_model_id:
                        model_name = 'claude-sonnet-4'
                    elif 'claude-3-5-sonnet' in claude_model_id:
                        model_name = 'claude-3-5-sonnet'
                    else:
                        errors.append(f"Unsupported Claude model ID: {claude_model_id}. Supported: claude-sonnet-4, claude-3-5-sonnet")
                        model_name = None
                    
                    if model_name:
                        anthropic_provider = ProviderFactory.create_anthropic_provider(
                            api_key=aws_config.get('aws_access_key_id'),  # Use actual AWS key
                            model_name=model_name,
                            config_file="config.ini"
                        )
                        providers['bedrock-claude-sonnet-4'] = anthropic_provider
                        print("Created Bedrock Anthropic provider successfully")
            except Exception as e:
                errors.append(f"Failed to create Bedrock Anthropic provider: {e}")
    elif claude_model_id and not config.has_section('AWS_BEDROCK'):
        errors.append("Claude model ID specified but AWS_BEDROCK section is missing")
    
    # Create Google Provider - MANDATORY if google API key is present
    google_api_key = provider_config.get('google_api_key')
    if google_api_key and google_api_key.strip() and google_api_key != 'your_google_api_key_here':
        try:
            # Validate required Google configuration
            required_google_keys = ['gemini_model']
            missing_google_keys = [key for key in required_google_keys if not provider_config.get(key, '').strip()]
            if missing_google_keys:
                errors.append(f"Missing required Google configuration keys: {missing_google_keys}")
            else:
                google_provider = ProviderFactory.create_google_provider(
                    api_key=google_api_key,
                    model_name=provider_config.get('gemini_model'),
                    timeout=int(provider_config.get('gemini_timeout', '30'))
                )
                providers['google-gemini-pro'] = google_provider
                print("Created Google Gemini provider successfully")
        except Exception as e:
            errors.append(f"Failed to create Google Gemini provider: {e}")
    
    # If any errors occurred, fail fast with detailed error message
    if errors:
        error_message = "Provider initialization failed with the following errors:\n" + "\n".join(f"  • {error}" for error in errors)
        raise ValueError(error_message)
    
    # Ensure at least one provider was created
    if not providers:
        raise ValueError(
            "No providers were created. Please ensure at least one provider is properly configured in config.ini:\n"
            "  • Azure OpenAI: Set azure_openai_api_key, azure_openai_endpoint, azure_openai_deployment_name\n"
            "  • Bedrock Claude: Set claude_model_id in [PROVIDER_CONFIGS] and AWS credentials in [AWS_BEDROCK]\n"
            "  • Google Gemini: Set google_api_key and gemini_model"
        )
    
    print(f"Successfully created {len(providers)} real provider(s): {list(providers.keys())}")
    return providers


def load_config(config_file: str = "config.ini", use_real_providers: bool = False) -> Dict[str, Any]:
    """Load configuration from INI file.
    
    Args:
        config_file: Path to configuration file
        use_real_providers: If True, create real provider instances; if False, use mock configs
        
    Returns:
        Configuration dictionary with either mock configs or real provider instances
    """
    config = configparser.ConfigParser()
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    config.read(config_path)
    
    if use_real_providers:
        # STRICT MODE: Real providers are MANDATORY - no fallbacks allowed
        try:
            providers = create_real_providers(config)
        except Exception as e:
            # Re-raise with additional context for API server initialization
            raise ValueError(f"Real provider initialization failed: {e}") from e
        
        # Create base allocation for real providers (MANDATORY - no defaults)
        base_allocation = {}
        if 'ALLOCATION_PERCENTAGES' in config:
            allocation_section = config['ALLOCATION_PERCENTAGES']
            
            # Map real provider names to allocations (MANDATORY validation)
            for provider_name in providers.keys():
                # Dynamic allocation mapping - check if exact match exists first
                allocation_key = provider_name
                allocation_value = allocation_section.get(allocation_key)
                
                if not allocation_value:
                    # Fallback mappings for backward compatibility
                    if provider_name.startswith('azure-openai-'):
                        # Try legacy mapping for Azure OpenAI providers
                        if provider_name == 'azure-openai-gpt4o':
                            allocation_key = 'azure-openai-gpt4o'
                        else:
                            # Keep the exact provider name as key
                            allocation_key = provider_name
                    elif provider_name == 'bedrock-claude-sonnet-4':
                        allocation_key = 'bedrock-claude-sonnet-4'
                    elif provider_name == 'google-gemini-pro':
                        allocation_key = 'google-gemini-pro'
                    else:
                        allocation_key = provider_name
                    
                    allocation_value = allocation_section.get(allocation_key)
                
                if not allocation_value:
                    raise ValueError(f"Missing allocation percentage for provider {provider_name} in ALLOCATION_PERCENTAGES section. Expected key: {allocation_key}")
                
                try:
                    base_allocation[provider_name] = float(allocation_value)
                except ValueError:
                    raise ValueError(f"Invalid allocation percentage for {provider_name}: {allocation_value}")
        else:
            raise ValueError("ALLOCATION_PERCENTAGES section is mandatory when using real providers")
        
        # Validate allocation percentages sum correctly
        total_allocation = sum(base_allocation.values())
        if abs(total_allocation - 1.0) > 0.01:  # Allow small floating point tolerance
            raise ValueError(f"Provider allocation percentages must sum to 1.0, got {total_allocation}")
        
        # Parse vector DB config (defaults allowed here)
        vector_db_config = {}
        if 'ROUTING_CONFIG' in config:
            routing_section = config['ROUTING_CONFIG']
            vector_db_config["index_path"] = routing_section.get('index_path', './vector_indexes')
            vector_db_config["similarity_threshold"] = float(routing_section.get('similarity_threshold', '0.7'))
        else:
            # Use defaults if ROUTING_CONFIG section missing
            vector_db_config = {
                "index_path": "./vector_indexes",
                "similarity_threshold": 0.7
            }
        
        # Create llm_configs from real provider configurations for compatibility
        llm_configs = {}
        if 'PROVIDER_CONFIGS' in config:
            provider_config = config['PROVIDER_CONFIGS']
            
            # Extract config for each created provider
            for provider_name in providers.keys():
                # Dynamic config extraction based on provider name
                if provider_name.startswith('azure-openai-'):
                    # Extract model key from provider name (e.g., "gpt4o", "gpt4o-mini")
                    model_key = provider_name.replace('azure-openai-', '')
                    
                    # Get model-specific timeout or default
                    timeout_key = f"{model_key}_timeout"
                    timeout_value = provider_config.get(timeout_key, provider_config.get('gpt4o_timeout', '30'))
                    
                    llm_configs[provider_name] = {
                        "base_response_time": float(timeout_value) / 10,  # Convert timeout to reasonable response time
                        "quality_factor": 0.9 if 'gpt4o' in model_key else 0.85,  # Higher quality for full GPT-4o
                        "error_rate": 0.02 if 'gpt4o' in model_key else 0.025
                    }
                elif provider_name == 'bedrock-claude-sonnet-4':
                    llm_configs[provider_name] = {
                        "base_response_time": float(provider_config.get('claude_timeout', '30')) / 10,
                        "quality_factor": 0.85,  # High quality for Claude Sonnet
                        "error_rate": 0.015
                    }
                elif provider_name == 'google-gemini-pro':
                    llm_configs[provider_name] = {
                        "base_response_time": float(provider_config.get('gemini_timeout', '30')) / 10,
                        "quality_factor": 0.8,  # Good quality for Gemini
                        "error_rate": 0.025
                    }
                else:
                    # Default configuration for unknown providers
                    llm_configs[provider_name] = {
                        "base_response_time": 3.0,
                        "quality_factor": 0.75,
                        "error_rate": 0.03
                    }
        
        # Parse selection weights if available
        selection_weights = {}
        if 'SELECTION_WEIGHTS' in config:
            weights_section = config['SELECTION_WEIGHTS']
            for key in weights_section:
                value = weights_section[key]
                # Handle comments - take only the part before the comment
                if '#' in value:
                    value = value.split('#')[0].strip()
                
                # Try to convert to appropriate type
                if value.lower() in ['true', 'false']:
                    selection_weights[key] = value.lower() == 'true'
                else:
                    try:
                        # Try float first (for numeric values)
                        selection_weights[key] = float(value)
                    except ValueError:
                        # Keep as string if conversion fails
                        selection_weights[key] = value
        
        return {
            "providers": providers,
            "llm_configs": llm_configs,
            "base_allocation": base_allocation,
            "vector_db_config": vector_db_config,
            "selection_weights": selection_weights,
            "use_real_providers": True
        }
    
    else:
        # Original mock configuration loading
        # Parse LLM configs
        llm_configs = {}
        if 'MOCK_CONFIGS' in config:
            llm_config_section = config['MOCK_CONFIGS']
            
            # Parse gpt4_turbo config
            llm_configs["gpt4_turbo"] = {
                "base_response_time": float(llm_config_section.get('gpt4_turbo_base_response_time', '2.0')),
                "quality_factor": float(llm_config_section.get('gpt4_turbo_quality_factor', '0.9')),
                "error_rate": float(llm_config_section.get('gpt4_turbo_error_rate', '0.02'))
            }
            
            # Parse claude3_sonnet config
            llm_configs["claude3_sonnet"] = {
                "base_response_time": float(llm_config_section.get('claude3_sonnet_base_response_time', '1.5')),
                "quality_factor": float(llm_config_section.get('claude3_sonnet_quality_factor', '0.85')),
                "error_rate": float(llm_config_section.get('claude3_sonnet_error_rate', '0.015'))
            }
            
            # Parse gemini_pro config
            llm_configs["gemini_pro"] = {
                "base_response_time": float(llm_config_section.get('gemini_pro_base_response_time', '1.2')),
                "quality_factor": float(llm_config_section.get('gemini_pro_quality_factor', '0.8')),
                "error_rate": float(llm_config_section.get('gemini_pro_error_rate', '0.025'))
            }
        
        # Parse base allocation
        base_allocation = {}
        if 'ALLOCATION_PERCENTAGES' in config:
            allocation_section = config['ALLOCATION_PERCENTAGES']
            base_allocation = {
                "gpt4_turbo": float(allocation_section.get('gpt4_turbo', '0.30')),
                "claude3_sonnet": float(allocation_section.get('claude3_sonnet', '0.30')),
                "gemini_pro": float(allocation_section.get('gemini_pro', '0.25'))
            }
        
        # Parse vector DB config
        vector_db_config = {
            "index_path": "./vector_indexes",
            "similarity_threshold": 0.7
        }
        if 'ROUTING_CONFIG' in config:
            routing_section = config['ROUTING_CONFIG']
            vector_db_config["similarity_threshold"] = float(routing_section.get('similarity_threshold', '0.7'))
        
        # Parse selection weights if available
        selection_weights = {}
        if 'SELECTION_WEIGHTS' in config:
            weights_section = config['SELECTION_WEIGHTS']
            for key in weights_section:
                value = weights_section[key]
                # Handle comments - take only the part before the comment
                if '#' in value:
                    value = value.split('#')[0].strip()
                
                # Try to convert to appropriate type
                if value.lower() in ['true', 'false']:
                    selection_weights[key] = value.lower() == 'true'
                else:
                    try:
                        # Try float first (for numeric values)
                        selection_weights[key] = float(value)
                    except ValueError:
                        # Keep as string if conversion fails
                        selection_weights[key] = value
        
        return {
            "llm_configs": llm_configs,
            "base_allocation": base_allocation,
            "vector_db_config": vector_db_config,
            "selection_weights": selection_weights,
            "use_real_providers": False
        }


def create_router(config_file: str = "config.ini", use_real_providers: bool = True) -> VectorDBEnhancedRouter:
    """
    Create and return a configured VectorDBEnhancedRouter instance.
    
    Args:
        config_file: Path to configuration file
        use_real_providers: If True, use real provider instances; if False, use mock configs
        
    Returns:
        VectorDBEnhancedRouter: Configured router instance
        
    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If config file is not found
    """
    try:
        config = load_config(config_file, use_real_providers)
        print("Configuration loaded successfully from config.ini")
    except FileNotFoundError:
        print("Warning: config.ini not found, using fallback configuration")
        # Fallback to hardcoded values if config file is missing
        config = {
            "llm_configs": {
                "gpt4_turbo": {"base_response_time": 2.0, "quality_factor": 0.9, "error_rate": 0.02},
                "claude3_sonnet": {"base_response_time": 1.5, "quality_factor": 0.85, "error_rate": 0.015},
                "gemini_pro": {"base_response_time": 1.2, "quality_factor": 0.8, "error_rate": 0.025}
            },
            "base_allocation": {"gpt4_turbo": 0.3, "claude3_sonnet": 0.3, "gemini_pro": 0.25},
            "vector_db_config": {"index_path": "./vector_indexes", "similarity_threshold": 0.7},
            "use_real_providers": False
        }
    except Exception as e:
        if use_real_providers:
            print(f"Real provider initialization failed: {e}")
            print("Falling back to mock mode...")
            # Fallback to mock mode if real providers fail
            try:
                config = load_config(config_file, use_real_providers=False)
                print("Successfully loaded mock configuration")
            except Exception as fallback_error:
                print(f"Mock configuration also failed: {fallback_error}")
                raise
        else:
            print(f"Error loading configuration: {e}")
            raise
    
    # Initialize the enhanced router with loaded configuration
    if config.get("use_real_providers", False):
        # Real providers mode - pass providers directly
        router = VectorDBEnhancedRouter(
            llm_configs=config["llm_configs"],
            base_allocation=config["base_allocation"],
            vector_db_config=config["vector_db_config"],
            providers=config["providers"],
            use_real_providers=True,
            config=config
        )
    else:
        # Mock mode
        router = VectorDBEnhancedRouter(
            llm_configs=config["llm_configs"],
            base_allocation=config["base_allocation"],
            vector_db_config=config["vector_db_config"],
            config=config
        )
    
    # Initialize vector database evaluator with human evaluation data
    print("\\nInitializing vector database evaluator...")
    try:
        eval_set = create_human_eval_set()
        print(f"Loading {len(eval_set)} human evaluation records...")
        
        # Ingest the evaluation set into the vector database
        success = router.vector_db_evaluator.ingest_human_eval_set(eval_set)
        if success:
            print("Vector database evaluator initialized successfully with human eval data")
        else:
            print("Warning: Vector database evaluator initialization had issues, but continuing...")
            
    except FileNotFoundError as e:
        print(f"Warning: Human evaluation data not found: {e}")
        print("Vector evaluator will use fallback similarity calculation")
    except KeyboardInterrupt:
        print("\\nWarning: User interrupted vector database setup - continuing with fallback...")
    except Exception as e:
        print(f"Warning: Vector evaluator initialization failed: {str(e)[:100]}...")
        print("Continuing with fallback similarity calculation...")
    
    return router


def main():
    """Main demonstration of the Vector Database Enhanced Multi-LLM Router"""
    print("=== Vector Database Enhanced Multi-LLM Router Demo ===\\n")
    
    # Load configuration from INI file
    try:
        config = load_config()
        print("Configuration loaded successfully from config.ini")
    except FileNotFoundError:
        print("Warning: config.ini not found, using fallback configuration")
        # Fallback to hardcoded values if config file is missing
        config = {
            "llm_configs": {
                "gpt4_turbo": {"base_response_time": 2.0, "quality_factor": 0.9, "error_rate": 0.02},
                "claude3_sonnet": {"base_response_time": 1.5, "quality_factor": 0.85, "error_rate": 0.015},
                "gemini_pro": {"base_response_time": 1.2, "quality_factor": 0.8, "error_rate": 0.025}
            },
            "base_allocation": {"gpt4_turbo": 0.3, "claude3_sonnet": 0.3, "gemini_pro": 0.25},
            "vector_db_config": {"index_path": "./vector_indexes", "similarity_threshold": 0.7}
        }
    except Exception as e:
        print(f"Error loading configuration: {e}")
        raise
    
    # Initialize the enhanced router with loaded configuration
    router = VectorDBEnhancedRouter(
        llm_configs=config["llm_configs"],
        base_allocation=config["base_allocation"],
        vector_db_config=config["vector_db_config"],
        config=config
    )
    
    # Load human-curated evaluation set
    print("\\nLoading human evaluation set...")
    eval_set = create_human_eval_set()
    
    try:
        router.vector_db_evaluator.ingest_human_eval_set(eval_set)
        print("Human evaluation set loaded successfully")
    except KeyboardInterrupt:
        print("\\nWarning: User interrupted ColBERT setup - continuing with fallback...")
    except Exception as e:
        print(f"Warning: Evaluation set loading failed: {str(e)[:100]}...")
        print("Continuing with fallback similarity calculation...")
    
    # Test queries that should match our evaluation set
    test_queries = [
        "How does machine learning work?",
        "Create a function to compute fibonacci sequence",
        "What is the difference between supervised and unsupervised ML?",
        "Best practices for handling missing data",
        "Explain RESTful web services"
    ]
    
    print("\\n=== Testing Router with Vector Database Evaluation ===")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\\n--- Query {i}: {query} ---")
        
        try:
            # Route request with evaluation
            response, quality_score = router.route_request_with_evaluation(query)
            
            print(f"Selected LLM: {response.llm_id}")
            print(f"Response: {response.content[:150]}...")
            print(f"Generation Time: {response.generation_time:.2f}s")
            print(f"Quality Score: {quality_score:.3f}")
            
            # Show similar examples found
            similar = router.vector_db_evaluator.find_similar_examples(query, top_k=2)
            if similar:
                print("\\nSimilar examples found:")
                for record, similarity in similar:
                    print(f"  - {record.id} (similarity: {similarity:.3f}): {record.prompt[:50]}...")
                    
        except Exception as e:
            print(f"Error: Error processing query: {e}")
        
        # Small delay between requests
        time.sleep(0.5)
    
    # Display final statistics
    print("\\n=== Final Router Statistics ===")
    stats = router.get_enhanced_stats()
    
    print("\\nCurrent Allocation:")
    for llm_id, allocation in stats["current_allocation"].items():
        print(f"  {llm_id}: {allocation:.3f}")
    
    print("\\nPerformance Summary:")
    for llm_id, perf in stats["performance_summary"].items():
        print(f"  {llm_id}:")
        print(f"    Avg Performance: {perf['avg_performance']:.3f}")
        print(f"    Avg Quality: {perf['avg_quality']:.3f}")
        print(f"    Avg Response Time: {perf['avg_response_time']:.2f}s")
        print(f"    Total Requests: {perf['total_requests']}")
        print(f"    Errors: {perf['error_count']}")
    
    print("\\nVector Database Statistics:")
    vdb_stats = stats["vector_db_stats"]
    print(f"  Total Evaluation Records: {vdb_stats['total_eval_records']}")
    print(f"  ColBERT Available: {vdb_stats['colbert_available']}")
    print(f"  Similarity Threshold: {vdb_stats['similarity_threshold']}")
    
    print("\\n=== Demo Complete ===")


if __name__ == "__main__":
    main()