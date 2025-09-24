#!/usr/bin/env python3
"""
FastAPI LLM Router Demo

This script demonstrates how to use the LLM Router FastAPI service.
It shows examples of calling all the API endpoints.
"""

import requests
import json
import time
from typing import Dict, Any


class LLMRouterClient:
    """Client for interacting with the LLM Router FastAPI service"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the service is healthy"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def initialize_system(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """Initialize the vector index"""
        data = {"force_rebuild": force_rebuild}
        response = self.session.post(f"{self.base_url}/init", json=data)
        response.raise_for_status()
        return response.json()
    
    def query_llm(self, prompt: str, max_tokens: int = None, temperature: float = None) -> Dict[str, Any]:
        """Query the LLM router"""
        data = {"prompt": prompt}
        if max_tokens:
            data["max_tokens"] = max_tokens
        if temperature:
            data["temperature"] = temperature
            
        response = self.session.post(f"{self.base_url}/query", json=data)
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        response = self.session.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()
    
    def get_simple_status(self) -> Dict[str, Any]:
        """Get simple status"""
        response = self.session.get(f"{self.base_url}/status")
        response.raise_for_status()
        return response.json()
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get router configuration"""
        response = self.session.get(f"{self.base_url}/config")
        response.raise_for_status()
        return response.json()


def format_json(data: Dict[str, Any]) -> str:
    """Format JSON data for pretty printing"""
    return json.dumps(data, indent=2, default=str)


def demo_api_usage():
    """Demonstrate the FastAPI LLM Router usage"""
    print("LLM Router FastAPI Demo")
    print("=" * 50)
    
    # Initialize client
    client = LLMRouterClient()
    
    # 1. Health Check
    print("\n1. 🏥 Health Check")
    print("-" * 20)
    try:
        health = client.health_check()
        print("Service is healthy")
        print(f"Status: {health['status']}")
        print(f"Router Initialized: {health['router_initialized']}")
        print(f"Index Loaded: {health['index_loaded']}")
    except requests.exceptions.ConnectionError:
        print("Service is not running. Please start it with: uvicorn fastapi_app:app --reload")
        return
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    # 2. System Status
    print("\n2. System Status")
    print("-" * 20)
    try:
        status = client.get_simple_status()
        print(f"Status: {status['status']}")
        print(f"Initialized: {status['initialized']}")
        print(f"Uptime: {status['uptime']:.1f} seconds")
        print(f"Version: {status['version']}")
    except Exception as e:
        print(f"Status check failed: {e}")
    
    # 3. Initialize System (if not already initialized)
    print("\n3. 🔧 Initialize System")
    print("-" * 20)
    try:
        health = client.health_check()
        if not health['router_initialized']:
            print("Initializing router...")
            init_result = client.initialize_system(force_rebuild=False)
            print("Initialization completed")
            print(f"Records: {init_result['total_records']}")
            print(f"Time: {init_result['initialization_time']:.2f}s")
            print(f"Implementation: {init_result['vector_implementation']}")
        else:
            print("Router already initialized")
    except Exception as e:
        print(f"Initialization failed: {e}")
        print("Proceeding with queries anyway...")
    
    # 4. Configuration Check
    print("\n4. ⚙️ Configuration")
    print("-" * 20)
    try:
        config = client.get_configuration()
        print("LLM Configs:")
        for llm_id, cfg in config['llm_configs'].items():
            print(f"  • {llm_id}: quality={cfg['quality_factor']}, response_time={cfg['base_response_time']}")
        print("\nBase Allocation:")
        for llm_id, alloc in config['base_allocation'].items():
            print(f"  • {llm_id}: {alloc*100:.1f}%")
    except Exception as e:
        print(f"Configuration retrieval failed: {e}")
    
    # 5. Query Examples
    print("\n5. 💬 Query Examples")
    print("-" * 20)
    
    test_queries = [
        "What is machine learning?",
        "Write a Python function to calculate fibonacci numbers",
        "Explain the difference between supervised and unsupervised learning",
        "What are the best practices for handling missing data in ML?"
    ]
    
    query_results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query[:50]}...")
        try:
            result = client.query_llm(query)
            query_results.append(result)
            
            print(f"  Selected LLM: {result['selected_llm']}")
            print(f"  Generation Time: {result['generation_time']:.3f}s")
            print(f"  Quality Score: {result['quality_score']:.3f}")
            print(f"  Response: {result['response_content'][:100]}...")
            
            # Show LLM stats
            llm_stats = result['llm_stats']
            print(f"  LLM Stats - Allocation: {llm_stats['current_allocation']:.3f}, "
                  f"Requests: {llm_stats['total_requests']}, "
                  f"Avg Quality: {llm_stats['average_quality_score']:.3f}")
            
        except Exception as e:
            print(f"  Query failed: {e}")
    
    # 6. System Statistics
    print("\n6. 📈 System Statistics")
    print("-" * 20)
    try:
        stats = client.get_stats()
        print(f"System Status: {stats['system_status']}")
        print(f"Total Queries: {stats['total_queries']}")
        print(f"Uptime: {stats['uptime_seconds']:.1f} seconds")
        
        print("\nLLM Statistics:")
        for llm_stat in stats['llm_stats']:
            print(f"  • {llm_stat['llm_id']}:")
            print(f"    - Current Allocation: {llm_stat['current_allocation']:.3f}")
            print(f"    - Total Requests: {llm_stat['total_requests']}")
            print(f"    - Success Rate: {(1-llm_stat['error_rate'])*100:.1f}%")
            print(f"    - Avg Response Time: {llm_stat['average_response_time']:.3f}s")
            print(f"    - Avg Quality Score: {llm_stat['average_quality_score']:.3f}")
        
        print(f"\nPID Controller Status:")
        pid_status = stats['pid_controller_status']
        print(f"  • Indexed Records: {pid_status['total_indexed_records']}")
        if pid_status['last_init_time']:
            init_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(pid_status['last_init_time']))
            print(f"  • Last Initialized: {init_time}")
        
    except Exception as e:
        print(f"Stats retrieval failed: {e}")
    
    # 7. Summary
    print("\n7. 📋 Demo Summary")
    print("-" * 20)
    print(f"Completed {len(query_results)} queries")
    
    if query_results:
        # Calculate some summary stats
        avg_response_time = sum(r['generation_time'] for r in query_results) / len(query_results)
        avg_quality = sum(r['quality_score'] for r in query_results) / len(query_results)
        llm_usage = {}
        for result in query_results:
            llm = result['selected_llm']
            llm_usage[llm] = llm_usage.get(llm, 0) + 1
        
        print(f"Average Response Time: {avg_response_time:.3f}s")
        print(f"Average Quality Score: {avg_quality:.3f}")
        print("LLM Usage Distribution:")
        for llm, count in llm_usage.items():
            percentage = (count / len(query_results)) * 100
            print(f"  • {llm}: {count} queries ({percentage:.1f}%)")
    
    print("\nDemo completed successfully!")
    print("\nTo interact with the API manually:")
    print("  • Health: GET  http://localhost:8000/health")
    print("  • Init:   POST http://localhost:8000/init")
    print("  • Query:  POST http://localhost:8000/query")
    print("  • Stats:  GET  http://localhost:8000/stats")
    print("  • Docs:   http://localhost:8000/docs (Swagger UI)")


if __name__ == "__main__":
    demo_api_usage()