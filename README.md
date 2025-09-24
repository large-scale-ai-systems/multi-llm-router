# LLM Router - Intelligent Multi-Provider AI System

A sophisticated multi-LLM routing system that uses **Control Theory (PID Controllers)** and **Vector Database evaluation** to dynamically allocate requests across multiple Language Models based on real-time performance metrics, cost efficiency, and quality assessment.

## Key Features

### Core Capabilities
- **Multi-Provider Support**: Pluggable any LLM including frontier OpenAI (GPT-4o), Anthropic (Claude), Google (Gemini), AWS Bedrock
- **Intelligent Routing**: PID controller-based adaptive load balancing with performance optimization
- **Vector Database Integration**: Pluggable. Default integrated with ColBERT-powered semantic similarity matching with FAISS fallback
- **Real-time Quality Assessment**: Response evaluation against human-curated golden standards
- **Cost Optimization**: Budget-conscious routing with cost efficiency tracking
- **Selection Transparency**: Detailed explanations of why each LLM was chosen
- **Cross-Platform**: Windows (FAISS) and Linux (ColBERT) optimized deployments
- **Production Ready**: RESTful API with comprehensive monitoring and statistics

### Advanced Features
- **Enhanced Query API**: Complete selection statistics and comparative analysis
- **Configurable Weights**: Golden similarity (40%), relative rank (20%), cost efficiency (15%), time efficiency (15%), uniqueness (10%)
- **Adaptive Learning**: PID controllers that learn from performance history
- **Graceful Fallbacks**: Token-overlap similarity when vector databases unavailable
- **Comprehensive Tuning**: Detailed configuration guide for different use cases
- **Health Monitoring**: Real-time provider status and performance tracking

## Architecture Overview

### Core Components
```
llm_router/
├── core/
│   ├── data_models.py          # Core data structures and models
│   └── pid_controller.py       # Control theory implementation
├── routers/
│   ├── multi_llm_router.py     # Base PID-controlled routing
│   └── vector_enhanced_router.py # Vector DB enhanced routing
├── vector_db/
│   ├── abstract_vector_evaluator.py # Base evaluator interface
│   ├── faiss_evaluator.py     # Windows-optimized FAISS implementation
│   ├── colbert_evaluator.py   # Linux-optimized ColBERT implementation
│   ├── token_overlap_evaluator.py # Universal fallback evaluator
│   └── factory.py             # Auto-selection factory pattern
├── llm_providers/
│   ├── base_provider.py       # Abstract provider interface
│   ├── factory.py             # Provider factory and manager
│   ├── openai/gpt4o_provider.py # GPT-4o implementation
│   ├── anthropic/claude_provider.py # Claude implementation
│   └── google/gemini_provider.py # Gemini implementation
├── fastapi_app.py             # RESTful API server
├── main.py                    # Demo and utilities
├── config.ini                 # Configuration management
└── data/
    ├── human_eval_set.csv     # Human evaluation standards
    └── vector_indexes/        # Pre-built vector indexes
```

### Control Theory Implementation
The system implements **Proportional-Integral-Derivative (PID)** control theory for dynamic LLM routing:
- **Proportional (P)**: Direct response to current performance deviation
- **Integral (I)**: Accumulates historical routing decisions through persistent bias updates  
- **Derivative (D)**: Adaptive learning rate schedules responding to allocation changes

### Vector Database Evaluation
Enhanced with **Vector Database** evaluation for real-time quality assessment:
- **FAISS**: High-performance Windows-compatible vector search (~58s indexing, ~0.1s search)
- **ColBERT**: Linux-optimized implementation with C++ acceleration
- **Token Overlap**: Universal fallback using Jaccard/Cosine similarity (cross-platform)
- **Factory Pattern**: Automatic platform detection and implementation selection

## Quick Start

### Prerequisites
- **Python 3.10+** installed on your system
- **VS Code** with Python extension (recommended)
- **Virtual environment** for dependency management

### Installation

1. **Clone and Setup Environment**
   ```bash
   git clone <repository-url>
   cd llm_router
   
   # Create and activate virtual environment
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   # Core dependencies
   pip install -r requirements.txt
   
   # Provider-specific dependencies (optional)
   pip install -r requirements_providers.txt
   ```

3. **Configure API Keys**
   ```bash
   # Copy configuration template
   cp config.sample.ini config.ini
   
   # Edit config.ini with your API keys
   [OPENAI]
   api_key = your_openai_api_key_here
   
   [ANTHROPIC]  
   api_key = your_anthropic_api_key_here
   
   [GOOGLE]
   api_key = your_google_api_key_here
   ```

### Running the System

**Start the API Server:**
```bash
# Debug mode (detailed logging)
python debug_server.py --mode debug

# Production mode  
python start_server.py
```

**Test the API:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain machine learning in simple terms", "max_tokens": 150}'
```

**Basic Demo:**
```bash
python main.py
```

## API Reference

### Query Endpoint
**POST** `/query`

**Request:**
```json
{
  "query": "Explain machine learning",
  "max_tokens": 150,
  "temperature": 0.7,
  "use_multi_llm_comparison": true
}
```

**Response:**
```json
{
  "response_content": "Machine learning is a branch of artificial intelligence...",
  "selected_llm": "azure-openai-gpt4o-mini", 
  "quality_score": 0.823,
  "total_cost": 0.000045,
  "selection_details": {
    "routing_mode": "multi_llm",
    "selection_reason": "Selected azure-openai-gpt4o-mini with composite score 0.745",
    "similarity_threshold": 0.5,
    "evaluated_llms": ["azure-openai-gpt4o", "azure-openai-gpt4o-mini", "bedrock-claude-sonnet-4"],
    "all_scores": {
      "azure-openai-gpt4o-mini": {
        "golden_similarity": 0.723,
        "composite_score": 0.745,
        "ranking_position": 1
      }
    },
    "selection_criteria": {
      "primary_metric": "composite_score",
      "weighting_strategy": "accuracy_first_with_efficiency_factors"
    },
    "comparative_analysis": {
      "recommendations": [
        "Best overall accuracy: azure-openai-gpt4o-mini (0.723)",
        "Most cost-efficient: azure-openai-gpt4o-mini (0.856)"
      ]
    }
  }
}
```

### Health Check Endpoint
**GET** `/health`

Returns system health status, provider availability, and performance metrics.

### Statistics Endpoint  
**GET** `/stats`

Returns comprehensive routing statistics, allocation percentages, and performance data.

## Configuration Guide

### Selection Weights Configuration
Control how different factors influence LLM selection (must sum to 1.0):

```ini
[SELECTION_WEIGHTS]
golden_similarity = 0.4    # Accuracy against human standards
relative_rank = 0.2        # Performance vs other LLMs  
cost_efficiency = 0.15     # Cost-effectiveness
time_efficiency = 0.15     # Response speed
uniqueness = 0.1          # Diversity and creativity
```

### Provider Configuration
```ini
[PROVIDER_ALLOCATIONS]
azure-openai-gpt4o = 0.4
azure-openai-gpt4o-mini = 0.35
bedrock-claude-sonnet-4 = 0.25

[PID_CONTROLLER]
kp = 1.0          # Proportional gain - responsiveness
ki = 0.1          # Integral gain - historical bias correction
kd = 0.05         # Derivative gain - stability and prediction
setpoint = 0.5    # Target performance level
```

### Performance Tuning
```ini
[PERFORMANCE]
max_acceptable_response_time = 30.0
time_penalty_threshold = 10.0
time_penalty_factor = 0.1
ignore_performance_time = false
focus_on_accuracy_only = false
use_cost_optimization = true

[VECTOR_DB]
similarity_threshold = 0.5
vector_db_type = faiss     # faiss, colbert, token_overlap
```

### Use Case Scenarios

#### Real-time Customer Service
```ini
[SELECTION_WEIGHTS]
time_efficiency = 0.35     # Prioritize speed
golden_similarity = 0.3
cost_efficiency = 0.1
relative_rank = 0.2
uniqueness = 0.05

[PERFORMANCE]
max_acceptable_response_time = 8.0
time_penalty_threshold = 3.0
```

#### Technical Documentation  
```ini
[SELECTION_WEIGHTS]
golden_similarity = 0.6    # Accuracy critical
relative_rank = 0.25
time_efficiency = 0.05
cost_efficiency = 0.05
uniqueness = 0.05

[PERFORMANCE]
focus_on_accuracy_only = true
max_acceptable_response_time = 45.0
```

#### High-Volume Processing
```ini
[SELECTION_WEIGHTS]
cost_efficiency = 0.4      # Cost critical
golden_similarity = 0.2
time_efficiency = 0.2
relative_rank = 0.15
uniqueness = 0.05
```

#### Creative Content Generation
```ini
[SELECTION_WEIGHTS]
uniqueness = 0.3          # Prioritize diversity
golden_similarity = 0.25
cost_efficiency = 0.2
time_efficiency = 0.1
relative_rank = 0.15

[PERFORMANCE]
ignore_performance_time = true
```

## System Workflow

1. **Initialization**: Load LLM provider configurations and base allocations
2. **Evaluation Set Loading**: Import human evaluation data from CSV
3. **Vector DB Setup**: Initialize FAISS/ColBERT indexing with automatic fallback
4. **Request Processing**:
   - **Single-LLM Mode**: Fast PID-based selection for simple queries
   - **Multi-LLM Mode**: Comprehensive evaluation against all providers
   - Execute request and measure performance (response time, cost, quality)
   - Evaluate response quality against golden standards using vector similarity
5. **PID Control Update**: Adjust future allocations based on performance metrics
6. **Statistics Tracking**: Monitor and report system performance over time
## Example Usage

### Basic Router Usage
```python
from routers.vector_enhanced_router import VectorDBEnhancedRouter
from llm_providers import create_provider, ProviderManager

# Method 1: Using Real Providers
manager = ProviderManager()
manager.add_provider("gpt4o", "openai", "your-api-key", "gpt-4o")
manager.add_provider("claude", "anthropic", "your-api-key")

router = VectorDBEnhancedRouter(
    provider_manager=manager,
    base_allocation={"gpt4o": 0.6, "claude": 0.4}
)

# Route request with quality evaluation  
response, quality_score = router.route_request_with_evaluation(
    "Explain machine learning in simple terms"
)
print(f"Selected: {response.llm_id}")
print(f"Quality Score: {quality_score:.3f}")
print(f"Response: {response.content}")
```

### Mock System Demo
```python
from main import load_config

# Load configuration (defaults to mock providers)
router, eval_set = load_config(use_real_providers=False)

# Test queries with different complexities
test_queries = [
    "What is Python?",
    "Explain quantum computing in detail",
    "Write a sorting algorithm in Python"
]

for query in test_queries:
    response, quality = router.route_request_with_evaluation(query)
    print(f"\nQuery: {query}")
    print(f"Selected LLM: {response.llm_id}")
    print(f"Quality Score: {quality:.3f}")
    print(f"Response Time: {response.response_time:.2f}s")
    
    # View current allocation after PID adjustment
    stats = router.get_detailed_statistics()
    print("Current Allocations:", stats["allocation_percentages"])
```

### FastAPI Client Usage
```python
import requests

# Query the API server
response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "How does machine learning work?",
        "max_tokens": 200,
        "temperature": 0.7,
        "use_multi_llm_comparison": True
    }
)

result = response.json()
print(f"Response: {result['response_content']}")
print(f"Selected LLM: {result['selected_llm']}")
print(f"Quality Score: {result['quality_score']}")

# View detailed selection reasoning
selection = result['selection_details']
print(f"Routing Mode: {selection['routing_mode']}")
print(f"Reason: {selection['selection_reason']}")
print("Evaluated LLMs:", selection['evaluated_llms'])
```

## Performance Metrics

The system tracks comprehensive performance metrics:

### Core Metrics
- **Allocation Distribution**: Real-time routing percentages per LLM provider
- **Response Times**: Average, min, max latency per LLM endpoint
- **Quality Scores**: Semantic similarity to human-curated golden standards
- **Error Rates**: Failed requests and timeout statistics per LLM
- **Cost Tracking**: Total costs, cost per request, cost efficiency ratios
- **Composite Performance**: Weighted combination of quality, speed, and cost

### Vector Database Metrics
- **Evaluation Records**: Number of human evaluation standards ingested
- **Index Performance**: FAISS/ColBERT indexing and search times
- **Similarity Scores**: Distribution of golden similarity matches (0.3-0.7 typical)
- **Fallback Usage**: When token overlap similarity is used vs vector DB

### Selection Statistics  
- **Single vs Multi-LLM**: Percentage of requests using each routing mode
- **Selection Factors**: How often each weight factor (accuracy, cost, speed) drives decisions
- **Provider Utilization**: Request distribution across different LLM providers
- **Quality Distribution**: Histogram of response quality scores over time

### Example Statistics Output
```json
{
  "allocation_percentages": {
    "azure-openai-gpt4o": 0.38,
    "azure-openai-gpt4o-mini": 0.41, 
    "bedrock-claude-sonnet-4": 0.21
  },
  "response_times": {
    "azure-openai-gpt4o": {"avg": 2.1, "min": 0.8, "max": 4.2},
    "azure-openai-gpt4o-mini": {"avg": 1.3, "min": 0.5, "max": 2.8}
  },
  "quality_scores": {
    "mean_similarity": 0.685,
    "accuracy_leader": "azure-openai-gpt4o-mini",
    "cost_leader": "azure-openai-gpt4o-mini"
  },
  "vector_db": {
    "total_records": 23,
    "index_type": "faiss",
    "avg_search_time": 0.1,
    "similarity_threshold": 0.5
  },
  "cost_analysis": {
    "total_cost": 0.0234,
    "avg_cost_per_request": 0.000018,
    "cost_efficiency_leader": "azure-openai-gpt4o-mini"
  }
}
```

## Provider Integration

### Supported LLM Providers

#### OpenAI GPT Models
- **Models**: `gpt-4o`, `gpt-4o-mini`
- **Capabilities**: Text, Vision, JSON mode, Function calling
- **Context**: Up to 128K tokens
- **Cost**: $0.0025-$0.01 per 1K input tokens

#### Anthropic Claude
- **Models**: `claude-3-5-sonnet-20241022`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`
- **Capabilities**: Text, Vision, Document analysis, Code generation
- **Context**: Up to 200K tokens
- **Cost**: $0.003-$0.015 per 1K input tokens

#### Google Gemini  
- **Models**: `gemini-1.5-pro`, `gemini-1.5-flash`, `gemini-pro`
- **Capabilities**: Text, Vision, Audio, Video, Multimodal
- **Context**: Up to 2M tokens
- **Cost**: $0.001-$0.007 per 1K input tokens

#### AWS Bedrock
- **Models**: Various Claude, Llama, and proprietary models
- **Capabilities**: Enterprise-grade with compliance features
- **Context**: Varies by model
- **Cost**: Pay-per-use with volume discounts

### Adding New Providers
```python
# 1. Create provider implementation
from llm_providers.base_provider import BaseLLMProvider

class CustomProvider(BaseLLMProvider):
    def generate_response(self, request: LLMRequest) -> LLMResponse:
        # Implement API integration
        pass
    
    def get_model_info(self) -> ModelInfo:
        # Return model capabilities and costs
        pass

# 2. Register with factory
from llm_providers.factory import ProviderManager

manager = ProviderManager()
manager.add_provider("custom", "custom", api_key, model_name)
```

### Cost Estimation & Tracking
```python
# Real-time cost estimation
provider = create_provider("openai", api_key, "gpt-4o")
request = LLMRequest(prompt="Explain AI", max_tokens=100)

input_cost, total_cost = provider.estimate_cost(request)
print(f"Estimated cost: ${total_cost:.4f}")

# Actual cost tracking
response = provider.generate_response(request)
print(f"Actual cost: ${response.total_cost:.4f}")

# Usage statistics
stats = provider.get_stats()
print(f"Total requests: {stats['request_count']}")
print(f"Total cost: ${stats['total_cost']:.4f}")
```

## Advanced Configuration

### Custom Evaluation Data
Create your own golden standards in `data/human_eval_set.csv`:
```csv
id,prompt,golden_output,category,difficulty,source
custom001,"Explain photosynthesis","Photosynthesis is the process by which plants convert light energy into chemical energy...",biology,medium,domain_expert
custom002,"Write Python hello world","print('Hello, World!')",programming,easy,software_engineer
custom003,"Describe quantum entanglement","Quantum entanglement is a phenomenon where two particles become correlated...",physics,hard,research_scientist
```

### PID Controller Tuning
```python
# Aggressive tuning for fast adaptation
PIDController(
    kp=2.0,     # High responsiveness
    ki=0.3,     # Strong bias correction  
    kd=0.1      # High stability
)

# Conservative tuning for stability
PIDController(
    kp=0.5,     # Gentle responsiveness
    ki=0.05,    # Minimal bias correction
    kd=0.02     # Light damping
)

# Real-time applications
PIDController(
    kp=1.5,     # Balanced responsiveness  
    ki=0.1,     # Standard correction
    kd=0.05     # Moderate stability
)
```

### Vector Database Optimization
```ini
# High precision configuration
[VECTOR_DB]
similarity_threshold = 0.8      # Strict matching
vector_db_type = colbert       # Best accuracy (Linux)

# High recall configuration  
[VECTOR_DB]
similarity_threshold = 0.3      # Loose matching
vector_db_type = token_overlap  # Fast fallback

# Production balanced
[VECTOR_DB]
similarity_threshold = 0.5      # Moderate matching
vector_db_type = faiss         # Good performance (Windows)
```

## � Linux Deployment

### System Requirements
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev
sudo apt install -y build-essential gcc g++ ninja-build

# CentOS/RHEL
sudo yum groupinstall -y "Development Tools"
sudo yum install -y python3.10 python3.10-devel
```

### Performance Optimization
```bash
# Enable ColBERT C++ acceleration (Linux only)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True

# Verify ColBERT C++ extensions
python -c "
from colbert.modeling.colbert import ColBERT
from colbert.infra import ColBERTConfig
model = ColBERT('answerdotai/answerai-colbert-small-v1')
print('ColBERT C++ extensions working!')
"
```

### Expected Performance Improvements
| Feature | Windows (FAISS) | Linux (ColBERT) |
|---------|----------------|-----------------|
| Vector Indexing | ~58s | ~15-20s |
| Search Speed | ~100ms | ~50ms |
| Memory Usage | Higher | Lower |
| Accuracy | Good | Excellent |

## Testing & Validation

### Unit Testing
```bash
# Run comprehensive test suite
python -m pytest tests/ -v

# Test specific components
python -c "
from core.pid_controller import PIDController
controller = PIDController(kp=1.0, ki=0.1, kd=0.01)
assert controller.update(0.8) is not None
print('PID Controller: OK')
"

# Test vector database fallbacks
python -c "
from vector_db.factory import create_vector_evaluator
evaluator = create_vector_evaluator('auto')
print(f'Vector DB Type: {type(evaluator).__name__}')
"

# Test provider integration
python demos/demo_provider_integration.py
```

### Performance Benchmarking
```bash
# Benchmark vector database performance
python -c "
import time
from vector_db.factory import create_vector_evaluator
from main import load_human_eval_set

evaluator = create_vector_evaluator('faiss')
eval_set = load_human_eval_set()

start = time.time()
evaluator.ingest_human_eval_set(eval_set[:10])
index_time = time.time() - start

start = time.time() 
results = evaluator.find_similar_examples('test query', top_k=5)
search_time = time.time() - start

print(f'Indexing: {index_time:.2f}s, Search: {search_time:.3f}s')
"
```

### Load Testing
```bash
# Test API server under load
pip install locust

# Create locustfile.py
cat > locustfile.py << EOF
from locust import HttpUser, task

class QueryUser(HttpUser):
    @task
    def query_api(self):
        self.client.post("/query", json={
            "query": "What is machine learning?",
            "max_tokens": 100
        })
EOF

# Run load test
locust -f locustfile.py --host=http://localhost:8000
```

## Troubleshooting

### Performance Issues

#### Slow Response Times
**Symptoms**: High average response times, timeout errors
**Solutions**:
1. Increase `time_efficiency` weight to 0.3-0.4
2. Lower `max_acceptable_response_time` in config
3. Use faster models (gpt-4o-mini vs gpt-4o)
4. Enable response caching for repeated queries

#### High Costs
**Symptoms**: Budget overruns, expensive model overuse
**Solutions**:
1. Increase `cost_efficiency` weight to 0.3-0.4
2. Adjust base allocations to favor cheaper models
3. Set stricter `max_tokens` limits
4. Use cost-optimized configuration presets

#### Poor Quality Responses  
**Symptoms**: Low similarity scores, user dissatisfaction
**Solutions**:
1. Increase `golden_similarity` weight to 0.5-0.7
2. Set `focus_on_accuracy_only = true`
3. Expand human evaluation dataset
4. Increase `similarity_threshold` to 0.7+

#### Vector Database Issues
**Symptoms**: Search failures, index corruption
**Solutions**:
1. Clear vector indexes: `rm -rf data/vector_indexes/*`
2. Switch to fallback mode: `vector_db_type = token_overlap`
3. Rebuild indexes: `python -c "from main import rebuild_indexes; rebuild_indexes()"`
4. Check disk space and permissions

### Common Error Messages

**"ColBERT not available, using fallback"**
- **Cause**: ColBERT installation incomplete or Windows limitations
- **Solution**: System automatically uses token overlap similarity
- **Impact**: Slightly reduced accuracy, no functionality loss

**"Provider authentication failed"**
- **Cause**: Invalid API key or expired credentials
- **Solution**: Update API keys in `config.ini`
- **Check**: Verify key format and provider account status

**"PID controller unstable oscillations"**
- **Cause**: PID gains too high for system dynamics
- **Solution**: Reduce `kp`, `ki`, `kd` values by 50%
- **Prevention**: Start with conservative gains and increase gradually

**"Vector index not found"**
- **Cause**: Missing or corrupted vector database files
- **Solution**: Delete `data/vector_indexes/` and restart system
- **Prevention**: Regular backup of index files

## Monitoring & Analytics

### Real-time Monitoring
```python
# Monitor system health
import time
from fastapi_app import app

def monitor_system():
    while True:
        stats = app.router.get_detailed_statistics()
        
        print(f"Active LLMs: {len(stats['allocation_percentages'])}")
        print(f"Avg Response Time: {stats['avg_response_time']:.2f}s")
        print(f"Quality Score: {stats['avg_quality_score']:.3f}")
        print(f"Total Cost: ${stats['total_cost']:.4f}")
        
        time.sleep(60)  # Check every minute

monitor_system()
```

### Performance Analytics
```python
# Analyze routing decisions over time
import pandas as pd
from datetime import datetime

def analyze_routing_history():
    # Load routing logs (implement log collection)
    logs = load_routing_logs()  
    
    df = pd.DataFrame(logs)
    
    # Provider utilization analysis
    provider_usage = df.groupby('selected_llm').size()
    print("Provider Usage:", provider_usage)
    
    # Quality trends over time
    quality_trend = df.groupby('timestamp')['quality_score'].mean()
    
    # Cost analysis
    cost_by_provider = df.groupby('selected_llm')['cost'].sum()
    
    return {
        'provider_usage': provider_usage,
        'quality_trend': quality_trend,
        'cost_analysis': cost_by_provider
    }
```

### Alerting & Notifications
```python
# Set up performance alerts
def setup_alerts():
    def check_performance():
        stats = router.get_detailed_statistics()
        
        # Alert on high costs
        if stats['total_cost'] > 10.0:
            send_alert(f"High cost alert: ${stats['total_cost']:.2f}")
        
        # Alert on poor quality
        if stats['avg_quality_score'] < 0.3:
            send_alert(f"Low quality alert: {stats['avg_quality_score']:.3f}")
        
        # Alert on slow responses
        if stats['avg_response_time'] > 30.0:
            send_alert(f"Slow response alert: {stats['avg_response_time']:.1f}s")
    
    # Schedule regular checks
    import schedule
    schedule.every(5).minutes.do(check_performance)
```

## Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "start_server.py"]
```

### Environment Variables
```bash
# Production configuration
export LLM_ROUTER_ENV=production
export LOG_LEVEL=INFO
export MAX_WORKERS=4
export VECTOR_DB_TYPE=faiss

# API Keys (use secrets management)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-...
export GOOGLE_API_KEY=...
```

### Scaling Considerations
- **Horizontal Scaling**: Deploy multiple instances behind load balancer
- **Database Sharing**: Use shared vector database for consistent routing
- **Caching**: Implement Redis for response caching
- **Monitoring**: Use Prometheus + Grafana for metrics collection

## System Evolution & Roadmap

### Current Status: Production Ready - Version 1.0

**Core Features Included in v1.0:**
- **Multi-LLM Router**: PID controller-based intelligent routing system with adaptive load balancing
- **Vector Database Integration**: Cross-platform implementation (FAISS for Windows, ColBERT for Linux)
- **Comprehensive Provider Support**: Azure OpenAI (GPT-4o, GPT-4o-mini), Anthropic Claude, Google Gemini, AWS Bedrock
- **Intelligent Routing Modes**: Auto, fast, multi-LLM comparison, and quality-focused routing
- **Quality Assessment**: Real-time evaluation against human-curated golden standards
- **Cost Optimization**: Budget-conscious routing with detailed cost tracking and efficiency metrics
- **Selection Transparency**: Detailed explanations of routing decisions with comparative analysis
- **RESTful API**: Production-ready FastAPI server with health checks and statistics endpoints
- **Configuration Management**: Flexible INI-based configuration with multiple tuning presets
- **Performance Monitoring**: Comprehensive statistics tracking and real-time performance metrics
- **Testing Suite**: 22/23 tests passing (95.7% success rate) with automated validation
- **Professional Logging**: Clean technical language suitable for enterprise environments
- **Cross-Platform Deployment**: Windows and Linux optimized with automatic fallbacks

### Version 1.0 - Complete Feature Set
All features listed above represent the complete v1.0 release. The system is production-ready with:
- Stable API interfaces
- Comprehensive documentation
- Full provider integration
- Advanced routing algorithms
- Quality assessment capabilities
- Cost optimization features
- Performance monitoring
- Professional deployment support

### Future Enhancement Roadmap
- **v1.1**: Advanced caching and response optimization
- **v1.2**: Enhanced analytics and reporting dashboards  
- **v2.0**: Custom model fine-tuning integration
- **v2.1**: Multi-modal support (text, image, audio)
- **v3.0**: Enterprise features (SSO, audit trails, compliance)

## Technical References

### Control Theory Background
- **PID Controllers**: Classical control theory for system stability and performance optimization
- **Anti-windup Protection**: Prevents integral term saturation in bounded systems
- **Tuning Methodologies**: Ziegler-Nichols, Cohen-Coon, and manual tuning approaches

### Vector Database & Information Retrieval
- **ColBERT**: Efficient dense retrieval with late interaction architecture
- **FAISS**: Facebook AI Similarity Search for high-performance vector operations
- **Semantic Similarity**: Dense embeddings vs traditional keyword-based matching
- **Retrieval-Augmented Generation**: Using retrieved context to improve LLM responses

### LLM Architecture & Routing
- **Multi-Agent Systems**: Coordinating multiple AI models for optimal performance
- **Load Balancing**: Weighted random selection with performance-based allocation
- **Quality Assessment**: Automated evaluation against human-curated golden standards
- **Cost Optimization**: Balancing performance requirements with budget constraints

## Implementation Notes

### Technical Architecture Details
The system implements several advanced features:

**Multi-LLM Comparison System**: Parallel response generation from all providers with comprehensive evaluation using golden similarity (40%), relative ranking (20%), cost efficiency (15%), time efficiency (15%), and uniqueness (10%) scoring.

**Adaptive Routing Modes**: 
- `auto`: Intelligent mode selection based on prompt complexity analysis
- `fast`: Single LLM pre-selection for speed-critical applications  
- `multi_llm`: Forced multi-provider evaluation for quality analysis
- `comprehensive`/`quality`: Full comparison mode for accuracy-critical tasks

**Provider API Compatibility**: Handles model-specific requirements like GPT-4o-mini's `max_completion_tokens` parameter instead of `max_tokens`, fixed temperature constraints, and restricted parameter handling.

**Learning and Adaptation**: PID controllers continuously adapt allocation percentages based on comparative performance results, with trend analysis and performance variance detection.

### Development References
For detailed technical implementation information, see:
- Multi-LLM comparison architecture and scoring algorithms
- API compatibility fixes for different model variants  
- Forced routing modes for deployment analysis and tuning
- Performance optimization strategies and benchmarking results

## Contributing

### Development Setup
```bash
# 1. Fork and clone repository
git clone https://github.com/your-username/llm_router.git
cd llm_router

# 2. Create development environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or .\venv\Scripts\activate  # Windows

# 3. Install development dependencies
pip install -r requirements.txt
pip install -r requirements_dev.txt  # If available

# 4. Set up pre-commit hooks
pre-commit install
```

### Contribution Guidelines
1. **Follow Architecture**: Maintain modular component separation
2. **Add Tests**: Include unit tests for new functionality  
3. **Update Documentation**: Reflect changes in README and docstrings
4. **Code Style**: Use type hints, docstrings, and consistent formatting
5. **Performance**: Consider impact on routing speed and memory usage

### Code Review Checklist
- [ ] Type hints throughout new code
- [ ] Comprehensive docstrings for public methods
- [ ] Error handling with graceful fallbacks
- [ ] Unit tests covering main functionality
- [ ] Performance impact assessment
- [ ] Documentation updates
- [ ] Backward compatibility maintained

## License

This project is open source and available under the **MIT License**.

```
MIT License

Copyright (c) 2024 LLM Router Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## Acknowledgments

- **ColBERT Team**: Efficient dense retrieval framework enabling semantic search
- **FAISS Developers**: High-performance similarity search and clustering library  
- **Control Theory Community**: Classical PID control foundations and tuning methods
- **LLM Provider APIs**: OpenAI, Anthropic, Google, AWS for making powerful models accessible
- **Open Source Community**: FastAPI, PyTorch, NumPy, and other foundational libraries

---

**Built for intelligent multi-LLM routing, cost optimization, and quality assessment in production environments.**
