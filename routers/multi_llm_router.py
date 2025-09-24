"""
Multi-LLM Router with Control Theory Based Load Balancing.

This module implements the core routing logic using PID controllers
for adaptive load balancing across multiple LLM endpoints.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque

from core.data_models import LLMResponse, LLMRequest as CoreLLMRequest
from llm_providers.base_provider import LLMRequest as ProviderLLMRequest
from core.pid_controller import PIDController


class MultiLLMRouter:
    """Multi-LLM Router with Control Theory Based Load Balancing"""
    
    def __init__(self, llm_configs: Optional[Dict[str, Dict]] = None, base_allocation: Optional[Dict[str, float]] = None, 
                 providers: Optional[Dict[str, Any]] = None, use_real_providers: bool = False):
        """
        Initialize MultiLLMRouter with support for both mock and real providers.
        
        Args:
            llm_configs: Mock LLM configurations (used when use_real_providers=False)
            base_allocation: Allocation percentages for LLMs/providers
            providers: Real provider instances (used when use_real_providers=True)
            use_real_providers: Whether to use real providers or mock configs
        """
        self.use_real_providers = use_real_providers
        
        if use_real_providers and providers:
            # Use real providers
            self.providers = providers
            self.llm_configs = llm_configs  # Keep llm_configs for enhanced features
            self.llm_ids = list(providers.keys())
            print(f"Initialized MultiLLMRouter with {len(self.llm_ids)} real providers: {self.llm_ids}")
            print(f"Base allocation: {base_allocation}")
        else:
            # Use mock configs (backward compatibility)
            if not llm_configs:
                raise ValueError("llm_configs must be provided when use_real_providers=False")
            self.llm_configs = llm_configs
            self.providers = None
            self.llm_ids = list(llm_configs.keys())
            print(f"Initialized MultiLLMRouter with {len(self.llm_ids)} mock LLMs: {self.llm_ids}")
            print(f"Initialized MultiLLMRouter with {len(self.llm_ids)} mock LLMs: {self.llm_ids}")
        
        # Initialize allocations
        if base_allocation:
            self.base_allocation = base_allocation
        else:
            # Equal allocation by default
            equal_share = 1.0 / len(self.llm_ids)
            self.base_allocation = {llm_id: equal_share for llm_id in self.llm_ids}
        
        self.current_allocation = self.base_allocation.copy()
        
        # Performance tracking
        self.performance_history = defaultdict(lambda: deque(maxlen=100))
        self.response_times = defaultdict(lambda: deque(maxlen=50))
        self.quality_scores = defaultdict(lambda: deque(maxlen=50))
        self.error_counts = defaultdict(int)
        
        # PID Controllers for each LLM
        self.pid_controllers = {}
        for llm_id in self.llm_ids:
            self.pid_controllers[llm_id] = PIDController(
                kp=0.5,  # Moderate proportional response
                ki=0.1,  # Slow integral correction
                kd=0.05, # Mild derivative response
                target=self.base_allocation[llm_id],
                output_limits=(-0.3, 0.3)  # Limit allocation changes
            )
        
        # Adaptive learning parameters
        self.learning_rate = 0.01
        self.quality_weight = 0.7
        self.speed_weight = 0.3
        self.min_allocation = 0.05  # Minimum 5% allocation
        
        print(f"Base allocation: {self.base_allocation}")
    
    def mock_llm_call(self, llm_id: str, prompt: str) -> LLMResponse:
        """Mock LLM API call with realistic behavior simulation"""
        config = self.llm_configs[llm_id]
        
        # Simulate different response characteristics
        base_time = config.get('base_response_time', 1.0)
        quality_factor = config.get('quality_factor', 1.0)
        
        # Add some randomness
        response_time = base_time + np.random.exponential(0.3)
        
        # Simulate occasional errors
        if np.random.random() < config.get('error_rate', 0.01):
            raise Exception(f"Mock error from {llm_id}")
        
        # Generate mock response
        mock_responses = [
            f"This is a response from {llm_id} model. The query was about: {prompt[:50]}...",
            f"{llm_id} processed your request: {prompt[:30]}... with high quality output.",
            f"Response from {llm_id}: Analyzing the prompt '{prompt[:40]}...' yields the following insights..."
        ]
        
        content = np.random.choice(mock_responses)
        token_count = len(content.split()) + np.random.randint(10, 100)
        
        time.sleep(response_time)  # Simulate actual response time
        
        return LLMResponse(
            content=content,
            generation_time=response_time,
            token_count=token_count,
            llm_id=llm_id,
            metadata={"quality_hint": quality_factor}
        )
    
    def real_provider_call(self, provider_name: str, prompt: str) -> LLMResponse:
        """Make actual call to real provider instance"""
        if not self.use_real_providers or not self.providers:
            raise ValueError("Real providers not configured")
        
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not found")
        
        provider = self.providers[provider_name]
        start_time = time.time()
        
        try:
            # Create LLMRequest object for the provider (using provider format)
            request = ProviderLLMRequest(
                prompt=prompt,
                max_tokens=4096,
                temperature=0.7
            )
            
            # Call the provider's generate_response method
            provider_response = provider.generate_response(request)
            
            # Convert provider response to router format
            router_response = LLMResponse(
                content=provider_response.content,
                generation_time=provider_response.generation_time,
                token_count=provider_response.token_count,
                llm_id=provider_name,  # Use provider name as llm_id
                metadata=provider_response.metadata or {},
                cost=provider_response.total_cost  # Map total_cost to cost
            )
            
            return router_response
            
        except Exception as e:
            # Convert provider exceptions to our format
            response_time = time.time() - start_time
            raise Exception(f"Real provider error from {provider_name}: {str(e)}")
    
    def calculate_composite_score(self, llm_id: str) -> float:
        """Calculate composite performance score for an LLM"""
        if llm_id not in self.performance_history:
            return 0.5  # Neutral score for new LLMs
        
        recent_performances = list(self.performance_history[llm_id])
        if not recent_performances:
            return 0.5
        
        # Get recent quality and speed metrics
        recent_quality = list(self.quality_scores[llm_id]) if self.quality_scores[llm_id] else [0.5]
        recent_times = list(self.response_times[llm_id]) if self.response_times[llm_id] else [1.0]
        
        # Calculate averages
        avg_quality = np.mean(recent_quality)
        avg_time = np.mean(recent_times)
        
        # Normalize speed score (lower time = higher score)
        max_time = 5.0  # Assume 5s is very slow
        speed_score = max(0.0, (max_time - avg_time) / max_time)
        
        # Composite score
        composite = (self.quality_weight * avg_quality + 
                    self.speed_weight * speed_score)
        
        return np.clip(composite, 0.0, 1.0)
    
    def update_allocation_with_pid(self):
        """Update allocation using PID controllers"""
        # Calculate current performance for each LLM
        current_performance = {}
        for llm_id in self.llm_ids:
            current_performance[llm_id] = self.calculate_composite_score(llm_id)
        
        # Update PID controllers
        pid_adjustments = {}
        for llm_id in self.llm_ids:
            current_score = current_performance[llm_id]
            adjustment = self.pid_controllers[llm_id].update(current_score)
            pid_adjustments[llm_id] = adjustment
        
        # Apply PID adjustments to allocation
        new_allocation = {}
        for llm_id in self.llm_ids:
            base_alloc = self.base_allocation[llm_id]
            adjustment = pid_adjustments[llm_id] * self.learning_rate
            new_allocation[llm_id] = base_alloc + adjustment
        
        # Normalize and apply minimum constraints
        total_allocation = sum(new_allocation.values())
        if total_allocation > 0:
            for llm_id in self.llm_ids:
                new_allocation[llm_id] = max(
                    self.min_allocation,
                    new_allocation[llm_id] / total_allocation
                )
            
            # Renormalize after applying minimum constraints
            total_allocation = sum(new_allocation.values())
            for llm_id in self.llm_ids:
                new_allocation[llm_id] /= total_allocation
        
        self.current_allocation = new_allocation
    
    def select_llm(self, prompt: str) -> str:
        """Select LLM based on current allocation using weighted random selection"""
        llm_ids = list(self.current_allocation.keys())
        weights = list(self.current_allocation.values())
        
        # Handle edge case where all weights are zero
        if sum(weights) == 0:
            weights = [1.0] * len(weights)
        
        selected_llm = np.random.choice(llm_ids, p=weights)
        return selected_llm
    
    ''' Initial Performance Example:
        GPT-4: Quality=0.85, Speed=1.2s → Composite=0.75
        Claude: Quality=0.80, Speed=0.8s → Composite=0.74  
        Gemini: Quality=0.70, Speed=0.6s → Composite=0.67
    
    GPT-4: Quality drops to 0.60, Speed increases to 2.5s → Composite=0.45
    
    # GPT-4 PID Controller
        target = 0.33  # Target allocation
        current_performance = 0.45
        error = 0.33 - 0.45 = -0.12  # Negative error = underperforming

    # PID Components:
        P = 0.5 × (-0.12) = -0.06   # Immediate 6% reduction signal
        I = accumulated error over time  # Builds up if error persists
        D = rate of change           # Responds to rapid degradation

    adjustment = P + I + D = -0.08  # 8% reduction signal
    
    # Apply adjustment with learning rate
        new_allocation["gpt4"] = 0.33 + (-0.08 × 0.01) = 0.3292  # Slight reduction 

    # Normalize across all LLMs
        gpt4:   32.2% → 30.8% (decreased)
        claude: 33.3% → 34.6% (increased) 
        gemini: 33.3% → 34.6% (increased)
    '''
    def route_request(self, prompt: str, quality_score: Optional[float] = None) -> LLMResponse:
        """Route request to selected LLM and update performance metrics"""
        selected_llm = self.select_llm(prompt)
        
        start_time = time.time()
        try:
            # Use real provider or mock call based on configuration
            if self.use_real_providers:
                response = self.real_provider_call(selected_llm, prompt)
            else:
                response = self.mock_llm_call(selected_llm, prompt)
            
            # Record successful performance
            actual_time = time.time() - start_time
            self.response_times[selected_llm].append(actual_time)
            
            # Use provided quality score or estimate from metadata
            if quality_score is not None:
                quality = quality_score
            else:
                # Fallback quality estimation
                if self.use_real_providers:
                    # For real providers, use a basic quality estimation
                    quality = 0.8 + np.random.normal(0, 0.1)  # Assume good quality with some variance
                else:
                    # Original mock quality estimation
                    quality = response.metadata.get('quality_hint', 0.5) + np.random.normal(0, 0.1)
                quality = np.clip(quality, 0.0, 1.0)
            
            self.quality_scores[selected_llm].append(quality)
            
            # Calculate composite performance score
            performance = self.calculate_composite_score(selected_llm)
            self.performance_history[selected_llm].append(performance)
            
            # Update allocation using PID control
            self.update_allocation_with_pid()
            
            return response
            
        except Exception as e:
            # Record error
            self.error_counts[selected_llm] += 1
            
            # Penalize performance for errors
            self.quality_scores[selected_llm].append(0.1)  # Low quality for errors
            performance = self.calculate_composite_score(selected_llm)
            self.performance_history[selected_llm].append(performance)
            
            # Update allocation
            self.update_allocation_with_pid()
            
            print(f"Error from {selected_llm}: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current router statistics"""
        stats = {
            "current_allocation": self.current_allocation.copy(),
            "error_counts": dict(self.error_counts),
            "performance_summary": {}
        }
        
        for llm_id in self.llm_ids:
            perf_history = list(self.performance_history[llm_id])
            quality_history = list(self.quality_scores[llm_id])
            time_history = list(self.response_times[llm_id])
            
            stats["performance_summary"][llm_id] = {
                "avg_performance": np.mean(perf_history) if perf_history else 0.0,
                "avg_quality": np.mean(quality_history) if quality_history else 0.0,
                "avg_response_time": np.mean(time_history) if time_history else 0.0,
                "total_requests": len(perf_history),
                "error_count": self.error_counts[llm_id]
            }
        
        return stats