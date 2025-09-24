"""
Vector Database Enhanced Multi-LLM Router.

This module extends the base MultiLLMRouter with vector database
evaluation capabilities for real-time quality assessment, predictive
LLM selection, and context-aware routing.
"""

from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict, deque
import numpy as np
import time

from .multi_llm_router import MultiLLMRouter
from vector_db.factory import VectorEvaluatorFactory
from core.data_models import LLMResponse, HumanEvalRecord


class VectorDBEnhancedRouter(MultiLLMRouter):
    """Enhanced Multi-LLM Router with Vector Database evaluation and predictive routing"""
    
    def __init__(self, llm_configs: Optional[Dict[str, Dict]] = None, base_allocation: Optional[Dict[str, float]] = None,
                 vector_db_config: Optional[Dict[str, Any]] = None, providers: Optional[Dict[str, Any]] = None,
                 use_real_providers: bool = False, config: Optional[Dict[str, Any]] = None):
        """
        Initialize VectorDB Enhanced Router with support for both mock and real providers.
        
        Args:
            llm_configs: Mock LLM configurations (used when use_real_providers=False)
            base_allocation: Allocation percentages for LLMs/providers
            vector_db_config: Vector database configuration
            providers: Real provider instances (used when use_real_providers=True)
            use_real_providers: Whether to use real providers or mock configs
            config: Full configuration object including SELECTION_WEIGHTS
        """
        if use_real_providers and providers:
            # Initialize with real providers
            super().__init__(llm_configs=llm_configs, base_allocation=base_allocation, 
                           providers=providers, use_real_providers=True)
        else:
            # Initialize with mock configs (backward compatibility)
            super().__init__(llm_configs=llm_configs, base_allocation=base_allocation)
        
        # Load selection weights from configuration
        self._load_selection_weights(config)
        
        # Initialize vector database evaluator using factory pattern
        db_config = vector_db_config or {}
        index_path = db_config.get('index_path', './vector_indexes')
        similarity_threshold = db_config.get('similarity_threshold', 0.7)
        
        # Use factory to get the best available evaluator for this platform
        self.vector_db_evaluator = VectorEvaluatorFactory.create_evaluator(
            index_path=index_path,
            similarity_threshold=similarity_threshold
        )
        
        # Enhanced tracking for vector-based routing
        self.llm_category_performance = defaultdict(lambda: defaultdict(lambda: deque(maxlen=20)))
        self.successful_prompt_patterns = defaultdict(list)  # LLM -> list of successful prompts
        self.category_routing_history = defaultdict(lambda: deque(maxlen=50))
        
        # Vector-based quality prediction
        self.quality_prediction_cache = {}
        self.cache_max_size = 100
        
        # Multi-objective optimization weights
        self.optimization_weights = {
            'quality': 0.5,
            'speed': 0.3,
            'cost': 0.2
        }
        
        print("VectorDB Enhanced Router initialized")
    
    def _load_selection_weights(self, config: Optional[Dict[str, Any]]) -> None:
        """
        Load selection weights from configuration with fallback to defaults.
        
        Args:
            config: Full configuration dictionary containing SELECTION_WEIGHTS section
        """
        # Default selection weights
        default_weights = {
            'golden_similarity': 0.4,
            'relative_rank': 0.2,
            'cost_efficiency': 0.15,
            'time_efficiency': 0.15,
            'uniqueness': 0.1
        }
        
        default_performance = {
            'ignore_performance_time': False,
            'focus_on_accuracy_only': False,
            'use_cost_optimization': True,
            'max_acceptable_response_time': 30.0,
            'time_penalty_threshold': 10.0,
            'time_penalty_factor': 0.1
        }
        
        if config and 'selection_weights' in config:
            weights_config = config['selection_weights']
            # Load selection weights with fallback
            self.selection_weights = {
                'golden_similarity': weights_config.get('golden_similarity', default_weights['golden_similarity']),
                'relative_rank': weights_config.get('relative_rank', default_weights['relative_rank']),
                'cost_efficiency': weights_config.get('cost_efficiency', default_weights['cost_efficiency']),
                'time_efficiency': weights_config.get('time_efficiency', default_weights['time_efficiency']),
                'uniqueness': weights_config.get('uniqueness', default_weights['uniqueness'])
            }
            
            # Load performance configuration
            self.performance_config = {
                'ignore_performance_time': weights_config.get('ignore_performance_time', default_performance['ignore_performance_time']),
                'focus_on_accuracy_only': weights_config.get('focus_on_accuracy_only', default_performance['focus_on_accuracy_only']),
                'use_cost_optimization': weights_config.get('use_cost_optimization', default_performance['use_cost_optimization']),
                'max_acceptable_response_time': weights_config.get('max_acceptable_response_time', default_performance['max_acceptable_response_time']),
                'time_penalty_threshold': weights_config.get('time_penalty_threshold', default_performance['time_penalty_threshold']),
                'time_penalty_factor': weights_config.get('time_penalty_factor', default_performance['time_penalty_factor'])
            }
            print(f"Selection weights loaded from configuration: {self.selection_weights}")
        else:
            # Use defaults if config not available
            self.selection_weights = default_weights
            self.performance_config = default_performance
            print(f"Using default selection weights: {self.selection_weights}")

    def predict_llm_quality(self, prompt: str, llm_id: str) -> float:
        """
        Predict quality score for a given LLM on a specific prompt
        using vector similarity to historical successful responses
        """
        if not self.vector_db_evaluator.human_eval_records:
            return 0.5  # Default neutral score
        
        # Check cache first
        cache_key = f"{hash(prompt)}_{llm_id}"
        if cache_key in self.quality_prediction_cache:
            return self.quality_prediction_cache[cache_key]
        
        # Find similar prompts in evaluation set
        similar_examples = self.vector_db_evaluator.find_similar_examples(prompt, top_k=5)
        
        if not similar_examples:
            predicted_quality = 0.5
        else:
            # Get historical performance for this LLM on similar prompts
            llm_scores = []
            for record, similarity in similar_examples:
                # Check if we have performance history for this LLM on similar categories
                category = record.category
                if category in self.llm_category_performance[llm_id]:
                    category_scores = list(self.llm_category_performance[llm_id][category])
                    if category_scores:
                        # Weight by similarity
                        weighted_score = np.mean(category_scores) * similarity
                        llm_scores.append(weighted_score)
            
            if llm_scores:
                predicted_quality = np.mean(llm_scores)
            else:
                # Use overall LLM quality if no category-specific data
                if self.quality_scores[llm_id]:
                    predicted_quality = np.mean(list(self.quality_scores[llm_id]))
                else:
                    predicted_quality = 0.5
        
        # Cache the result
        if len(self.quality_prediction_cache) >= self.cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.quality_prediction_cache))
            del self.quality_prediction_cache[oldest_key]
        
        self.quality_prediction_cache[cache_key] = predicted_quality
        return predicted_quality
    
    def select_llm_with_vector_context(self, prompt: str) -> str:
        """
        Enhanced LLM selection using vector similarity and predictive quality scoring
        """
        if not self.llm_ids:
            raise ValueError("No LLMs configured")
        
        # Get quality predictions for all LLMs
        llm_scores = {}
        for llm_id in self.llm_ids:
            # Combine predicted quality, current allocation, and multi-objective optimization
            predicted_quality = self.predict_llm_quality(prompt, llm_id)
            
            # Get speed factor (inverse of response time)
            avg_response_time = np.mean(list(self.response_times[llm_id])) if self.response_times[llm_id] else self.llm_configs[llm_id]['base_response_time']
            speed_factor = 1.0 / max(avg_response_time, 0.1)  # Avoid division by zero
            
            # Get cost factor (assuming lower is better, normalize by base factor)
            cost_factor = 1.0 / self.llm_configs[llm_id].get('cost_factor', 1.0)
            
            # Multi-objective score
            composite_score = (
                predicted_quality * self.optimization_weights['quality'] +
                speed_factor * self.optimization_weights['speed'] +
                cost_factor * self.optimization_weights['cost']
            )
            
            # Apply current allocation as a selection probability modifier
            allocation_weight = self.current_allocation.get(llm_id, 0.1)
            final_score = composite_score * (1 + allocation_weight)
            
            llm_scores[llm_id] = final_score
        
        # Select LLM with highest score, but add some randomness for exploration
        scores_array = np.array(list(llm_scores.values()))
        if np.sum(scores_array) > 0:
            # Softmax selection with temperature for exploration
            temperature = 0.5
            probabilities = np.exp(scores_array / temperature) / np.sum(np.exp(scores_array / temperature))
            selected_idx = np.random.choice(len(self.llm_ids), p=probabilities)
            return self.llm_ids[selected_idx]
        else:
            # Fallback to base router selection
            return super().select_llm(prompt)
    
    def route_request_with_evaluation(self, prompt: str) -> Tuple[LLMResponse, float]:
        """Enhanced routing with predictive selection and comprehensive evaluation"""
        # Use vector-context aware selection if we have evaluation data
        if self.vector_db_evaluator.human_eval_records:
            selected_llm = self.select_llm_with_vector_context(prompt)
        else:
            # Fallback to base router selection
            selected_llm = super().select_llm(prompt)
        
        # Get response using selected LLM (use real or mock based on configuration)
        if self.use_real_providers:
            response = self.real_provider_call(selected_llm, prompt)
        else:
            response = self.mock_llm_call(selected_llm, prompt)
        
        # Evaluate quality using vector database
        quality_score = self.vector_db_evaluator.evaluate_response_quality(
            prompt, response.content
        )
        
        # Enhanced performance tracking
        self._update_enhanced_performance_tracking(prompt, response, quality_score)
        
        # Update PID controllers with vector-informed quality
        self.quality_scores[response.llm_id].append(quality_score)
        self.performance_history[response.llm_id].append(
            1.0 / max(response.generation_time, 0.1)  # Performance as inverse of time
        )
        self.response_times[response.llm_id].append(response.generation_time)
        
        # Trigger allocation update with enhanced metrics
        self.update_allocation_with_pid()
        
        return response, quality_score
    
    def generate_all_responses(self, prompt: str) -> Dict[str, LLMResponse]:
        """
        Generate responses from all available LLMs in parallel for comparison
        
        Args:
            prompt: Input prompt for all LLMs
            
        Returns:
            Dict mapping LLM names to their responses
        """
        all_responses = {}
        
        # Generate responses from all available LLMs
        if self.use_real_providers:
            # Use real providers
            for provider_name in self.providers.keys():
                try:
                    response = self.real_provider_call(provider_name, prompt)
                    all_responses[provider_name] = response
                except Exception as e:
                    print(f"Warning: Failed to get response from {provider_name}: {e}")
                    # Continue with other providers
        else:
            # Use mock LLMs
            for llm_id in self.llm_ids:
                try:
                    response = self.mock_llm_call(llm_id, prompt)
                    all_responses[llm_id] = response
                except Exception as e:
                    print(f"Warning: Failed to get mock response from {llm_id}: {e}")
                    # Continue with other LLMs
        
        return all_responses
    
    def evaluate_all_responses(self, prompt: str, responses: Dict[str, LLMResponse]) -> Dict[str, Dict[str, float]]:
        """
        Comprehensive evaluation of all responses against golden output and each other
        
        Args:
            prompt: Original prompt
            responses: Dict of LLM responses
            
        Returns:
            Dict mapping LLM names to evaluation scores
        """
        evaluation_results = {}
        
        for llm_name, response in responses.items():
            scores = {}
            
            # 1. Golden output similarity (if available)
            golden_similarity = self.vector_db_evaluator.evaluate_response_quality(response)
            scores['golden_similarity'] = golden_similarity
            
            # 2. Response quality metrics
            scores['response_length'] = len(response.content.split())
            scores['generation_time'] = response.generation_time
            
            # Apply configurable time efficiency calculation
            if self.performance_config['ignore_performance_time']:
                scores['time_efficiency'] = 1.0  # Neutral score when ignoring performance
            else:
                base_time_efficiency = 1.0 / max(response.generation_time, 0.1)
                
                # Apply time penalty if response time exceeds threshold
                if response.generation_time > self.performance_config['time_penalty_threshold']:
                    penalty = (response.generation_time - self.performance_config['time_penalty_threshold']) * self.performance_config['time_penalty_factor']
                    time_efficiency = base_time_efficiency * (1.0 - min(penalty, 0.9))  # Cap penalty at 90%
                else:
                    time_efficiency = base_time_efficiency
                
                # Apply maximum acceptable response time check
                if response.generation_time > self.performance_config['max_acceptable_response_time']:
                    time_efficiency *= 0.1  # Heavy penalty for exceeding max time
                
                scores['time_efficiency'] = max(time_efficiency, 0.01)  # Minimum score
            
            # 3. Cost efficiency (if available)
            if hasattr(response, 'cost') and response.cost:
                scores['cost'] = response.cost
                scores['cost_efficiency'] = golden_similarity / max(response.cost, 0.001)
            else:
                scores['cost'] = 0.0
                scores['cost_efficiency'] = golden_similarity
            
            evaluation_results[llm_name] = scores
        
        # 4. Relative ranking - compare responses against each other
        evaluation_results = self._add_relative_rankings(responses, evaluation_results)
        
        return evaluation_results
    
    def _add_relative_rankings(self, responses: Dict[str, LLMResponse], 
                              evaluation_results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Add relative rankings by comparing responses against each other
        
        Args:
            responses: Dict of LLM responses
            evaluation_results: Current evaluation scores
            
        Returns:
            Updated evaluation results with relative rankings
        """
        if len(responses) < 2:
            # Can't do relative ranking with less than 2 responses
            for llm_name in evaluation_results:
                evaluation_results[llm_name]['relative_rank'] = 1.0
            return evaluation_results
        
        # Calculate pairwise similarities between all responses
        response_texts = {name: resp.content for name, resp in responses.items()}
        llm_names = list(response_texts.keys())
        
        # Create similarity matrix
        similarity_matrix = {}
        for i, llm1 in enumerate(llm_names):
            similarity_matrix[llm1] = {}
            for j, llm2 in enumerate(llm_names):
                if i == j:
                    similarity_matrix[llm1][llm2] = 1.0
                else:
                    # Calculate token overlap similarity
                    sim = self._calculate_response_similarity(
                        response_texts[llm1], 
                        response_texts[llm2]
                    )
                    similarity_matrix[llm1][llm2] = sim
        
        # Calculate relative rankings based on golden similarity and uniqueness
        for llm_name in evaluation_results:
            golden_score = evaluation_results[llm_name]['golden_similarity']
            
            # Uniqueness score (lower similarity with others = more unique)
            avg_similarity_to_others = np.mean([
                similarity_matrix[llm_name][other] 
                for other in llm_names if other != llm_name
            ])
            uniqueness_score = 1.0 - avg_similarity_to_others
            
            # Combined relative ranking
            relative_rank = (golden_score * 0.7) + (uniqueness_score * 0.3)
            evaluation_results[llm_name]['relative_rank'] = relative_rank
            evaluation_results[llm_name]['uniqueness'] = uniqueness_score
            evaluation_results[llm_name]['avg_similarity_to_others'] = avg_similarity_to_others
        
        return evaluation_results
    
    def _calculate_response_similarity(self, response1: str, response2: str) -> float:
        """
        Calculate similarity between two responses using token overlap
        
        Args:
            response1: First response text
            response2: Second response text
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Simple token-based similarity (can be enhanced with semantic similarity later)
        tokens1 = set(response1.lower().split())
        tokens2 = set(response2.lower().split())
        
        if not tokens1 and not tokens2:
            return 1.0
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0
    
    def select_best_response(self, evaluations: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, Any]]:
        """
        Select the best LLM response based on comprehensive evaluation criteria
        
        Args:
            evaluations: Dict mapping LLM names to their evaluation scores
            
        Returns:
            Tuple of (best_llm_name, detailed_selection_info)
        """
        if not evaluations:
            raise ValueError("No evaluations provided for selection")
        
        if len(evaluations) == 1:
            llm_name = list(evaluations.keys())[0]
            selection_info = {
                'selection_method': 'single_option',
                'reason': 'Only one LLM available',
                'weights_used': {},
                'composite_scores': {llm_name: 1.0},
                'score_breakdown': evaluations[llm_name],
                'ranking': [llm_name]
            }
            return llm_name, selection_info
        
        # Use configured weights for different criteria
        weights = self.selection_weights
        
        # Calculate composite scores and track reasoning
        composite_scores = {}
        score_breakdowns = {}
        
        for llm_name, scores in evaluations.items():
            composite_score = 0.0
            breakdown = {}
            
            for criterion, weight in weights.items():
                if criterion in scores:
                    # Normalize score to 0-1 range if needed
                    score_value = scores[criterion]
                    if criterion == 'cost' and score_value > 0:
                        # For cost, lower is better, so invert
                        score_value = 1.0 / (1.0 + score_value)
                    
                    normalized_score = min(max(score_value, 0.0), 1.0)
                    weighted_contribution = weight * normalized_score
                    composite_score += weighted_contribution
                    
                    breakdown[criterion] = {
                        'raw_score': score_value,
                        'normalized_score': normalized_score,
                        'weight': weight,
                        'contribution': weighted_contribution
                    }
            
            composite_scores[llm_name] = composite_score
            score_breakdowns[llm_name] = breakdown
        
        # Select LLM with highest composite score
        best_llm = max(composite_scores, key=composite_scores.get)
        
        # Create ranking from best to worst
        ranking = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        ranking_list = [llm for llm, _ in ranking]
        
        # Generate detailed selection reasoning
        selection_reason_parts = [
            f"Selected {best_llm} with composite score {composite_scores[best_llm]:.3f}",
            f"Primary factors: Golden similarity ({weights['golden_similarity']*100:.0f}%), Ranking ({weights['relative_rank']*100:.0f}%)",
            f"Beat {len(composite_scores)-1} other LLMs by {composite_scores[best_llm] - min(composite_scores.values()):.3f} points"
        ]
        
        # Add specific advantages
        best_breakdown = score_breakdowns[best_llm]
        advantages = []
        for criterion, details in best_breakdown.items():
            if details['contribution'] > 0.1:  # Significant contribution
                advantages.append(f"{criterion}: {details['contribution']:.3f}")
        
        if advantages:
            selection_reason_parts.append(f"Key strengths: {', '.join(advantages)}")
        
        detailed_selection_info = {
            'selection_method': 'weighted_composite_scoring',
            'reason': ' | '.join(selection_reason_parts),
            'weights_used': weights,
            'composite_scores': composite_scores,
            'score_breakdowns': score_breakdowns,
            'ranking': ranking_list,
            'selection_criteria': {
                'primary_metric': 'composite_score',
                'threshold_used': self.vector_db_evaluator.similarity_threshold,
                'weighting_strategy': 'accuracy_first_with_efficiency_factors'
            }
        }
        
        # Log selection reasoning
        print(f"Selected {best_llm} with composite score {composite_scores[best_llm]:.3f}")
        print("Score breakdown:")
        for llm_name, score in ranking:
            print(f"  {llm_name}: {score:.3f}")
        
        return best_llm, detailed_selection_info
    
    def route_request_with_multi_llm_evaluation(self, prompt: str) -> Tuple[LLMResponse, float, Dict[str, Any]]:
        """
        Full multi-LLM comparison mode - generate responses from all LLMs and select the best
        
        Args:
            prompt: Input prompt
            
        Returns:
            Tuple of (best_response, quality_score, detailed_evaluation)
        """
        print(f"Multi-LLM evaluation mode for prompt: {prompt[:100]}...")
        
        # Step 1: Generate responses from all available LLMs
        all_responses = self.generate_all_responses(prompt)
        
        if not all_responses:
            raise RuntimeError("Failed to generate responses from any LLM")
        
        print(f"Generated {len(all_responses)} responses from: {list(all_responses.keys())}")
        
        # Step 2: Evaluate all responses comprehensively
        all_evaluations = self.evaluate_all_responses(prompt, all_responses)
        
        # Step 3: Select the best response with detailed selection information
        best_llm_name, selection_info = self.select_best_response(all_evaluations)
        best_response = all_responses[best_llm_name]
        best_quality_score = all_evaluations[best_llm_name]['golden_similarity']
        
        # Add composite scores back to evaluations for API response
        for llm_name, composite_score in selection_info['composite_scores'].items():
            all_evaluations[llm_name]['composite_score'] = composite_score
            # Add ranking position
            all_evaluations[llm_name]['rank'] = selection_info['ranking'].index(llm_name) + 1
        
        # Step 4: Update performance tracking for all LLMs
        for llm_name, response in all_responses.items():
            evaluation = all_evaluations[llm_name]
            self._update_multi_llm_performance_tracking(llm_name, response, evaluation)
        
        # Step 5: Prepare detailed evaluation summary
        detailed_evaluation = {
            'mode': 'multi_llm',
            'all_responses': {name: resp.content[:200] + "..." if len(resp.content) > 200 else resp.content 
                             for name, resp in all_responses.items()},
            'all_evaluations': all_evaluations,
            'selection_reason': selection_info['reason'],
            'selection_criteria': selection_info['selection_criteria'],
            'comparative_analysis': self._generate_comparative_analysis(all_evaluations),
            'detailed_scores': selection_info['score_breakdowns'],
            'llm_ranking': selection_info['ranking']
        }
        
        print(f"Selected best response from {best_llm_name}")
        
        return best_response, best_quality_score, detailed_evaluation
    
    def _update_multi_llm_performance_tracking(self, llm_name: str, response: LLMResponse, 
                                             evaluation: Dict[str, float]):
        """
        Update performance tracking for multi-LLM comparisons
        
        Args:
            llm_name: Name of the LLM
            response: LLM response
            evaluation: Evaluation scores
        """
        # Use the actual LLM name for tracking
        llm_id = llm_name if self.use_real_providers else response.llm_id
        
        # Update quality tracking
        quality_score = evaluation['golden_similarity']
        self.quality_scores[llm_id].append(quality_score)
        
        # Update performance metrics
        self.performance_history[llm_id].append(evaluation['time_efficiency'])
        self.response_times[llm_id].append(response.generation_time)
        
        # Enhanced tracking for multi-LLM mode
        if not hasattr(self, 'multi_llm_stats'):
            self.multi_llm_stats = defaultdict(lambda: defaultdict(list))
        
        # Track detailed statistics
        self.multi_llm_stats[llm_id]['golden_similarity'].append(quality_score)
        self.multi_llm_stats[llm_id]['relative_rank'].append(evaluation.get('relative_rank', 0.5))
        self.multi_llm_stats[llm_id]['uniqueness'].append(evaluation.get('uniqueness', 0.5))
        self.multi_llm_stats[llm_id]['cost_efficiency'].append(evaluation.get('cost_efficiency', 0.5))
    
    def _generate_comparative_analysis(self, evaluations: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Generate comprehensive comparative analysis of all LLM performances with detailed reasoning
        
        Args:
            evaluations: All LLM evaluations including composite scores and rankings
            
        Returns:
            Detailed comparative analysis with selection reasoning
        """
        analysis = {
            'best_in_category': {},
            'performance_spread': {},
            'ranking_explanations': {},
            'selection_factors': {},
            'recommendations': [],
            'quality_assessment': {}
        }
        
        # Find best performer in each category
        categories = ['golden_similarity', 'time_efficiency', 'cost_efficiency', 'uniqueness', 'relative_rank', 'composite_score']
        
        for category in categories:
            if any(category in eval_scores for eval_scores in evaluations.values()):
                category_scores = {llm: scores.get(category, 0) for llm, scores in evaluations.items()}
                best_llm = max(category_scores, key=category_scores.get)
                worst_llm = min(category_scores, key=category_scores.get)
                
                analysis['best_in_category'][category] = {
                    'best_llm': best_llm,
                    'best_score': category_scores[best_llm],
                    'worst_llm': worst_llm,
                    'worst_score': category_scores[worst_llm],
                    'advantage': category_scores[best_llm] - category_scores[worst_llm]
                }
        
        # Calculate performance spread for each category
        for category in categories:
            if any(category in eval_scores for eval_scores in evaluations.values()):
                scores = [scores.get(category, 0) for scores in evaluations.values()]
                analysis['performance_spread'][category] = {
                    'min': min(scores),
                    'max': max(scores),
                    'std': float(np.std(scores)),
                    'range': max(scores) - min(scores),
                    'mean': float(np.mean(scores))
                }
        
        # Generate detailed ranking explanations for each LLM
        for llm_name, scores in evaluations.items():
            ranking_position = scores.get('rank', 'N/A')
            composite_score = scores.get('composite_score', scores.get('golden_similarity', 0))
            
            # Analyze strengths and weaknesses
            strengths = []
            weaknesses = []
            
            for category in ['golden_similarity', 'time_efficiency', 'cost_efficiency', 'uniqueness', 'relative_rank']:
                if category in scores:
                    score = scores[category]
                    category_best = analysis['best_in_category'].get(category, {}).get('best_score', 0)
                    category_worst = analysis['best_in_category'].get(category, {}).get('worst_score', 0)
                    
                    # Determine if this is a strength or weakness
                    if category_best > 0 and score >= category_best * 0.9:  # Within 90% of best
                        strengths.append(f"{category}: {score:.3f} (top performer)")
                    elif category_worst < 1 and score <= category_worst * 1.1:  # Within 110% of worst
                        weaknesses.append(f"{category}: {score:.3f} (needs improvement)")
            
            # Generate explanation for this LLM's ranking
            explanation_parts = [
                f"Ranked #{ranking_position} with composite score {composite_score:.3f}"
            ]
            
            if strengths:
                explanation_parts.append(f"Strengths: {', '.join(strengths)}")
            if weaknesses:
                explanation_parts.append(f"Areas for improvement: {', '.join(weaknesses)}")
            
            analysis['ranking_explanations'][llm_name] = {
                'rank': ranking_position,
                'composite_score': composite_score,
                'explanation': ' | '.join(explanation_parts),
                'strengths': strengths,
                'weaknesses': weaknesses
            }
        
        # Identify key selection factors that drove the decision
        analysis['selection_factors'] = {
            'primary_differentiator': self._identify_primary_differentiator(evaluations),
            'quality_variance': analysis['performance_spread'].get('golden_similarity', {}).get('std', 0),
            'efficiency_variance': analysis['performance_spread'].get('time_efficiency', {}).get('std', 0),
            'cost_variance': analysis['performance_spread'].get('cost_efficiency', {}).get('std', 0)
        }
        
        # Generate actionable recommendations
        golden_scores = {llm: scores.get('golden_similarity', 0) for llm, scores in evaluations.items()}
        best_accuracy = max(golden_scores, key=golden_scores.get)
        worst_accuracy = min(golden_scores, key=golden_scores.get)
        
        time_scores = {llm: scores.get('time_efficiency', 0) for llm, scores in evaluations.items()}
        fastest = max(time_scores, key=time_scores.get) if time_scores else None
        
        cost_scores = {llm: scores.get('cost_efficiency', 0) for llm, scores in evaluations.items()}
        most_efficient = max(cost_scores, key=cost_scores.get) if cost_scores else None
        
        analysis['recommendations'] = [
            f"Best overall accuracy: {best_accuracy} ({golden_scores[best_accuracy]:.3f})",
        ]
        
        if fastest and fastest != best_accuracy:
            analysis['recommendations'].append(f"Fastest response: {fastest} ({time_scores[fastest]:.3f})")
        
        if most_efficient and most_efficient not in [best_accuracy, fastest]:
            analysis['recommendations'].append(f"Most cost-efficient: {most_efficient} ({cost_scores[most_efficient]:.3f})")
        
        if golden_scores[best_accuracy] - golden_scores[worst_accuracy] > 0.2:
            analysis['recommendations'].append("Significant quality differences detected - consider LLM specialization")
        
        if analysis['performance_spread'].get('golden_similarity', {}).get('std', 0) < 0.1:
            analysis['recommendations'].append("LLMs performed similarly - consider cost/speed optimization")
        
        # Overall quality assessment
        mean_quality = analysis['performance_spread'].get('golden_similarity', {}).get('mean', 0)
        analysis['quality_assessment'] = {
            'overall_quality': 'high' if mean_quality > 0.7 else 'moderate' if mean_quality > 0.5 else 'low',
            'consistency': 'high' if analysis['performance_spread'].get('golden_similarity', {}).get('std', 1) < 0.15 else 'moderate' if analysis['performance_spread'].get('golden_similarity', {}).get('std', 1) < 0.3 else 'low',
            'mean_similarity_score': mean_quality
        }
        
        return analysis
    
    def _identify_primary_differentiator(self, evaluations: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Identify which factor was most important in the selection decision
        
        Args:
            evaluations: All LLM evaluations
            
        Returns:
            Information about the primary differentiating factor
        """
        # Calculate variance for each metric to see which had the biggest spread
        variances = {}
        
        for category in ['golden_similarity', 'time_efficiency', 'cost_efficiency', 'uniqueness', 'relative_rank']:
            if any(category in eval_scores for eval_scores in evaluations.values()):
                scores = [scores.get(category, 0) for scores in evaluations.values()]
                variances[category] = float(np.var(scores))
        
        if not variances:
            return {'factor': 'unknown', 'variance': 0, 'explanation': 'Insufficient data'}
        
        # The metric with highest variance was likely the deciding factor
        primary_factor = max(variances, key=variances.get)
        
        return {
            'factor': primary_factor,
            'variance': variances[primary_factor],
            'explanation': f"{primary_factor} showed the highest variance ({variances[primary_factor]:.4f}), making it the key differentiator",
            'all_variances': variances
        }
    
    def get_multi_llm_performance_insights(self) -> Dict[str, Any]:
        """
        Get insights from multi-LLM performance tracking
        
        Returns:
            Performance insights and recommendations
        """
        if not hasattr(self, 'multi_llm_stats') or not self.multi_llm_stats:
            return {"message": "No multi-LLM performance data available"}
        
        insights = {
            'llm_performance': {},
            'trends': {},
            'recommendations': []
        }
        
        # Analyze each LLM's performance
        for llm_id, stats in self.multi_llm_stats.items():
            llm_insights = {}
            
            # Calculate averages for each metric
            for metric, values in stats.items():
                if values:
                    llm_insights[f'avg_{metric}'] = np.mean(values)
                    llm_insights[f'std_{metric}'] = np.std(values)
                    llm_insights[f'trend_{metric}'] = self._calculate_trend(values)
                    llm_insights[f'samples_{metric}'] = len(values)
            
            insights['llm_performance'][llm_id] = llm_insights
        
        # Generate comparative insights
        insights['comparative'] = self._generate_comparative_insights()
        
        # Generate recommendations based on performance data
        insights['recommendations'] = self._generate_performance_recommendations()
        
        return insights
    
    def _calculate_trend(self, values: List[float], window_size: int = 10) -> str:
        """
        Calculate trend direction for a series of values
        
        Args:
            values: List of metric values
            window_size: Window size for trend calculation
            
        Returns:
            Trend direction: 'improving', 'declining', 'stable'
        """
        if len(values) < window_size:
            return 'insufficient_data'
        
        # Compare recent average vs earlier average
        recent_avg = np.mean(values[-window_size:])
        earlier_avg = np.mean(values[:window_size])
        
        diff = recent_avg - earlier_avg
        threshold = 0.05  # 5% threshold for trend detection
        
        if diff > threshold:
            return 'improving'
        elif diff < -threshold:
            return 'declining'
        else:
            return 'stable'
    
    def _generate_comparative_insights(self) -> Dict[str, Any]:
        """
        Generate comparative insights across all LLMs
        
        Returns:
            Comparative analysis of LLM performance
        """
        if not hasattr(self, 'multi_llm_stats'):
            return {}
        
        comparative = {
            'best_performers': {},
            'consistency_rankings': {},
            'specialization_analysis': {}
        }
        
        # Find best average performers in each category
        metrics = ['golden_similarity', 'relative_rank', 'uniqueness', 'cost_efficiency']
        
        for metric in metrics:
            llm_averages = {}
            for llm_id, stats in self.multi_llm_stats.items():
                if metric in stats and stats[metric]:
                    llm_averages[llm_id] = np.mean(stats[metric])
            
            if llm_averages:
                best_llm = max(llm_averages, key=llm_averages.get)
                comparative['best_performers'][metric] = {
                    'llm': best_llm,
                    'score': llm_averages[best_llm],
                    'all_scores': llm_averages
                }
        
        # Analyze consistency (lower std deviation = more consistent)
        for metric in metrics:
            llm_consistency = {}
            for llm_id, stats in self.multi_llm_stats.items():
                if metric in stats and stats[metric] and len(stats[metric]) > 1:
                    llm_consistency[llm_id] = np.std(stats[metric])
            
            if llm_consistency:
                most_consistent = min(llm_consistency, key=llm_consistency.get)
                comparative['consistency_rankings'][metric] = {
                    'most_consistent': most_consistent,
                    'consistency_scores': llm_consistency
                }
        
        return comparative
    
    def _generate_performance_recommendations(self) -> List[str]:
        """
        Generate actionable recommendations based on performance data
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if not hasattr(self, 'multi_llm_stats'):
            return ["Insufficient performance data for recommendations"]
        
        # Check if we have enough data
        total_samples = sum(
            len(stats.get('golden_similarity', []))
            for stats in self.multi_llm_stats.values()
        )
        
        if total_samples < 10:
            recommendations.append("Need more evaluation samples for reliable recommendations")
            return recommendations
        
        # Analyze performance patterns
        comparative = self._generate_comparative_insights()
        
        # Quality recommendations
        if 'golden_similarity' in comparative['best_performers']:
            best_quality = comparative['best_performers']['golden_similarity']
            recommendations.append(
                f"For highest quality: prefer {best_quality['llm']} "
                f"(avg score: {best_quality['score']:.3f})"
            )
        
        # Speed vs Quality trade-off analysis
        quality_scores = {}
        time_efficiency = {}
        
        for llm_id, stats in self.multi_llm_stats.items():
            if 'golden_similarity' in stats and stats['golden_similarity']:
                quality_scores[llm_id] = np.mean(stats['golden_similarity'])
            
            # Get time efficiency from base router
            if llm_id in self.performance_history and self.performance_history[llm_id]:
                time_efficiency[llm_id] = np.mean(self.performance_history[llm_id])
        
        # Find optimal trade-off
        if quality_scores and time_efficiency:
            trade_off_scores = {}
            for llm_id in quality_scores:
                if llm_id in time_efficiency:
                    # Combined score: 70% quality, 30% speed
                    trade_off_scores[llm_id] = (quality_scores[llm_id] * 0.7 + 
                                               time_efficiency[llm_id] * 0.3)
            
            if trade_off_scores:
                best_tradeoff = max(trade_off_scores, key=trade_off_scores.get)
                recommendations.append(
                    f"Best quality/speed trade-off: {best_tradeoff} "
                    f"(combined score: {trade_off_scores[best_tradeoff]:.3f})"
                )
        
        # Specialization recommendations
        uniqueness_leaders = comparative['best_performers'].get('uniqueness', {})
        if uniqueness_leaders:
            recommendations.append(
                f"For creative/unique responses: prefer {uniqueness_leaders['llm']} "
                f"(uniqueness: {uniqueness_leaders['score']:.3f})"
            )
        
        # Consistency recommendations
        if 'golden_similarity' in comparative['consistency_rankings']:
            consistent_leader = comparative['consistency_rankings']['golden_similarity']
            recommendations.append(
                f"Most consistent quality: {consistent_leader['most_consistent']} "
                f"(std dev: {consistent_leader['consistency_scores'][consistent_leader['most_consistent']]:.3f})"
            )
        
        return recommendations
    
    def adapt_allocation_from_multi_llm_performance(self):
        """
        Adapt LLM allocation weights based on multi-LLM comparison performance
        """
        if not hasattr(self, 'multi_llm_stats') or not self.multi_llm_stats:
            print("No multi-LLM performance data available for adaptation")
            return
        
        # Calculate performance-based weights
        performance_weights = {}
        
        for llm_id, stats in self.multi_llm_stats.items():
            if 'golden_similarity' in stats and stats['golden_similarity']:
                # Base weight on average golden similarity performance
                avg_quality = np.mean(stats['golden_similarity'])
                
                # Bonus for consistency (lower std deviation)
                consistency_bonus = 1.0 - min(np.std(stats['golden_similarity']), 0.3)
                
                # Bonus for uniqueness (diversity)
                uniqueness_bonus = 1.0
                if 'uniqueness' in stats and stats['uniqueness']:
                    uniqueness_bonus = 1.0 + (np.mean(stats['uniqueness']) * 0.2)
                
                # Calculate combined performance weight
                performance_weights[llm_id] = avg_quality * consistency_bonus * uniqueness_bonus
        
        if not performance_weights:
            print("No performance data available for weight adaptation")
            return
        
        # Normalize weights to sum to 1.0
        total_weight = sum(performance_weights.values())
        if total_weight > 0:
            for llm_id in performance_weights:
                performance_weights[llm_id] /= total_weight
        
        # Update router allocations with performance-based weights
        print("Adapting LLM allocations based on multi-LLM performance:")
        for llm_id, weight in performance_weights.items():
            if llm_id in self.current_allocation:
                old_weight = self.current_allocation[llm_id]
                # Gradual adaptation: 70% old weight + 30% performance weight
                new_weight = old_weight * 0.7 + weight * 0.3
                self.current_allocation[llm_id] = new_weight
                
                print(f"  {llm_id}: {old_weight:.3f} -> {new_weight:.3f} "
                      f"(performance score: {performance_weights[llm_id]:.3f})")
        
        # Renormalize allocations
        total_allocation = sum(self.current_allocation.values())
        if total_allocation > 0:
            for llm_id in self.current_allocation:
                self.current_allocation[llm_id] /= total_allocation
    
    def route_request_adaptive(self, prompt: str, mode: str = "auto") -> Tuple[LLMResponse, float, Dict[str, Any]]:
        """
        Adaptive routing that can switch between single-LLM and multi-LLM modes
        
        Args:
            prompt: Input prompt
            mode: Routing mode - "auto", "fast", "multi_llm", "comprehensive", "quality"
                - "auto": Intelligent routing based on prompt complexity
                - "fast": Single LLM pre-selection for speed
                - "multi_llm": Force multi-LLM evaluation for analysis
                - "comprehensive": Multi-LLM comparison for quality
                - "quality": Multi-LLM comparison prioritizing accuracy
            
        Returns:
            Tuple of (response, quality_score, evaluation_details)
        """
        # Determine routing strategy based on mode and context
        if mode == "auto":
            routing_mode = self._determine_optimal_routing_mode(prompt)
        else:
            routing_mode = mode
        
        print(f"Using routing mode: {routing_mode}")
        
        if routing_mode == "fast":
            # Single LLM selection for speed
            response, quality_score = self.route_request_with_evaluation(prompt)
            evaluation_details = {
                'mode': 'single_llm',
                'selection_reason': 'Fast mode - single LLM pre-selection based on PID allocation',
                'selection_criteria': {
                    'primary_metric': 'pid_allocation',
                    'threshold_used': self.vector_db_evaluator.similarity_threshold,
                    'weighting_strategy': 'performance_based_allocation'
                },
                'all_evaluations': {response.llm_id: {'golden_similarity': quality_score}},
                'comparative_analysis': None
            }
            return response, quality_score, evaluation_details
            
        elif routing_mode in ["comprehensive", "quality", "multi_llm"]:
            # Multi-LLM comparison for best quality
            return self.route_request_with_multi_llm_evaluation(prompt)
            
        else:
            # Default to single LLM
            response, quality_score = self.route_request_with_evaluation(prompt)
            evaluation_details = {
                'mode': 'single_llm', 
                'selection_reason': 'Default mode - single LLM pre-selection based on historical performance',
                'selection_criteria': {
                    'primary_metric': 'historical_performance',
                    'threshold_used': self.vector_db_evaluator.similarity_threshold,
                    'weighting_strategy': 'allocation_based_selection'
                },
                'all_evaluations': {response.llm_id: {'golden_similarity': quality_score}},
                'comparative_analysis': None
            }
            return response, quality_score, evaluation_details
    
    def _determine_optimal_routing_mode(self, prompt: str) -> str:
        """
        Intelligently determine the best routing mode based on prompt and context
        
        Args:
            prompt: Input prompt to analyze
            
        Returns:
            Recommended routing mode: "fast" or "comprehensive"
        """
        # Analyze prompt characteristics
        prompt_length = len(prompt.split())
        
        # Check if prompt seems complex or critical
        complexity_indicators = [
            'analyze', 'compare', 'evaluate', 'critical', 'important', 'decision',
            'recommendation', 'strategy', 'detailed', 'comprehensive', 'explain',
            'summarize', 'review'
        ]
        
        has_complexity_indicators = any(
            indicator in prompt.lower() 
            for indicator in complexity_indicators
        )
        
        # Check recent performance variance to see if LLMs differ significantly
        performance_variance = self._calculate_recent_performance_variance()
        
        # Decision logic
        if prompt_length > 50:  # Long prompts benefit from multi-LLM comparison
            return "comprehensive"
        elif has_complexity_indicators:  # Complex tasks benefit from comparison
            return "comprehensive"
        elif performance_variance > 0.15:  # High variance means LLMs differ significantly
            return "comprehensive"
        else:
            return "fast"  # Simple prompts can use fast mode
    
    def _calculate_recent_performance_variance(self) -> float:
        """
        Calculate variance in recent LLM performance to inform routing decisions
        
        Returns:
            Performance variance (0.0 = all LLMs perform equally, higher = more variance)
        """
        if not hasattr(self, 'multi_llm_stats'):
            return 0.1  # Default moderate variance
        
        recent_performances = []
        
        # Get recent performance scores from each LLM
        for llm_id, stats in self.multi_llm_stats.items():
            if 'golden_similarity' in stats and stats['golden_similarity']:
                # Use last 5 scores or all if less than 5
                recent_scores = stats['golden_similarity'][-5:]
                if recent_scores:
                    recent_performances.append(np.mean(recent_scores))
        
        if len(recent_performances) < 2:
            return 0.1  # Default if insufficient data
        
        return np.std(recent_performances)
    
    def get_routing_mode_recommendations(self, prompt: str) -> Dict[str, Any]:
        """
        Get recommendations for routing mode based on prompt analysis
        
        Args:
            prompt: Input prompt
            
        Returns:
            Routing recommendations with reasoning
        """
        recommendations = {
            'recommended_mode': self._determine_optimal_routing_mode(prompt),
            'analysis': {},
            'alternatives': {}
        }
        
        # Analyze prompt characteristics
        prompt_length = len(prompt.split())
        complexity_score = self._calculate_prompt_complexity(prompt)
        performance_variance = self._calculate_recent_performance_variance()
        
        recommendations['analysis'] = {
            'prompt_length': prompt_length,
            'complexity_score': complexity_score,
            'performance_variance': performance_variance,
            'reasoning': self._generate_routing_reasoning(
                prompt_length, complexity_score, performance_variance
            )
        }
        
        # Provide alternatives with trade-offs
        if recommendations['recommended_mode'] == 'fast':
            recommendations['alternatives'] = {
                'comprehensive': {
                    'pros': ['Higher quality', 'Better for complex tasks', 'Comprehensive comparison'],
                    'cons': ['Slower', 'Higher cost', 'May be overkill for simple tasks'],
                    'use_when': 'Quality is more important than speed'
                }
            }
        else:
            recommendations['alternatives'] = {
                'fast': {
                    'pros': ['Much faster', 'Lower cost', 'Good for simple tasks'],
                    'cons': ['May miss best response', 'No comparison benefit'],
                    'use_when': 'Speed is critical or task is simple'
                }
            }
        
        return recommendations
    
    def _calculate_prompt_complexity(self, prompt: str) -> float:
        """
        Calculate complexity score for a prompt
        
        Args:
            prompt: Input prompt
            
        Returns:
            Complexity score (0.0 = simple, 1.0 = very complex)
        """
        complexity_score = 0.0
        
        # Length factor
        word_count = len(prompt.split())
        length_score = min(word_count / 100.0, 1.0)  # Normalize to 0-1
        complexity_score += length_score * 0.3
        
        # Complexity indicators
        complexity_indicators = {
            'high': ['analyze', 'compare', 'evaluate', 'comprehensive', 'detailed', 'strategy'],
            'medium': ['explain', 'describe', 'summarize', 'review', 'discuss'],
            'low': ['what', 'how', 'when', 'where', 'simple', 'quick']
        }
        
        prompt_lower = prompt.lower()
        
        for level, indicators in complexity_indicators.items():
            count = sum(1 for indicator in indicators if indicator in prompt_lower)
            if level == 'high':
                complexity_score += count * 0.3
            elif level == 'medium':
                complexity_score += count * 0.2
            # Low indicators actually reduce complexity
            else:
                complexity_score -= count * 0.1
        
        # Question vs statement (questions tend to be simpler)
        if prompt.strip().endswith('?'):
            complexity_score -= 0.1
        
        return max(min(complexity_score, 1.0), 0.0)
    
    def _generate_routing_reasoning(self, prompt_length: int, complexity_score: float, 
                                  performance_variance: float) -> List[str]:
        """
        Generate human-readable reasoning for routing mode selection
        
        Args:
            prompt_length: Number of words in prompt
            complexity_score: Calculated complexity score
            performance_variance: Recent performance variance across LLMs
            
        Returns:
            List of reasoning statements
        """
        reasoning = []
        
        if prompt_length > 50:
            reasoning.append(f"Long prompt ({prompt_length} words) benefits from comprehensive comparison")
        elif prompt_length < 20:
            reasoning.append(f"Short prompt ({prompt_length} words) suitable for fast mode")
        
        if complexity_score > 0.6:
            reasoning.append(f"High complexity score ({complexity_score:.2f}) suggests comprehensive mode")
        elif complexity_score < 0.3:
            reasoning.append(f"Low complexity score ({complexity_score:.2f}) suitable for fast mode")
        
        if performance_variance > 0.15:
            reasoning.append(f"High performance variance ({performance_variance:.2f}) - LLMs differ significantly")
        elif performance_variance < 0.05:
            reasoning.append(f"Low performance variance ({performance_variance:.2f}) - LLMs perform similarly")
        
        if not reasoning:
            reasoning.append("Based on balanced prompt characteristics")
        
        return reasoning
    
    def _update_enhanced_performance_tracking(self, prompt: str, response: LLMResponse, quality_score: float):
        """Update enhanced performance tracking with category and pattern analysis"""
        # Find similar examples to determine category
        similar_examples = self.vector_db_evaluator.find_similar_examples(prompt, top_k=1)
        category = 'general'  # default
        
        if similar_examples:
            category = similar_examples[0][0].category
        
        # Update category-specific performance
        self.llm_category_performance[response.llm_id][category].append(quality_score)
        
        # Track successful patterns (high quality responses)
        if quality_score > 0.7:  # Threshold for "successful"
            if len(self.successful_prompt_patterns[response.llm_id]) < 50:  # Limit storage
                self.successful_prompt_patterns[response.llm_id].append({
                    'prompt': prompt[:200],  # Truncate for memory
                    'category': category,
                    'quality': quality_score,
                    'response_time': response.generation_time
                })
        
        # Update category routing history
        self.category_routing_history[category].append({
            'llm_id': response.llm_id,
            'quality': quality_score,
            'timestamp': time.time()
        })
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get enhanced statistics including vector database and predictive routing metrics"""
        base_stats = self.get_stats()
        
        # Get implementation info from the evaluator
        impl_info = self.vector_db_evaluator.get_implementation_info()
        
        # Calculate category performance insights
        category_insights = {}
        for llm_id in self.llm_ids:
            llm_category_stats = {}
            for category, scores in self.llm_category_performance[llm_id].items():
                if scores:
                    llm_category_stats[category] = {
                        'avg_quality': float(np.mean(list(scores))),
                        'sample_count': len(scores),
                        'consistency': float(1.0 - np.std(list(scores))) if len(scores) > 1 else 1.0
                    }
            if llm_category_stats:
                category_insights[llm_id] = llm_category_stats
        
        # Calculate routing efficiency metrics
        total_predictions = len(self.quality_prediction_cache)
        successful_patterns_count = sum(len(patterns) for patterns in self.successful_prompt_patterns.values())
        
        # Enhanced vector database statistics
        base_stats["vector_db_stats"] = {
            "total_eval_records": len(self.vector_db_evaluator.human_eval_records),
            "implementation": impl_info.get("implementation", "Unknown"),
            "colbert_available": getattr(self.vector_db_evaluator, 'colbert_available', False),
            "similarity_threshold": self.vector_db_evaluator.similarity_threshold
        }
        
        # Add predictive routing statistics
        base_stats["predictive_routing_stats"] = {
            "quality_predictions_cached": total_predictions,
            "successful_patterns_learned": successful_patterns_count,
            "category_insights": category_insights,
            "optimization_weights": self.optimization_weights,
            "cache_efficiency": min(total_predictions / max(self.cache_max_size, 1), 1.0)
        }
        
        # Add multi-objective optimization insights
        if any(self.quality_scores.values()):
            overall_quality = np.mean([np.mean(list(scores)) for scores in self.quality_scores.values() if scores])
            overall_speed = np.mean([1.0/np.mean(list(times)) for times in self.response_times.values() if times])
            
            base_stats["optimization_insights"] = {
                "overall_quality_score": float(overall_quality),
                "overall_speed_factor": float(overall_speed),
                "quality_speed_tradeoff": float(overall_quality * overall_speed)
            }
        
        return base_stats
    
    def update_optimization_weights(self, quality_weight: float = None, 
                                  speed_weight: float = None, 
                                  cost_weight: float = None):
        """Update multi-objective optimization weights"""
        if quality_weight is not None:
            self.optimization_weights['quality'] = max(0, min(1, quality_weight))
        if speed_weight is not None:
            self.optimization_weights['speed'] = max(0, min(1, speed_weight))
        if cost_weight is not None:
            self.optimization_weights['cost'] = max(0, min(1, cost_weight))
        
        # Normalize weights to sum to 1
        total_weight = sum(self.optimization_weights.values())
        if total_weight > 0:
            for key in self.optimization_weights:
                self.optimization_weights[key] /= total_weight
    
    def get_category_recommendations(self, category: str) -> Dict[str, Any]:
        """Get LLM recommendations for a specific category"""
        recommendations = {
            'category': category,
            'best_llm': None,
            'performance_ranking': [],
            'insights': []
        }
        
        # Analyze performance by LLM for this category
        llm_performance = {}
        for llm_id in self.llm_ids:
            if category in self.llm_category_performance[llm_id]:
                scores = list(self.llm_category_performance[llm_id][category])
                if scores:
                    llm_performance[llm_id] = {
                        'avg_quality': np.mean(scores),
                        'consistency': 1.0 - np.std(scores) if len(scores) > 1 else 1.0,
                        'sample_count': len(scores)
                    }
        
        if llm_performance:
            # Rank LLMs by composite score
            ranked_llms = sorted(llm_performance.items(), 
                               key=lambda x: x[1]['avg_quality'] * x[1]['consistency'], 
                               reverse=True)
            
            recommendations['best_llm'] = ranked_llms[0][0]
            recommendations['performance_ranking'] = [
                {'llm_id': llm_id, **stats} for llm_id, stats in ranked_llms
            ]
            
            # Generate insights
            best_score = ranked_llms[0][1]['avg_quality']
            if best_score > 0.8:
                recommendations['insights'].append(f"Excellent performance available for {category}")
            elif best_score > 0.6:
                recommendations['insights'].append(f"Good performance available for {category}")
            else:
                recommendations['insights'].append(f"Limited high-quality options for {category}")
        else:
            recommendations['insights'].append(f"No historical data for category: {category}")
        
        return recommendations
    
    def clear_performance_cache(self):
        """Clear prediction cache and reset learning data"""
        self.quality_prediction_cache.clear()
        self.successful_prompt_patterns.clear()
        self.llm_category_performance.clear()
        self.category_routing_history.clear()
        print("Performance cache and learning data cleared")