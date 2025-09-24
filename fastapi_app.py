#!/usr/bin/env python3
"""
FastAPI Application for LLM Router Service

This FastAPI application exposes the Vector Database Enhanced Multi-LLM Router
as REST APIs for easy integration with other services.

Endpoints:
- POST /init: Initialize vector index from golden evaluation set
- POST /query: Route queries through LLM router with quality evaluation
- GET /stats: Get current PID controller statistics for all LLMs
- GET /health: Health check endpoint
"""

import asyncio
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import our LLM Router components
from main import load_config, create_human_eval_set, smart_ingest_evaluation_set
from routers.vector_enhanced_router import VectorDBEnhancedRouter
from core.data_models import HumanEvalRecord

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global router instance
router_instance: Optional[VectorDBEnhancedRouter] = None
initialization_status = {
    "initialized": False,
    "index_loaded": False,
    "last_init_time": None,
    "total_records": 0,
    "error": None
}


# Pydantic Models for API Request/Response
class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    prompt: str = Field(..., description="The query prompt to route to an LLM", min_length=1, max_length=10000)
    max_tokens: Optional[int] = Field(None, description="Override default max tokens", ge=1, le=8192)
    temperature: Optional[float] = Field(None, description="Override default temperature", ge=0.0, le=2.0)
    timeout: Optional[int] = Field(None, description="Request timeout in seconds", ge=1, le=300)
    routing_mode: Optional[str] = Field(None, description="Force specific routing mode: 'auto', 'fast', 'multi_llm', 'comprehensive', 'quality'", 
                                      pattern="^(auto|fast|multi_llm|comprehensive|quality)$")


class LLMStats(BaseModel):
    """Statistics for a single LLM"""
    llm_id: str
    current_allocation: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    average_quality_score: float
    error_rate: float
    last_used: Optional[str] = None


class EvaluationScore(BaseModel):
    """Detailed evaluation scores for an LLM response"""
    golden_similarity: float = Field(..., description="Similarity to golden evaluation set")
    composite_score: float = Field(..., description="Overall composite evaluation score")
    length_score: Optional[float] = Field(None, description="Response length appropriateness")
    consensus_score: Optional[float] = Field(None, description="Agreement with other LLMs")
    ranking_position: Optional[int] = Field(None, description="Rank among all evaluated responses")


class SelectionDetails(BaseModel):
    """Detailed information about LLM selection process"""
    routing_mode: str = Field(..., description="Routing mode used (single_llm, multi_llm, etc.)")
    selection_reason: str = Field(..., description="Explanation of why this LLM was selected")
    similarity_threshold: float = Field(..., description="Similarity threshold used for evaluation")
    evaluated_llms: List[str] = Field(..., description="List of all LLMs that were evaluated")
    all_scores: Dict[str, EvaluationScore] = Field(..., description="Evaluation scores for all LLMs")
    selection_criteria: Dict[str, Any] = Field(..., description="Criteria used for selection decision")
    comparative_analysis: Optional[Dict[str, Any]] = Field(None, description="Comparative analysis between LLMs")


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    response_content: str
    selected_llm: str
    generation_time: float
    quality_score: float
    request_id: str
    timestamp: str
    llm_stats: LLMStats
    total_cost: Optional[float] = None
    selection_details: Optional[SelectionDetails] = Field(None, description="Detailed selection process information")


class InitRequest(BaseModel):
    """Request model for init endpoint"""
    force_rebuild: bool = Field(False, description="Force rebuild of vector index even if it exists")


class InitResponse(BaseModel):
    """Response model for init endpoint"""
    success: bool
    message: str
    total_records: int
    initialization_time: float
    vector_implementation: str


class StatsResponse(BaseModel):
    """Response model for stats endpoint"""
    system_status: str
    initialized: bool
    total_queries: int
    uptime_seconds: float
    llm_stats: List[LLMStats]
    pid_controller_status: Dict[str, Any]


class HealthResponse(BaseModel):
    """Response model for health endpoint"""
    status: str
    timestamp: str
    version: str = "1.0.0"
    router_initialized: bool
    index_loaded: bool


# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown"""
    logger.info("Starting LLM Router FastAPI service...")
    
    # Startup
    try:
        await initialize_router()
        logger.info("LLM Router service started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize router on startup: {e}")
        # Continue anyway - router can be initialized via API
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM Router service...")
    # Cleanup resources if needed


# FastAPI app instance
app = FastAPI(
    title="LLM Router API",
    description="Vector Database Enhanced Multi-LLM Router with PID Control",
    version="1.0.0",
    lifespan=lifespan
)

# Store app start time for uptime calculation
app_start_time = time.time()


async def initialize_router(force_rebuild: bool = False) -> Dict[str, Any]:
    """Initialize the router and vector index"""
    global router_instance, initialization_status
    
    start_time = time.time()
    
    try:
        # STRICT MODE: Load configuration with real providers ONLY - no fallbacks
        config = load_config(use_real_providers=True)
        logger.info("Configuration loaded successfully with real providers")
        
        # Validate that real providers were created
        if not config.get("providers") or not config.get("use_real_providers", False):
            raise ValueError("Real provider initialization failed - no providers were created")
        
        # Initialize router with real providers ONLY
        router_instance = VectorDBEnhancedRouter(
            llm_configs=config["llm_configs"],
            providers=config["providers"],
            base_allocation=config["base_allocation"],
            vector_db_config=config["vector_db_config"],
            use_real_providers=True
        )
        logger.info(f"Router instance created with real providers: {list(config['providers'].keys())}")
        
        # Load human evaluation set
        eval_set = create_human_eval_set()
        logger.info(f"Loaded {len(eval_set)} evaluation records")
        
        # Initialize vector index
        smart_ingest_evaluation_set(
            router_instance.vector_db_evaluator, 
            eval_set, 
            force_rebuild=force_rebuild
        )
        
        initialization_time = time.time() - start_time
        
        # Update status
        initialization_status.update({
            "initialized": True,
            "index_loaded": True,
            "last_init_time": time.time(),
            "total_records": len(eval_set),
            "error": None
        })
        
        logger.info(f"Router initialization completed in {initialization_time:.2f}s")
        
        return {
            "success": True,
            "message": "Router initialized successfully",
            "total_records": len(eval_set),
            "initialization_time": initialization_time,
            "vector_implementation": router_instance.vector_db_evaluator.get_implementation_info()["implementation"]
        }
        
    except Exception as e:
        error_msg = f"Router initialization failed: {str(e)}"
        logger.error(error_msg)
        
        initialization_status.update({
            "initialized": False,
            "index_loaded": False,
            "error": error_msg
        })
        
        raise HTTPException(status_code=500, detail=error_msg)


def get_llm_stats(llm_id: str) -> LLMStats:
    """Get statistics for a specific LLM"""
    if not router_instance:
        raise HTTPException(status_code=503, detail="Router not initialized")
    
    # Get stats from router
    base_stats = router_instance.get_stats()
    enhanced_stats = router_instance.get_enhanced_stats()
    
    # Extract data for specific LLM
    current_allocation = base_stats.get('current_allocation', {}).get(llm_id, 0.0)
    performance_summary = base_stats.get('performance_summary', {}).get(llm_id, {})
    error_count = base_stats.get('error_counts', {}).get(llm_id, 0)
    
    total_requests = performance_summary.get('total_requests', 0)
    successful_requests = max(0, total_requests - error_count)
    
    return LLMStats(
        llm_id=llm_id,
        current_allocation=current_allocation,
        total_requests=total_requests,
        successful_requests=successful_requests,
        failed_requests=error_count,
        average_response_time=performance_summary.get('avg_response_time', 0.0),
        average_quality_score=performance_summary.get('avg_quality', 0.0),
        error_rate=error_count / max(total_requests, 1),
        last_used=None  # This info is not available in current stats
    )


def _convert_evaluation_to_selection_details(detailed_evaluation: Dict[str, Any]) -> Optional[SelectionDetails]:
    """
    Convert detailed_evaluation from router to SelectionDetails format for API response
    
    Args:
        detailed_evaluation: Evaluation details from router
        
    Returns:
        SelectionDetails object or None if evaluation is incomplete
    """
    if not detailed_evaluation:
        return None
        
    try:
        # Extract basic routing information
        routing_mode = detailed_evaluation.get('mode', 'unknown')
        selection_reason = detailed_evaluation.get('selection_reason', 'No reason provided')
        
        # Get evaluation scores if available
        all_evaluations = detailed_evaluation.get('all_evaluations', {})
        evaluated_llms = list(all_evaluations.keys()) if all_evaluations else []
        
        # Convert evaluation scores to EvaluationScore objects
        all_scores = {}
        for llm_name, eval_data in all_evaluations.items():
            if isinstance(eval_data, dict):
                all_scores[llm_name] = EvaluationScore(
                    golden_similarity=eval_data.get('golden_similarity', 0.0),
                    composite_score=eval_data.get('composite_score', eval_data.get('golden_similarity', 0.0)),
                    length_score=eval_data.get('length_score'),
                    consensus_score=eval_data.get('consensus_score'),
                    ranking_position=eval_data.get('rank')
                )
        
        # Extract selection criteria and comparative analysis
        selection_criteria = detailed_evaluation.get('selection_criteria', {
            'primary_metric': 'composite_score',
            'threshold_used': 0.7,
            'weighting_factors': 'golden_similarity + consensus + length'
        })
        
        comparative_analysis = detailed_evaluation.get('comparative_analysis', {})
        
        return SelectionDetails(
            routing_mode=routing_mode,
            selection_reason=selection_reason,
            similarity_threshold=selection_criteria.get('threshold_used', 0.7),
            evaluated_llms=evaluated_llms,
            all_scores=all_scores,
            selection_criteria=selection_criteria,
            comparative_analysis=comparative_analysis if comparative_analysis else None
        )
        
    except Exception as e:
        logger.warning(f"Failed to convert evaluation details: {e}")
        # Return basic details if conversion fails
        return SelectionDetails(
            routing_mode=detailed_evaluation.get('mode', 'unknown'),
            selection_reason=detailed_evaluation.get('reason', 'Conversion error occurred'),
            similarity_threshold=0.7,
            evaluated_llms=[],
            all_scores={},
            selection_criteria={'error': str(e)},
            comparative_analysis=None
        )


# API Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        router_initialized=initialization_status["initialized"],
        index_loaded=initialization_status["index_loaded"]
    )


@app.post("/init", response_model=InitResponse)
async def initialize_system(request: InitRequest, background_tasks: BackgroundTasks):
    """Initialize the vector index from golden evaluation set"""
    logger.info(f"Initialize request received (force_rebuild={request.force_rebuild})")
    
    try:
        result = await initialize_router(force_rebuild=request.force_rebuild)
        return InitResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {e}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_llm(request: QueryRequest):
    """Route query through LLM router with quality evaluation"""
    if not router_instance or not initialization_status["initialized"]:
        raise HTTPException(status_code=503, detail="Router not initialized. Please call /init first.")
    
    logger.info(f"Query request received: {request.prompt[:100]}...")
    
    try:
        start_time = time.time()
        
        # Use routing mode from request, default to "auto" if not specified
        routing_mode = request.routing_mode or "auto"
        logger.info(f"Using routing mode: {routing_mode} (requested: {request.routing_mode})")
        
        # Route request with adaptive evaluation (includes detailed selection statistics)
        response, quality_score, detailed_evaluation = router_instance.route_request_adaptive(request.prompt, mode=routing_mode)
        
        generation_time = time.time() - start_time
        request_id = f"req_{int(time.time() * 1000)}"
        
        # Get current stats for the selected LLM
        selected_llm_stats = get_llm_stats(response.llm_id)
        
        # Convert detailed_evaluation to SelectionDetails format
        selection_details = _convert_evaluation_to_selection_details(detailed_evaluation)
        
        return QueryResponse(
            response_content=response.content,
            selected_llm=response.llm_id,
            generation_time=generation_time,
            quality_score=quality_score,
            request_id=request_id,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            llm_stats=selected_llm_stats,
            total_cost=response.cost,
            selection_details=selection_details
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/stats", response_model=StatsResponse)
async def get_system_stats():
    """Get current system and PID controller statistics"""
    if not router_instance:
        raise HTTPException(status_code=503, detail="Router not initialized")
    
    try:
        # Get router statistics
        base_stats = router_instance.get_stats()
        enhanced_stats = router_instance.get_enhanced_stats()
        
        # Calculate total queries from performance summary
        total_queries = sum(
            perf.get('total_requests', 0) 
            for perf in base_stats.get('performance_summary', {}).values()
        )
        
        # Get stats for all LLMs
        all_llm_stats = []
        for llm_id in router_instance.llm_ids:
            all_llm_stats.append(get_llm_stats(llm_id))
        
        # Calculate uptime
        uptime = time.time() - app_start_time
        
        return StatsResponse(
            system_status="operational" if initialization_status["initialized"] else "not_initialized",
            initialized=initialization_status["initialized"],
            total_queries=total_queries,
            uptime_seconds=uptime,
            llm_stats=all_llm_stats,
            pid_controller_status={
                "enhanced_stats": enhanced_stats.get('predictive_routing_stats', {}),
                "vector_db_stats": enhanced_stats.get('vector_db_stats', {}),
                "last_init_time": initialization_status["last_init_time"],
                "total_indexed_records": initialization_status["total_records"]
            }
        )
        
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")


# Additional utility endpoints

@app.get("/status")
async def get_simple_status():
    """Simple status endpoint for monitoring"""
    return {
        "status": "online",
        "initialized": initialization_status["initialized"],
        "uptime": time.time() - app_start_time,
        "version": "1.0.0"
    }


@app.get("/config")
async def get_configuration():
    """Get current router configuration (without sensitive data)"""
    if not router_instance:
        raise HTTPException(status_code=503, detail="Router not initialized")
    
    try:
        config = load_config()
        
        # Remove sensitive information
        safe_config = {
            "llm_configs": {
                llm_id: {
                    "base_response_time": cfg["base_response_time"],
                    "quality_factor": cfg["quality_factor"],
                    "error_rate": cfg["error_rate"]
                }
                for llm_id, cfg in config["llm_configs"].items()
            },
            "base_allocation": config["base_allocation"],
            "vector_db_config": {
                "similarity_threshold": config["vector_db_config"]["similarity_threshold"],
                "index_path": config["vector_db_config"]["index_path"]
            }
        }
        
        return safe_config
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Configuration retrieval failed: {str(e)}")


if __name__ == "__main__":
    import os
    import sys
    
    # Enable debug mode if DEBUG environment variable is set
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    
    if debug_mode:
        # Set logging level to DEBUG for more verbose output
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        
        # Add debug logging
        logger.debug("Debug mode enabled")
        logger.debug(f"Python path: {sys.executable}")
        logger.debug(f"Working directory: {os.getcwd()}")
        
        print("🐛 DEBUG MODE ENABLED")
        print("📍 Set breakpoints in your code and use F5 to start debugging")
        print("Use the VS Code debugger with the 'Debug FastAPI App Directly' configuration")
    
    # Run with uvicorn for development
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug" if debug_mode else "info",
        access_log=True
    )