"""
Data models for the MultiLLMRouter system.

This module contains the core data structures used throughout the system.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class LLMRequest:
    """Structure for LLM request data"""
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stop_sequences: Optional[list] = None
    metadata: Dict[str, Any] = None


@dataclass
class LLMResponse:
    """Structure for LLM response data"""
    content: str
    generation_time: float
    token_count: int
    llm_id: str
    metadata: Dict[str, Any] = None
    cost: Optional[float] = None  # Total cost for this request


@dataclass
class HumanEvalRecord:
    """Human evaluation record for golden evaluation sets"""
    id: str
    prompt: str
    golden_output: str  # Best known output (human curated)
    category: str
    difficulty: str  # easy, medium, hard
    source: str  # human, expert, etc.