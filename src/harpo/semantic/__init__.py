"""
HARPO Semantic Trajectory Intelligence

Lightweight, heuristic analyzers for understanding WHY trajectories degrade.
No external dependencies — all pattern matching + token overlap.
"""

from .analyzer    import SemanticAnalysis, SemanticTrajectoryAnalyzer
from .contradiction import ContradictionResult, ContradictionEvent, detect_contradictions
from .assumptions  import AssumptionChain, AssumptionPropagationResult, analyze_assumption_propagation
from .reflection   import ReflectionEffect, ReflectionResult, analyze_reflection_effectiveness
from .coherence    import TurnCoherence, CoherenceResult, score_semantic_coherence

__all__ = [
    "SemanticTrajectoryAnalyzer",
    "SemanticAnalysis",
    "ContradictionResult",
    "ContradictionEvent",
    "detect_contradictions",
    "AssumptionChain",
    "AssumptionPropagationResult",
    "analyze_assumption_propagation",
    "ReflectionEffect",
    "ReflectionResult",
    "analyze_reflection_effectiveness",
    "TurnCoherence",
    "CoherenceResult",
    "score_semantic_coherence",
]
