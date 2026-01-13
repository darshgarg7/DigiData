from typing import List, Dict, Any, Callable
import numpy as np

class PWSTEvaluator:
    """
    Deterministic Progress-Weighted Semantic Trace evaluator.
    Complements LLM-based judging with milestone-based progress tracking.
    """
    def __init__(self, manifest: Dict[str, Any]):
        self.manifest = manifest
        self.milestones = manifest["milestones"]
        self.reference_steps = manifest["reference_steps"]

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def evaluate(
        self, 
        trajectory: List[Dict[str, Any]], 
        embedding_fn: Callable,
        ui_match_fn: Callable
    ) -> Dict[str, Any]:
        reached_ids = []
        m_idx = 0
        
        for state in trajectory:
            if m_idx >= len(self.milestones):
                break
                
            target = self.milestones[m_idx]
            
            # Visual check
            sim = self._cosine_similarity(embedding_fn(state["screenshot"]), target["visual_signature"])
            
            # Structural check
            struct_match = True
            if target.get("structural_anchor"):
                struct_match = ui_match_fn(state.get("ui_tree"), target["structural_anchor"])
                
            if sim >= target["thresholds"]["visual_sim"] and struct_match:
                reached_ids.append(target["id"])
                m_idx += 1

        progress = len(reached_ids) / len(self.milestones)
        efficiency = min(1.0, self.reference_steps / len(trajectory)) if trajectory else 0
        
        return {
            "pwst_score": round((progress * 0.8) + (efficiency * 0.2), 4),
            "milestones_reached": reached_ids,
            "bottleneck": self.milestones[m_idx]["id"] if m_idx < len(self.milestones) else None
        }
      
