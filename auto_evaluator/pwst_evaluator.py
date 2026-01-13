from typing import List, Dict, Any, Callable
import numpy as np

class PWSTEvaluator:
    """
    Deterministic Progress-Weighted Semantic Trace evaluator.
    """
    def __init__(
        self,
        manifest: Dict[str, Any],
        progress_weight: float = 0.8,
        efficiency_weight: float = 0.2
    ):
        self.manifest = manifest
        self.milestones = manifest["milestones"]
        self.reference_steps = manifest["reference_steps"]
        self.progress_weight = progress_weight
        self.efficiency_weight = efficiency_weight

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        return float(np.dot(v1, v2) / denom) if denom > 0 else 0.0

    def evaluate(
        self,
        trajectory: List[Dict[str, Any]],
        embedding_fn: Callable,
        ui_match_fn: Callable
    ) -> Dict[str, Any]:
        reached_ids = []
        missed_ids = []
        m_idx = 0

        for state in trajectory:
            if m_idx >= len(self.milestones):
                break

            target = self.milestones[m_idx]

            sim = self._cosine_similarity(
                embedding_fn(state["screenshot"]),
                target["visual_signature"]
            )

            struct_match = True
            if target.get("structural_anchor"):
                struct_match = ui_match_fn(
                    state.get("ui_tree"),
                    target["structural_anchor"]
                )

            if sim >= target["thresholds"]["visual_sim"] and struct_match:
                reached_ids.append(target["id"])

                if target["dependency_type"] == "strict":
                    m_idx += 1
                else:
                    # optional or skip-tolerant
                    m_idx += 1

        total = len(self.milestones)
        progress = len(reached_ids) / total if total > 0 else 0.0
        efficiency = (
            min(1.0, self.reference_steps / len(trajectory))
            if trajectory else 0.0
        )

        missed_ids = [
            m["id"] for m in self.milestones
            if m["id"] not in reached_ids
        ]

        score = (
            self.progress_weight * progress +
            self.efficiency_weight * efficiency
        )

        return {
            "pwst_score": round(score, 4),
            "progress": round(progress, 4),
            "efficiency": round(efficiency, 4),
            "milestones_reached": reached_ids,
            "milestones_missed": missed_ids,
            "bottleneck_id": (
                self.milestones[m_idx]["id"]
                if m_idx < total else None
            )
        }
        
