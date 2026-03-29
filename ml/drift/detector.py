import logging
from datetime import datetime
from typing import Any, Dict, List
import numpy as np
from scipy import stats
from app.core.config import settings
from app.schemas.schemas import DriftCheckResponse, DriftReport

logger = logging.getLogger(__name__)

_REFERENCE_SCORES = np.concatenate([
    np.random.default_rng(42).beta(8, 2, 700),
    np.random.default_rng(42).beta(2, 8, 300),
])

class DriftDetector:
    def __init__(self):
        self._reference = _REFERENCE_SCORES.copy()
        self.history: List[Dict[str, Any]] = []
        self._stats_cache: Dict[str, float] = {}

    def check(self, current_scores: List[float]) -> DriftCheckResponse:
        current = np.array(current_scores, dtype=float)
        ks_stat, p_value = stats.ks_2samp(self._reference, current)
        drift_detected = p_value < settings.DRIFT_THRESHOLD
        report = DriftReport(feature="prediction_score", statistic=round(float(ks_stat), 4), p_value=round(float(p_value), 4), drift_detected=drift_detected, threshold=settings.DRIFT_THRESHOLD)
        response = DriftCheckResponse(drift_detected=drift_detected, reports=[report], sample_size=len(current_scores))
        self.history.append({"timestamp": datetime.utcnow().isoformat(), "drift_detected": drift_detected, "ks_stat": report.statistic, "p_value": report.p_value, "sample_size": len(current_scores)})
        if drift_detected:
            logger.warning(f"Drift detected! KS={ks_stat:.4f} p={p_value:.4f}")
        return response

    def reference_stats(self) -> Dict[str, float]:
        if not self._stats_cache:
            self._stats_cache = {"mean": round(float(np.mean(self._reference)), 4), "std": round(float(np.std(self._reference)), 4), "median": round(float(np.median(self._reference)), 4), "min": round(float(np.min(self._reference)), 4), "max": round(float(np.max(self._reference)), 4), "n_samples": len(self._reference)}
        return self._stats_cache

    def reset(self) -> None:
        self._reference = _REFERENCE_SCORES.copy()
        self._stats_cache = {}
        self.history = []
        logger.info("Drift detector reset.")
