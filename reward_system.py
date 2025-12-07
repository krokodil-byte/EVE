"""
EVE Advanced Reward System
Sistema di reward predittivo con metriche dettagliate
"""

from array_backend import xp
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class PredictionMetrics:
    """Metriche di predizione"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    hamming_distance: float = 0.0
    bit_correct: int = 0
    bit_total: int = 0

    def __str__(self):
        return (
            f"Acc: {self.accuracy:.3f} | "
            f"Prec: {self.precision:.3f} | "
            f"Rec: {self.recall:.3f} | "
            f"F1: {self.f1_score:.3f} | "
            f"Hamming: {self.hamming_distance:.3f}"
        )


@dataclass
class TrainingStats:
    """Statistiche di training"""
    generation: int = 0
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    worst_fitness: float = 0.0
    fitness_std: float = 0.0
    metrics: Optional[PredictionMetrics] = None
    history: deque = field(default_factory=lambda: deque(maxlen=100))

    def update(self, fitnesses: List[float], metrics: Optional[PredictionMetrics] = None):
        """Aggiorna stats con nuova generazione"""
        self.generation += 1
        self.best_fitness = max(fitnesses)
        self.avg_fitness = xp.mean(fitnesses)
        self.worst_fitness = min(fitnesses)
        self.fitness_std = xp.std(fitnesses)
        self.metrics = metrics

        self.history.append({
            'generation': self.generation,
            'best': self.best_fitness,
            'avg': self.avg_fitness,
            'worst': self.worst_fitness
        })

    def display(self) -> str:
        """Formatta stats per display"""
        lines = [
            f"Gen {self.generation:4d} | "
            f"Best: {self.best_fitness:.3f} | "
            f"Avg: {self.avg_fitness:.3f} ± {self.fitness_std:.3f} | "
            f"Worst: {self.worst_fitness:.3f}"
        ]
        if self.metrics:
            lines.append(f"         | {self.metrics}")
        return '\n'.join(lines)


class PredictiveReward:
    """
    Reward system per training predittivo
    Valuta la capacità di predire bit mascherati
    """

    def __init__(self, mask_positions: Optional[xp.ndarray] = None):
        """
        Args:
            mask_positions: Array booleano che indica quali bit sono mascherati
        """
        self.mask_positions = mask_positions
        self.stats = TrainingStats()

    def set_mask(self, mask_positions: xp.ndarray):
        """Imposta nuove posizioni mascherate"""
        self.mask_positions = mask_positions

    def compute_metrics(self, predicted: xp.ndarray, target: xp.ndarray) -> PredictionMetrics:
        """
        Calcola metriche dettagliate di predizione

        Args:
            predicted: Bit predetti
            target: Bit target (ground truth)

        Returns:
            PredictionMetrics object
        """
        if self.mask_positions is None:
            raise ValueError("mask_positions non impostato")

        predicted_masked = predicted[self.mask_positions]
        target_masked = target[self.mask_positions]

        if len(predicted_masked) == 0:
            return PredictionMetrics()

        correct = (predicted_masked == target_masked)
        accuracy = correct.sum() / len(correct)

        tp = ((predicted_masked == 1) & (target_masked == 1)).sum()
        fp = ((predicted_masked == 1) & (target_masked == 0)).sum()
        fn = ((predicted_masked == 0) & (target_masked == 1)).sum()
        tn = ((predicted_masked == 0) & (target_masked == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        hamming = (predicted_masked != target_masked).sum() / len(predicted_masked)

        return PredictionMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            hamming_distance=hamming,
            bit_correct=int(correct.sum()),
            bit_total=len(correct)
        )

    def __call__(self, predicted: xp.ndarray, target: xp.ndarray) -> float:
        """
        Calcola reward (fitness)

        Args:
            predicted: Bit array predetto
            target: Bit array target

        Returns:
            Fitness score (0.0 - 1.0)
        """
        metrics = self.compute_metrics(predicted, target)
        return metrics.accuracy


class MultiObjectiveReward:
    """
    Reward multi-obiettivo che combina accuracy e altre metriche
    """

    def __init__(
        self,
        mask_positions: Optional[xp.ndarray] = None,
        accuracy_weight: float = 0.7,
        f1_weight: float = 0.2,
        diversity_weight: float = 0.1
    ):
        self.predictive_reward = PredictiveReward(mask_positions)
        self.accuracy_weight = accuracy_weight
        self.f1_weight = f1_weight
        self.diversity_weight = diversity_weight
        self.prediction_history = deque(maxlen=10)

    def set_mask(self, mask_positions: xp.ndarray):
        self.predictive_reward.set_mask(mask_positions)

    def __call__(self, predicted: xp.ndarray, target: xp.ndarray) -> float:
        metrics = self.predictive_reward.compute_metrics(predicted, target)

        self.prediction_history.append(predicted.copy())
        diversity_score = self._compute_diversity()

        fitness = (
            self.accuracy_weight * metrics.accuracy +
            self.f1_weight * metrics.f1_score +
            self.diversity_weight * diversity_score
        )

        return fitness

    def _compute_diversity(self) -> float:
        """Calcola diversity score basato su varianza delle predizioni recenti"""
        if len(self.prediction_history) < 2:
            return 0.0

        recent = list(self.prediction_history)
        variances = []
        for i in range(len(recent[0])):
            bit_values = [pred[i] for pred in recent if i < len(pred)]
            if bit_values:
                variances.append(xp.var(bit_values))

        return xp.mean(variances) if variances else 0.0


class AdaptiveReward:
    """
    Reward system adattivo che aumenta difficoltà nel tempo
    """

    def __init__(self, initial_mask_ratio: float = 0.1, target_mask_ratio: float = 0.5):
        self.mask_ratio = initial_mask_ratio
        self.target_mask_ratio = target_mask_ratio
        self.current_performance = deque(maxlen=5)
        self.performance_threshold = 0.8

    def should_increase_difficulty(self) -> bool:
        """Determina se aumentare difficoltà"""
        if len(self.current_performance) < 5:
            return False

        avg_perf = xp.mean(list(self.current_performance))
        return avg_perf > self.performance_threshold and self.mask_ratio < self.target_mask_ratio

    def update_difficulty(self):
        """Aumenta mask ratio se performance è buona"""
        if self.should_increase_difficulty():
            old_ratio = self.mask_ratio
            self.mask_ratio = min(self.mask_ratio * 1.2, self.target_mask_ratio)
            return f"Difficulty increased: {old_ratio:.2f} -> {self.mask_ratio:.2f}"
        return None

    def record_performance(self, accuracy: float):
        """Registra performance per adattamento"""
        self.current_performance.append(accuracy)


if __name__ == "__main__":
    # Test reward system
    xp.random.seed(42)

    target = xp.random.randint(0, 2, 100).astype(xp.uint8)
    predicted = target.copy()
    mask = xp.random.rand(100) < 0.3
    predicted[mask] = xp.random.randint(0, 2, mask.sum()).astype(xp.uint8)

    reward = PredictiveReward(mask_positions=mask)
    metrics = reward.compute_metrics(predicted, target)

    print("Test Predictive Reward:")
    print(f"Target bits: {target[:20]}")
    print(f"Predicted:   {predicted[:20]}")
    print(f"Mask:        {mask[:20].astype(int)}")
    print(f"\n{metrics}")
    print(f"Fitness: {reward(predicted, target):.3f}")
