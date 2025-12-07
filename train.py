"""
EVE Training Module
Training evolutivo per predizione di bit mascherati
"""

import os
import pickle
import numpy as np
from typing import List, Optional, Tuple
from datetime import datetime

from brain import LatticeMap, propagate
from reward_system import PredictiveReward, TrainingStats, PredictionMetrics
from data_loader import BitStreamDataset
from config import EVEConfig


class EvolutionaryTrainer:
    """
    Trainer evolutivo per EVE
    Evolve lattice per predire bit mascherati
    """

    def __init__(self, config: EVEConfig, dataset: BitStreamDataset):
        """
        Args:
            config: Configurazione EVE
            dataset: Dataset di bit stream
        """
        self.config = config
        self.dataset = dataset

        import sys
        print(f"Creating population of {config.evolution.population_size} lattice maps...")
        sys.stdout.flush()

        self.population = [
            LatticeMap(
                size=config.lattice.size_per_dim,
                dim=config.lattice.dimensions
            )
            for _ in range(config.evolution.population_size)
        ]

        print(f"✓ Population created ({config.lattice.size_per_dim}x{config.lattice.size_per_dim} lattice)")
        sys.stdout.flush()

        self.reward = PredictiveReward()
        self.stats = TrainingStats()

        self.best_map = None
        self.best_fitness = 0.0

    def evaluate_individual(
        self,
        map_: LatticeMap,
        input_bits: np.ndarray,
        target_bits: np.ndarray,
        mask_positions: np.ndarray,
        start_coord: Optional[Tuple[int, ...]] = None
    ) -> Tuple[float, np.ndarray]:
        """
        Valuta un singolo lattice su un task

        Args:
            map_: LatticeMap da valutare
            input_bits: Bit di input (con alcuni mascherati)
            target_bits: Bit target completi
            mask_positions: Posizioni mascherate
            start_coord: Coordinata di partenza nel lattice

        Returns:
            (fitness, predicted_bits)
        """
        if start_coord is None:
            start_coord = tuple([0] * map_.dim)

        self.reward.set_mask(mask_positions)

        try:
            final_states = propagate(
                map_,
                input_bits,
                start_coord,
                max_steps=self.config.lattice.max_steps,
                beam_width=self.config.lattice.beam_width
            )

            if not final_states:
                return 0.0, input_bits

            predicted_bits = final_states[0].state

            if len(predicted_bits) != len(target_bits):
                predicted_bits = np.resize(predicted_bits, len(target_bits))

            fitness = self.reward(predicted_bits, target_bits)

            return fitness, predicted_bits

        except Exception as e:
            import sys
            print(f"[ERROR] Evaluation failed: {e}")
            sys.stdout.flush()
            import traceback
            traceback.print_exc()
            return 0.0, input_bits

    def evaluate_population_on_batch(
        self,
        batch: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
    ) -> List[float]:
        """
        Valuta tutta la popolazione su un batch

        Args:
            batch: Lista di (input_bits, target_bits, mask_positions)

        Returns:
            Lista di fitness scores (uno per individuo)
        """
        fitnesses = []

        for map_ in self.population:
            batch_fitnesses = []

            for input_bits, target_bits, mask_positions in batch:
                fitness, _ = self.evaluate_individual(
                    map_, input_bits, target_bits, mask_positions
                )
                batch_fitnesses.append(fitness)

            avg_fitness = np.mean(batch_fitnesses)
            fitnesses.append(avg_fitness)

        return fitnesses

    def selection_and_reproduction(self, fitnesses: List[float]) -> List[LatticeMap]:
        """
        Selezione elitaria e riproduzione con mutazione

        Args:
            fitnesses: Fitness di ogni individuo

        Returns:
            Nuova popolazione
        """
        n_elite = int(len(self.population) * self.config.evolution.elite_ratio)
        n_elite = max(1, n_elite)

        sorted_indices = np.argsort(fitnesses)[::-1]
        elites_indices = sorted_indices[:n_elite]

        elites = [self.population[i] for i in elites_indices]

        if fitnesses[elites_indices[0]] > self.best_fitness:
            self.best_fitness = fitnesses[elites_indices[0]]
            self.best_map = elites[0].clone()

        new_population = []

        new_population.extend([elite.clone() for elite in elites])

        while len(new_population) < self.config.evolution.population_size:
            parent = np.random.choice(elites)
            child = parent.clone()
            child.mutate(p_flip=self.config.evolution.mutation_rate)
            new_population.append(child)

        return new_population

    def train_epoch(self, epoch: int) -> PredictionMetrics:
        """
        Esegue un'epoca di training (una generazione evolutiva)

        Args:
            epoch: Numero epoca

        Returns:
            Metriche del migliore individuo
        """
        import sys
        print(f"[Gen {epoch}] Evaluating population...", end='', flush=True)

        batch = self.dataset.get_batch(self.config.training.batch_size)

        fitnesses = self.evaluate_population_on_batch(batch)

        print(" Done.", flush=True)

        best_idx = np.argmax(fitnesses)
        input_bits, target_bits, mask_positions = batch[0]
        self.reward.set_mask(mask_positions)
        _, predicted = self.evaluate_individual(
            self.population[best_idx], input_bits, target_bits, mask_positions
        )
        metrics = self.reward.compute_metrics(predicted, target_bits)

        self.stats.update(fitnesses, metrics)

        self.population = self.selection_and_reproduction(fitnesses)

        return metrics

    def train(self, generations: Optional[int] = None, verbose: bool = True):
        """
        Esegue training completo

        Args:
            generations: Numero di generazioni (usa config se None)
            verbose: Se stampare progress
        """
        if generations is None:
            generations = self.config.evolution.generations

        import sys
        print("╔═══════════════════════════════════════════════════════╗")
        sys.stdout.flush()
        print("║              EVE EVOLUTIONARY TRAINING                ║")
        sys.stdout.flush()
        print("╚═══════════════════════════════════════════════════════╝")
        sys.stdout.flush()
        print(f"Population: {self.config.evolution.population_size}")
        sys.stdout.flush()
        print(f"Generations: {generations}")
        sys.stdout.flush()
        print(f"Dataset: {len(self.dataset)} chunks")
        sys.stdout.flush()
        print(f"Mask ratio: {self.dataset.mask_ratio:.1%}")
        sys.stdout.flush()
        print("─" * 57)
        sys.stdout.flush()

        import time
        for gen in range(generations):
            start_time = time.time()
            metrics = self.train_epoch(gen)
            elapsed = time.time() - start_time

            print(f" [{elapsed:.2f}s]", flush=True)

            if verbose and (gen % 10 == 0 or gen == generations - 1):
                print(self.stats.display())
                sys.stdout.flush()

            if (gen + 1) % self.config.training.save_interval == 0:
                self.save_checkpoint(gen + 1)

        print("─" * 57)
        print(f"Training completato! Best fitness: {self.best_fitness:.3f}")
        sys.stdout.flush()

    def save_checkpoint(self, generation: int):
        """Salva checkpoint del modello"""
        os.makedirs(self.config.training.checkpoint_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eve_gen{generation:04d}_{timestamp}.pkl"
        filepath = os.path.join(self.config.training.checkpoint_dir, filename)

        checkpoint = {
            'generation': generation,
            'best_map': self.best_map,
            'best_fitness': self.best_fitness,
            'config': self.config,
            'stats': self.stats
        }

        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)

        print(f"[Checkpoint] Salvato: {filepath}")
        import sys
        sys.stdout.flush()

    @staticmethod
    def load_checkpoint(filepath: str) -> Tuple[LatticeMap, EVEConfig]:
        """
        Carica checkpoint

        Returns:
            (best_map, config)
        """
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)

        return checkpoint['best_map'], checkpoint['config']
