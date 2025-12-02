
# executor.py
import random
import numpy as np
from typing import List, Dict, Any, Tuple

from brain import LatticeMap, propagate
from interface import Translator, TextTranslator, RewardConnector, BitReconstructionReward, ExternalRewardConnector

BitArray = np.ndarray


class Executor:
    """
    Si occupa di:
    - creare popolazioni di mappe
    - generare task (via Translator)
    - far correre le mappe (via brain.propagate)
    - calcolare reward (via RewardConnector)
    - mutare/selezionare (evoluzione darwiniana)
    """
    def __init__(
        self,
        map_size: int = 4,
        map_dim: int = 2,
        population_size: int = 32,
        translator: Translator | None = None,
        reward_connector: RewardConnector | None = None,
    ):
        self.map_size = map_size
        self.map_dim = map_dim
        self.population_size = population_size

        self.population: List[LatticeMap] = [
            LatticeMap(size=map_size, dim=map_dim)
            for _ in range(population_size)
        ]

        self.translator = translator or TextTranslator()
        self.reward_connector = reward_connector or BitReconstructionReward()

        # punto di ingresso nel lattice (0,0,...)
        self.start_coord = tuple(0 for _ in range(map_dim))

    # =========================
    #  TASK GENERATION
    # =========================

    def sample_reconstruction_task(self) -> Tuple[Any, BitArray, Dict[str, Any]]:
        """
        Esempio: task di ricostruzione testo corrotto.
        Ritorna:
        - original_obj (es. stringa)
        - input_bits (bit corotti da dare al brain)
        - context (contiene target_bits, mask, ecc.)
        """
        # testo random come demo
        original_text = "hello world " + str(random.randint(0, 9999))
        target_bits = self.translator.encode(original_text)

        # maschera: nascondi ad esempio il 30% dei bit
        mask = np.random.rand(target_bits.size) < 0.3

        # input corrotto: i bit nascosti li mettiamo a 0
        input_bits = target_bits.copy()
        input_bits[mask] = 0

        context: Dict[str, Any] = {
            "target_bits": target_bits,
            "mask": mask,
            "task_type": "reconstruction",
        }
        return original_text, input_bits, context

    # =========================
    #  EVALUATION
    # =========================

    def eval_map_on_task(
        self,
        map_: LatticeMap,
        input_bits: BitArray,
        original: Any,
        context: Dict[str, Any],
        max_steps: int = 16,
        beam_width: int = 8,
    ) -> float:
        """
        - fa propagare la mappa
        - valuta i vari beam
        - prende il migliore secondo il reward_connector
        """
        beams = propagate(
            map_,
            x0=input_bits,
            start_coord=self.start_coord,
            max_steps=max_steps,
            beam_width=beam_width,
        )
        if not beams:
            return -1e9

        scored = []
        for b in beams:
            # qui passiamo SOLO i bit finali; se vuoi puoi aggiungere b.steps in context
            ctx = dict(context)
            ctx["steps"] = b.steps
            r = self.reward_connector(original, b.state, ctx)
            scored.append((r, b))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_reward, _ = scored[0]
        return best_reward

    # =========================
    #  EVOLUTION LOOP
    # =========================

    def evolve(
        self,
        num_generations: int = 100,
        batch_size: int = 8,
        max_steps: int = 16,
        beam_width: int = 8,
        mutation_rate: float = 0.02,
        elite_frac: float = 0.25,
    ):
        for gen in range(num_generations):
            # batch di task
            tasks = [self.sample_reconstruction_task() for _ in range(batch_size)]

            fitness: List[float] = []
            for m in self.population:
                total_r = 0.0
                for (original, input_bits, ctx) in tasks:
                    total_r += self.eval_map_on_task(
                        m, input_bits, original, ctx,
                        max_steps=max_steps, beam_width=beam_width
                    )
                fitness.append(total_r / batch_size)

            # ranking
            ranked = list(zip(fitness, self.population))
            ranked.sort(key=lambda x: x[0], reverse=True)

            best_f, _ = ranked[0]
            print(f"[gen {gen:03d}] best_fitness = {best_f:.4f}")

            # seleziona elite
            elite_count = max(1, int(self.population_size * elite_frac))
            elites = [m for _, m in ranked[:elite_count]]

            # nuova popolazione
            new_pop: List[LatticeMap] = []
            # tieni gli elite
            new_pop.extend(elites)

            # riempi con figli mutati dagli elite
            while len(new_pop) < self.population_size:
                parent = random.choice(elites)
                child = parent.clone()
                child.mutate(p_flip=mutation_rate)
                new_pop.append(child)

            self.population = new_pop

    # =========================
    #  FAST CONNECTOR SWITCH
    # =========================

    def set_reward_connector(self, connector: RewardConnector):
        """
        Cambi reward “al volo”:
        puoi attaccare qualsiasi funzione esterna (successo task reale, ecc.).
        """
        self.reward_connector = connector


if __name__ == "__main__":
    # Esempio di run base (ricostruzione testo corrotto su bit)
    ex = Executor(
        map_size=4,
        map_dim=2,
        population_size=32,
        translator=TextTranslator(),
        reward_connector=BitReconstructionReward(),
    )
    ex.evolve(num_generations=50)
