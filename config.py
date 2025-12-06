"""
EVE Configuration System
Gestisce tutte le impostazioni per training, inferenza, e sistema evolutivo
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class LatticeConfig:
    """Configurazione del lattice"""
    dimensions: int = 2
    size_per_dim: int = 8
    beam_width: int = 4
    max_steps: int = 16


@dataclass
class EvolutionConfig:
    """Configurazione sistema evolutivo"""
    population_size: int = 32
    elite_ratio: float = 0.25
    mutation_rate: float = 0.1
    generations: int = 100


@dataclass
class TrainingConfig:
    """Configurazione training"""
    mask_ratio: float = 0.3
    batch_size: int = 16
    dataset_path: Optional[str] = None
    save_interval: int = 10
    checkpoint_dir: str = "./checkpoints"


@dataclass
class InferenceConfig:
    """Configurazione inferenza/chat"""
    model_path: Optional[str] = None
    max_response_length: int = 512
    temperature: float = 1.0
    context_window: int = 1024


@dataclass
class EVEConfig:
    """Configurazione completa di EVE"""
    lattice: LatticeConfig
    evolution: EvolutionConfig
    training: TrainingConfig
    inference: InferenceConfig

    def __init__(
        self,
        lattice: Optional[LatticeConfig] = None,
        evolution: Optional[EvolutionConfig] = None,
        training: Optional[TrainingConfig] = None,
        inference: Optional[InferenceConfig] = None
    ):
        self.lattice = lattice or LatticeConfig()
        self.evolution = evolution or EvolutionConfig()
        self.training = training or TrainingConfig()
        self.inference = inference or InferenceConfig()

    def save(self, path: str):
        """Salva configurazione su file JSON"""
        config_dict = {
            'lattice': asdict(self.lattice),
            'evolution': asdict(self.evolution),
            'training': asdict(self.training),
            'inference': asdict(self.inference)
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'EVEConfig':
        """Carica configurazione da file JSON"""
        with open(path, 'r') as f:
            config_dict = json.load(f)

        return cls(
            lattice=LatticeConfig(**config_dict.get('lattice', {})),
            evolution=EvolutionConfig(**config_dict.get('evolution', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            inference=InferenceConfig(**config_dict.get('inference', {}))
        )

    def display(self) -> str:
        """Ritorna stringa formattata con tutte le settings"""
        lines = [
            "╔══════════════════════════════════════╗",
            "║        EVE CONFIGURATION             ║",
            "╚══════════════════════════════════════╝",
            "",
            "┌─ LATTICE ────────────────────────────┐",
            f"│ Dimensions:     {self.lattice.dimensions}                        │",
            f"│ Size per dim:   {self.lattice.size_per_dim}                        │",
            f"│ Beam width:     {self.lattice.beam_width}                        │",
            f"│ Max steps:      {self.lattice.max_steps}                       │",
            "└──────────────────────────────────────┘",
            "",
            "┌─ EVOLUTION ──────────────────────────┐",
            f"│ Population:     {self.evolution.population_size}                       │",
            f"│ Elite ratio:    {self.evolution.elite_ratio:.2f}                     │",
            f"│ Mutation rate:  {self.evolution.mutation_rate:.2f}                     │",
            f"│ Generations:    {self.evolution.generations}                      │",
            "└──────────────────────────────────────┘",
            "",
            "┌─ TRAINING ───────────────────────────┐",
            f"│ Mask ratio:     {self.training.mask_ratio:.2f}                     │",
            f"│ Batch size:     {self.training.batch_size}                       │",
            f"│ Save interval:  {self.training.save_interval}                       │",
            f"│ Checkpoint dir: {self.training.checkpoint_dir[:20]:20s} │",
            "└──────────────────────────────────────┘",
            "",
            "┌─ INFERENCE ──────────────────────────┐",
            f"│ Max length:     {self.inference.max_response_length}                      │",
            f"│ Temperature:    {self.inference.temperature:.2f}                     │",
            f"│ Context window: {self.inference.context_window}                     │",
            "└──────────────────────────────────────┘",
        ]
        return '\n'.join(lines)


# Default configuration globale
DEFAULT_CONFIG = EVEConfig()


if __name__ == "__main__":
    # Test configurazione
    config = EVEConfig()
    print(config.display())

    # Test save/load
    config.save("test_config.json")
    loaded = EVEConfig.load("test_config.json")
    print("\n[Configurazione caricata correttamente]")
