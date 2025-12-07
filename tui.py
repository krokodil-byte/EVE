#!/usr/bin/env python3
"""
EVE - Simple Training Interface
"""

import os
import sys
from pathlib import Path

# Force unbuffered
os.environ['PYTHONUNBUFFERED'] = '1'

from config import EVEConfig
from data_loader import BitStreamDataset
from train import EvolutionaryTrainer


def main():
    """Simple training interface"""

    print("\n" + "="*60)
    print("  EVE - Evolutionary Intelligence Training")
    print("="*60 + "\n")

    # Get dataset path
    dataset_path = input("Dataset path: ").strip()
    if not dataset_path or not Path(dataset_path).exists():
        print("❌ Invalid path!")
        return

    # Create config
    config = EVEConfig()

    # Ask for key parameters
    try:
        pop = input(f"Population size [{config.evolution.population_size}]: ").strip()
        if pop:
            config.evolution.population_size = int(pop)

        gens = input(f"Generations [{config.evolution.generations}]: ").strip()
        if gens:
            config.evolution.generations = int(gens)

        lattice = input(f"Lattice size [{config.lattice.size_per_dim}]: ").strip()
        if lattice:
            config.lattice.size_per_dim = int(lattice)

    except ValueError as e:
        print(f"❌ Invalid input: {e}")
        return

    # Load dataset
    print(f"\n📁 Loading dataset from: {dataset_path}")
    sys.stdout.flush()

    try:
        dataset = BitStreamDataset(
            data_source=dataset_path,
            chunk_size=128,
            mask_ratio=config.training.mask_ratio,
            seed=42
        )
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    print(f"✓ Dataset loaded: {len(dataset)} chunks\n")
    sys.stdout.flush()

    # Create trainer and train
    try:
        trainer = EvolutionaryTrainer(config, dataset)
        trainer.train(generations=config.evolution.generations, verbose=True)

        print(f"\n✓ Training completed!")
        print(f"Best model saved to: {config.training.checkpoint_dir}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
