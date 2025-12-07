#!/usr/bin/env python3 -u
"""
Simple training script - NON MOCK
Usa il tuo dataset vero
"""

import sys
import os

# Forza unbuffered
os.environ['PYTHONUNBUFFERED'] = '1'

from config import EVEConfig
from data_loader import BitStreamDataset
from train import EvolutionaryTrainer

def main():
    # Chiedi dataset path
    dataset_path = input("Dataset path: ").strip()

    if not os.path.exists(dataset_path):
        print(f"❌ Path non esiste: {dataset_path}")
        return

    print(f"\n📁 Loading dataset...")
    sys.stdout.flush()

    # Config
    config = EVEConfig()
    config.evolution.population_size = 16
    config.evolution.generations = 100
    config.training.batch_size = 8
    config.lattice.size_per_dim = 6

    print(f"Config: pop={config.evolution.population_size}, gen={config.evolution.generations}")
    sys.stdout.flush()

    # Dataset
    dataset = BitStreamDataset(
        data_source=dataset_path,
        chunk_size=128,
        mask_ratio=0.25,
        seed=42
    )

    print(f"✓ Dataset loaded: {len(dataset)} chunks")
    sys.stdout.flush()

    # Training
    print("\n🧬 Starting training...")
    sys.stdout.flush()

    trainer = EvolutionaryTrainer(config, dataset)
    trainer.train(generations=config.evolution.generations, verbose=True)

    print("\n✓ Done!")

if __name__ == "__main__":
    main()
