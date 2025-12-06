"""
EVE Quick Training - Config ottimizzata per partire subito
"""

from config import EVEConfig
from data_loader import BitStreamDataset
from train import EvolutionaryTrainer

# Config LEGGERA - parte in secondi!
config = EVEConfig()

# LATTICE piccolo
config.lattice.size_per_dim = 4      # 4x4 invece di 8x8
config.lattice.dimensions = 2
config.lattice.beam_width = 2        # Ridotto da 4
config.lattice.max_steps = 8         # Ridotto da 16

# EVOLUTION leggera
config.evolution.population_size = 8  # Invece di 32
config.evolution.elite_ratio = 0.25
config.evolution.mutation_rate = 0.1
config.evolution.generations = 50    # Start con poche

# TRAINING batch piccoli
config.training.batch_size = 4       # Invece di 16
config.training.mask_ratio = 0.25
config.training.save_interval = 10

print("🚀 EVE Quick Training - Config Ottimizzata")
print("=" * 60)
print(f"Lattice:    {config.lattice.size_per_dim}x{config.lattice.size_per_dim}")
print(f"Population: {config.evolution.population_size}")
print(f"Batch size: {config.training.batch_size}")
print(f"Generations: {config.evolution.generations}")
print("=" * 60)

# Dataset path - MODIFICA QUESTO
dataset_path = input("\nDataset path: ").strip()
if not dataset_path:
    dataset_path = "/home/sam/Documents/TRAIN"  # Default

# Carica dataset
print(f"\n📁 Loading dataset from: {dataset_path}")
dataset = BitStreamDataset(
    dataset_path,
    chunk_size=256,          # Chunk più piccoli
    mask_ratio=config.training.mask_ratio
)

print(f"✓ Dataset loaded: {len(dataset)} chunks\n")

# Crea trainer
trainer = EvolutionaryTrainer(config, dataset)

# Training
print("🧬 Starting training...")
print("(Ctrl+C per interrompere)\n")

try:
    trainer.train(generations=config.evolution.generations, verbose=True)
except KeyboardInterrupt:
    print("\n\n⚠️  Training interrotto dall'utente")

print(f"\n✅ Done! Best fitness: {trainer.best_fitness:.3f}")
print(f"Checkpoints salvati in: {config.training.checkpoint_dir}")
