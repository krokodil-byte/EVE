# debug_train.py - Trova dove si blocca EVE
from config import EVEConfig
from data_loader import BitStreamDataset
from brain import LatticeMap
import numpy as np
import sys

print("=" * 60)
print("DEBUG EVE TRAINING - Trova il blocco")
print("=" * 60)

# Test 1: LatticeMap
print("\n[1/5] Testing LatticeMap creation...")
sys.stdout.flush()
try:
    lattice = LatticeMap(size=3, dim=2)
    print(f"      ✓ Created: {len(lattice.nodes)} nodes, dim={lattice.dim}")
except Exception as e:
    print(f"      ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Propagate
print("\n[2/5] Testing propagate (with 3 sec timeout)...")
sys.stdout.flush()

from brain import propagate
import signal

def timeout_handler(signum, frame):
    print("      ✗ TIMEOUT - PROPAGATE BLOCKED!")
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(3)

try:
    test_bits = np.array([1,0,1,1,0,1,0,0], dtype=np.uint8)
    result = propagate(lattice, test_bits, (0,0), max_steps=4, beam_width=2)
    signal.alarm(0)
    print(f"      ✓ Works: {len(result)} states returned")
except Exception as e:
    signal.alarm(0)
    print(f"      ✗ FAILED: {e}")
    sys.exit(1)

# Test 3: Dataset - crea test dummy
print("\n[3/5] Testing dataset load...")
sys.stdout.flush()

import tempfile
import os
test_dir = tempfile.mkdtemp()
test_file = os.path.join(test_dir, "test.txt")
with open(test_file, 'w') as f:
    f.write("Test data for EVE" * 50)

try:
    dataset = BitStreamDataset(test_file, mask_ratio=0.2)
    print(f"      ✓ Loaded: {len(dataset)} chunks")
except Exception as e:
    print(f"      ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Trainer creation
print("\n[4/5] Creating EvolutionaryTrainer...")
sys.stdout.flush()

from train import EvolutionaryTrainer

config = EVEConfig()
config.evolution.population_size = 2  # MINIMAL
config.training.batch_size = 1
config.lattice.size_per_dim = 3

signal.alarm(5)
try:
    trainer = EvolutionaryTrainer(config, dataset)
    signal.alarm(0)
    print(f"      ✓ Trainer created with pop={len(trainer.population)}")
except Exception as e:
    signal.alarm(0)
    print(f"      ✗ FAILED during __init__: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: First epoch
print("\n[5/5] Running FIRST EPOCH (10 sec timeout)...")
sys.stdout.flush()

signal.alarm(10)
try:
    metrics = trainer.train_epoch(0)
    signal.alarm(0)
    print(f"      ✓ EPOCH COMPLETED! Accuracy: {metrics.accuracy:.3f}")
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - Training should work!")
    print("=" * 60)
except TimeoutError:
    signal.alarm(0)
    print("      ✗ TIMEOUT - train_epoch() BLOCKED!")
    print("\n🔴 BLOCCO IN train_epoch() - problema in evaluation/propagate loop")
    sys.exit(1)
except Exception as e:
    signal.alarm(0)
    print(f"      ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
