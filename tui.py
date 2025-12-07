#!/usr/bin/env python3
"""
EVE - Text User Interface with Menu System
Menu-based interface that uses CLI core underneath
"""

import os
import sys
from pathlib import Path

# Force unbuffered
os.environ['PYTHONUNBUFFERED'] = '1'

from config import EVEConfig
from data_loader import BitStreamDataset
from train import EvolutionaryTrainer


class EVETUI:
    """Menu-based TUI using CLI core"""

    def __init__(self):
        self.config = EVEConfig()
        self.dataset = None
        self.trainer = None

    def clear(self):
        os.system('clear' if os.name != 'nt' else 'cls')

    def banner(self):
        print("\n" + "="*60)
        print("  EVE - Evolutionary Intelligence")
        print("="*60 + "\n")

    def main_menu(self):
        """Main menu loop"""
        while True:
            self.clear()
            self.banner()

            print("1. 🎯 Train Model")
            print("2. ⚙️  Configure Settings")
            print("3. 📊 View Current Config")
            print("4. 💾 Save Config")
            print("5. 📁 Load Config")
            print("6. 🚪 Exit")

            choice = input("\nChoice: ").strip()

            if choice == '1':
                self.train_menu()
            elif choice == '2':
                self.settings_menu()
            elif choice == '3':
                self.view_config()
            elif choice == '4':
                self.save_config()
            elif choice == '5':
                self.load_config()
            elif choice == '6':
                print("\n👋 Goodbye!")
                break

    def train_menu(self):
        """Training menu - uses CLI core"""
        # Don't clear screen - let training output be visible
        print("\n" + "="*60)
        print("  TRAINING MODE")
        print("="*60 + "\n")

        # Get dataset path
        dataset_path = input("Dataset path: ").strip()
        if not dataset_path or not Path(dataset_path).exists():
            print("❌ Invalid path!")
            input("\nPress Enter...")
            return

        # Ask for generations (optional override)
        gens = input(f"Generations [{self.config.evolution.generations}]: ").strip()
        if gens:
            gen_count = int(gens)
        else:
            gen_count = self.config.evolution.generations

        # Load dataset
        print(f"\n📁 Loading dataset from: {dataset_path}")
        sys.stdout.flush()

        try:
            self.dataset = BitStreamDataset(
                data_source=dataset_path,
                chunk_size=128,
                mask_ratio=self.config.training.mask_ratio,
                seed=42
            )
        except Exception as e:
            print(f"❌ Error: {e}")
            input("\nPress Enter...")
            return

        print(f"✓ Dataset loaded: {len(self.dataset)} chunks\n")
        sys.stdout.flush()

        # Train using CLI core
        try:
            self.trainer = EvolutionaryTrainer(self.config, self.dataset)
            self.trainer.train(generations=gen_count, verbose=True)

            print(f"\n✓ Training completed!")
            print(f"Saved to: {self.config.training.checkpoint_dir}")

        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted")
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()

        input("\nPress Enter...")

    def settings_menu(self):
        """Settings configuration menu"""
        while True:
            self.clear()
            self.banner()
            print("─ SETTINGS ─\n")

            print("1. Lattice Settings")
            print("2. Evolution Settings")
            print("3. Training Settings")
            print("4. Backend Settings (CPU/GPU)")
            print("5. Back")

            choice = input("\nChoice: ").strip()

            if choice == '1':
                self.config_lattice()
            elif choice == '2':
                self.config_evolution()
            elif choice == '3':
                self.config_training()
            elif choice == '4':
                self.config_backend()
            elif choice == '5':
                break

    def config_lattice(self):
        """Configure lattice parameters"""
        self.clear()
        self.banner()
        print("─ LATTICE CONFIGURATION ─\n")

        print(f"Current dimensions: {self.config.lattice.dimensions}")
        print(f"Current size per dim: {self.config.lattice.size_per_dim}")
        print(f"Current beam width: {self.config.lattice.beam_width}")
        print(f"Current max steps: {self.config.lattice.max_steps}\n")

        dims = input("Dimensions [Enter to keep]: ").strip()
        if dims:
            self.config.lattice.dimensions = int(dims)

        size = input("Size per dimension [Enter to keep]: ").strip()
        if size:
            self.config.lattice.size_per_dim = int(size)

        beam = input("Beam width [Enter to keep]: ").strip()
        if beam:
            self.config.lattice.beam_width = int(beam)

        steps = input("Max steps [Enter to keep]: ").strip()
        if steps:
            self.config.lattice.max_steps = int(steps)

        print("\n✓ Lattice settings updated")
        input("Press Enter...")

    def config_evolution(self):
        """Configure evolution parameters"""
        self.clear()
        self.banner()
        print("─ EVOLUTION CONFIGURATION ─\n")

        print(f"Current population: {self.config.evolution.population_size}")
        print(f"Current elite ratio: {self.config.evolution.elite_ratio}")
        print(f"Current mutation rate: {self.config.evolution.mutation_rate}")
        print(f"Current generations: {self.config.evolution.generations}\n")

        pop = input("Population size [Enter to keep]: ").strip()
        if pop:
            self.config.evolution.population_size = int(pop)

        elite = input("Elite ratio (0.0-1.0) [Enter to keep]: ").strip()
        if elite:
            self.config.evolution.elite_ratio = float(elite)

        mut = input("Mutation rate (0.0-1.0) [Enter to keep]: ").strip()
        if mut:
            self.config.evolution.mutation_rate = float(mut)

        gens = input("Generations [Enter to keep]: ").strip()
        if gens:
            self.config.evolution.generations = int(gens)

        print("\n✓ Evolution settings updated")
        input("Press Enter...")

    def config_training(self):
        """Configure training parameters"""
        self.clear()
        self.banner()
        print("─ TRAINING CONFIGURATION ─\n")

        print(f"Current mask ratio: {self.config.training.mask_ratio}")
        print(f"Current batch size: {self.config.training.batch_size}")
        print(f"Current save interval: {self.config.training.save_interval}\n")

        mask = input("Mask ratio (0.0-1.0) [Enter to keep]: ").strip()
        if mask:
            self.config.training.mask_ratio = float(mask)

        batch = input("Batch size [Enter to keep]: ").strip()
        if batch:
            self.config.training.batch_size = int(batch)

        interval = input("Save interval [Enter to keep]: ").strip()
        if interval:
            self.config.training.save_interval = int(interval)

        print("\n✓ Training settings updated")
        input("Press Enter...")

    def view_config(self):
        """Display current configuration"""
        self.clear()
        self.banner()
        print(self.config.display())
        input("\nPress Enter...")

    def save_config(self):
        """Save configuration to file"""
        self.clear()
        self.banner()

        path = input("Config file path [./eve_config.json]: ").strip()
        if not path:
            path = "./eve_config.json"

        try:
            self.config.save(path)
            print(f"\n✓ Config saved to: {path}")
        except Exception as e:
            print(f"\n❌ Error: {e}")

        input("Press Enter...")

    def load_config(self):
        """Load configuration from file"""
        self.clear()
        self.banner()

        path = input("Config file path: ").strip()

        try:
            self.config = EVEConfig.load(path)
            print(f"\n✓ Config loaded from: {path}")
        except Exception as e:
            print(f"\n❌ Error: {e}")

        input("Press Enter...")

    def config_backend(self):
        """Configure computation backend (CPU/GPU)"""
        self.clear()
        self.banner()
        print("─ BACKEND CONFIGURATION ─\n")

        from array_backend import BACKEND, GPU_AVAILABLE

        print(f"Current backend: {BACKEND.upper()}")
        if GPU_AVAILABLE:
            print("✓ CuPy (GPU) is available")
        else:
            print("✗ CuPy (GPU) not installed")

        current_pref = os.environ.get('EVE_BACKEND', 'auto')
        print(f"Current preference: {current_pref}\n")

        print("Options:")
        print("1. Auto (use GPU if available)")
        print("2. Force CPU (use NumPy even if CuPy available)")
        print("3. Cancel")

        choice = input("\nChoice: ").strip()

        if choice == '1':
            preference = 'auto'
        elif choice == '2':
            preference = 'cpu'
        else:
            input("\nPress Enter...")
            return

        # Save preference to file
        with open('.eve_backend', 'w') as f:
            f.write(preference)

        print(f"\n✓ Backend preference set to: {preference}")
        print("\n⚠️  You need to RESTART the program for this to take effect!")
        print("\nTo restart with new backend:")
        print(f"  EVE_BACKEND={preference} python3 tui.py")
        print("\nOr just run: python3 tui.py (reads from .eve_backend)")

        input("\nPress Enter...")


def main():
    """Main entry point"""
    tui = EVETUI()
    tui.main_menu()


if __name__ == "__main__":
    main()
