#!/usr/bin/env python -u
"""
EVE Text User Interface (TUI)
Interfaccia interattiva per training e chat
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Forza VERAMENTE unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from config import EVEConfig
from data_loader import BitStreamDataset
from train import EvolutionaryTrainer
from inference import EVEInference, EVEChatSession, load_trained_model


class EVETUI:
    """
    Text User Interface per EVE
    """

    def __init__(self):
        self.config = EVEConfig()
        self.trainer: Optional[EvolutionaryTrainer] = None
        self.inference: Optional[EVEInference] = None
        self.dataset: Optional[BitStreamDataset] = None

    def clear_screen(self):
        """Pulisce lo schermo"""
        os.system('clear' if os.name != 'nt' else 'cls')

    def print_banner(self):
        """Stampa banner EVE"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ███████╗██╗   ██╗███████╗                                 ║
║   ██╔════╝██║   ██║██╔════╝                                 ║
║   █████╗  ██║   ██║█████╗                                   ║
║   ██╔══╝  ╚██╗ ██╔╝██╔══╝                                   ║
║   ███████╗ ╚████╔╝ ███████╗                                 ║
║   ╚══════╝  ╚═══╝  ╚══════╝                                 ║
║                                                              ║
║        Evolutionary Intelligence Through Bit Evolution       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
        print(banner)

    def print_menu(self):
        """Stampa menu principale"""
        print("\n┌─ MAIN MENU ──────────────────────────────────────────┐")
        print("│                                                      │")
        print("│  1. 🎯 Train new model                               │")
        print("│  2. 💬 Chat with trained model                       │")
        print("│  3. ⚙️  Configure settings                            │")
        print("│  4. 📊 View current configuration                    │")
        print("│  5. 💾 Save configuration                            │")
        print("│  6. 📁 Load configuration                            │")
        print("│  7. 🚪 Exit                                          │")
        print("│                                                      │")
        print("└──────────────────────────────────────────────────────┘")

    def configure_settings(self):
        """Menu configurazione interattiva"""
        while True:
            self.clear_screen()
            self.print_banner()
            print("\n┌─ CONFIGURATION ──────────────────────────────────────┐")
            print("│  1. Lattice settings                                 │")
            print("│  2. Evolution settings                               │")
            print("│  3. Training settings                                │")
            print("│  4. Inference settings                               │")
            print("│  5. Back to main menu                                │")
            print("└──────────────────────────────────────────────────────┘")

            choice = input("\nSelect: ").strip()

            if choice == '1':
                self.config_lattice()
            elif choice == '2':
                self.config_evolution()
            elif choice == '3':
                self.config_training()
            elif choice == '4':
                self.config_inference()
            elif choice == '5':
                break

    def config_lattice(self):
        """Configura parametri lattice"""
        print("\n┌─ LATTICE CONFIGURATION ──────────────────────────────┐")
        print(f"Current dimensions: {self.config.lattice.dimensions}")
        print(f"Current size per dim: {self.config.lattice.size_per_dim}")
        print(f"Current beam width: {self.config.lattice.beam_width}")
        print(f"Current max steps: {self.config.lattice.max_steps}")
        print("└──────────────────────────────────────────────────────┘")

        dims = input("Dimensions [Enter to keep current]: ").strip()
        if dims:
            self.config.lattice.dimensions = int(dims)

        size = input("Size per dimension [Enter to keep current]: ").strip()
        if size:
            self.config.lattice.size_per_dim = int(size)

        beam = input("Beam width [Enter to keep current]: ").strip()
        if beam:
            self.config.lattice.beam_width = int(beam)

        steps = input("Max steps [Enter to keep current]: ").strip()
        if steps:
            self.config.lattice.max_steps = int(steps)

        print("\n✓ Lattice configuration updated")
        input("Press Enter to continue...")

    def config_evolution(self):
        """Configura parametri evolutivi"""
        print("\n┌─ EVOLUTION CONFIGURATION ────────────────────────────┐")
        print(f"Current population: {self.config.evolution.population_size}")
        print(f"Current elite ratio: {self.config.evolution.elite_ratio}")
        print(f"Current mutation rate: {self.config.evolution.mutation_rate}")
        print(f"Current generations: {self.config.evolution.generations}")
        print("└──────────────────────────────────────────────────────┘")

        pop = input("Population size [Enter to keep current]: ").strip()
        if pop:
            self.config.evolution.population_size = int(pop)

        elite = input("Elite ratio (0.0-1.0) [Enter to keep current]: ").strip()
        if elite:
            self.config.evolution.elite_ratio = float(elite)

        mut = input("Mutation rate (0.0-1.0) [Enter to keep current]: ").strip()
        if mut:
            self.config.evolution.mutation_rate = float(mut)

        gen = input("Generations [Enter to keep current]: ").strip()
        if gen:
            self.config.evolution.generations = int(gen)

        print("\n✓ Evolution configuration updated")
        input("Press Enter to continue...")

    def config_training(self):
        """Configura parametri training"""
        print("\n┌─ TRAINING CONFIGURATION ─────────────────────────────┐")
        print(f"Current mask ratio: {self.config.training.mask_ratio}")
        print(f"Current batch size: {self.config.training.batch_size}")
        print(f"Current save interval: {self.config.training.save_interval}")
        print(f"Current checkpoint dir: {self.config.training.checkpoint_dir}")
        print("└──────────────────────────────────────────────────────┘")

        mask = input("Mask ratio (0.0-1.0) [Enter to keep current]: ").strip()
        if mask:
            self.config.training.mask_ratio = float(mask)

        batch = input("Batch size [Enter to keep current]: ").strip()
        if batch:
            self.config.training.batch_size = int(batch)

        interval = input("Save interval [Enter to keep current]: ").strip()
        if interval:
            self.config.training.save_interval = int(interval)

        checkpoint = input("Checkpoint directory [Enter to keep current]: ").strip()
        if checkpoint:
            self.config.training.checkpoint_dir = checkpoint

        print("\n✓ Training configuration updated")
        input("Press Enter to continue...")

    def config_inference(self):
        """Configura parametri inferenza"""
        print("\n┌─ INFERENCE CONFIGURATION ────────────────────────────┐")
        print(f"Current max response length: {self.config.inference.max_response_length}")
        print(f"Current temperature: {self.config.inference.temperature}")
        print(f"Current context window: {self.config.inference.context_window}")
        print("└──────────────────────────────────────────────────────┘")

        max_len = input("Max response length [Enter to keep current]: ").strip()
        if max_len:
            self.config.inference.max_response_length = int(max_len)

        temp = input("Temperature (0.0-2.0) [Enter to keep current]: ").strip()
        if temp:
            self.config.inference.temperature = float(temp)

        context = input("Context window [Enter to keep current]: ").strip()
        if context:
            self.config.inference.context_window = int(context)

        print("\n✓ Inference configuration updated")
        input("Press Enter to continue...")

    def train_mode(self):
        """Modalità training"""
        self.clear_screen()
        self.print_banner()

        print("\n┌─ TRAINING MODE ──────────────────────────────────────┐")

        dataset_path = input("Dataset path (file or directory): ").strip()
        if not dataset_path or not Path(dataset_path).exists():
            print("❌ Invalid path!")
            input("Press Enter to continue...")
            return

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
            print(f"❌ Error loading dataset: {e}")
            sys.stdout.flush()
            input("Press Enter to continue...")
            return

        print(f"✓ Dataset loaded: {len(self.dataset)} chunks")
        sys.stdout.flush()

        generations = input(f"\nGenerations [{self.config.evolution.generations}]: ").strip()
        if generations:
            gen_count = int(generations)
        else:
            gen_count = self.config.evolution.generations

        print("\n🧬 Starting evolutionary training...")
        sys.stdout.flush()

        try:
            print(f"[DEBUG] Config injection check: pop={self.config.evolution.population_size}, lattice={self.config.lattice.size_per_dim}")
            sys.stdout.flush()

            self.trainer = EvolutionaryTrainer(self.config, self.dataset)

            print(f"[DEBUG] Trainer created successfully")
            sys.stdout.flush()

            self.trainer.train(generations=gen_count, verbose=True)

            print("\n✓ Training completed!")
            print(f"Best model saved to: {self.config.training.checkpoint_dir}")
            sys.stdout.flush()

        except Exception as e:
            print(f"\n❌ TRAINING FAILED!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()

        input("\nPress Enter to continue...")

    def chat_mode(self):
        """Modalità chat"""
        self.clear_screen()
        self.print_banner()

        print("\n┌─ CHAT MODE ──────────────────────────────────────────┐")

        checkpoint_dir = self.config.training.checkpoint_dir

        if not Path(checkpoint_dir).exists():
            print(f"❌ No checkpoints found in {checkpoint_dir}")
            input("Press Enter to continue...")
            return

        checkpoints = sorted(Path(checkpoint_dir).glob("*.pkl"))

        if not checkpoints:
            print(f"❌ No .pkl files found in {checkpoint_dir}")
            input("Press Enter to continue...")
            return

        print("Available models:")
        for i, cp in enumerate(checkpoints, 1):
            print(f"  {i}. {cp.name}")

        choice = input("\nSelect model number: ").strip()

        try:
            idx = int(choice) - 1
            checkpoint_path = checkpoints[idx]
        except (ValueError, IndexError):
            print("❌ Invalid selection!")
            input("Press Enter to continue...")
            return

        print(f"\n📦 Loading model: {checkpoint_path.name}")
        sys.stdout.flush()

        try:
            self.inference, loaded_config = load_trained_model(str(checkpoint_path))
            print("✓ Model loaded successfully!")
            sys.stdout.flush()
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.stdout.flush()
            input("Press Enter to continue...")
            return

        session = EVEChatSession(self.inference)

        print("\n" + "═" * 60)
        print("CHAT SESSION - Type 'exit' or 'quit' to end")
        print("═" * 60)

        while True:
            user_input = input("\n You: ").strip()

            if user_input.lower() in ['exit', 'quit', 'q']:
                break

            if not user_input:
                continue

            response = session.send_message(user_input)
            print(f" EVE: {response}")

        print("\n✓ Chat session ended")
        input("Press Enter to continue...")

    def run(self):
        """Loop principale TUI"""
        while True:
            self.clear_screen()
            self.print_banner()
            self.print_menu()

            choice = input("\nSelect option: ").strip()

            if choice == '1':
                self.train_mode()
            elif choice == '2':
                self.chat_mode()
            elif choice == '3':
                self.configure_settings()
            elif choice == '4':
                self.clear_screen()
                self.print_banner()
                print(self.config.display())
                input("\nPress Enter to continue...")
            elif choice == '5':
                path = input("Save config to [eve_config.json]: ").strip()
                if not path:
                    path = "eve_config.json"
                self.config.save(path)
                print(f"✓ Configuration saved to {path}")
                input("Press Enter to continue...")
            elif choice == '6':
                path = input("Load config from: ").strip()
                if path and Path(path).exists():
                    self.config = EVEConfig.load(path)
                    print(f"✓ Configuration loaded from {path}")
                else:
                    print("❌ File not found!")
                input("Press Enter to continue...")
            elif choice == '7':
                print("\n👋 Goodbye!")
                sys.exit(0)
            else:
                print("❌ Invalid option!")
                input("Press Enter to continue...")


if __name__ == "__main__":
    tui = EVETUI()
    tui.run()
