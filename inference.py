"""
EVE Inference Module
Chat e inferenza con lattice trained
"""

import numpy as np
from typing import Optional, Tuple, List
import pickle

from brain import LatticeMap, propagate
from interace import TextTranslator
from config import EVEConfig


class EVEInference:
    """
    Sistema di inferenza per EVE
    Usa lattice trained per generare testo
    """

    def __init__(
        self,
        lattice_map: LatticeMap,
        config: EVEConfig,
        translator: Optional[TextTranslator] = None
    ):
        """
        Args:
            lattice_map: LatticeMap trained
            config: Configurazione
            translator: Traduttore testo-bit (default: TextTranslator)
        """
        self.map = lattice_map
        self.config = config
        self.translator = translator or TextTranslator()

        self.context_bits = np.array([], dtype=np.uint8)
        self.max_context_bits = config.inference.context_window * 8

    def reset_context(self):
        """Resetta il contesto conversazionale"""
        self.context_bits = np.array([], dtype=np.uint8)

    def add_to_context(self, text: str):
        """
        Aggiunge testo al contesto

        Args:
            text: Testo da aggiungere
        """
        new_bits = self.translator.encode(text)
        self.context_bits = np.concatenate([self.context_bits, new_bits])

        if len(self.context_bits) > self.max_context_bits:
            self.context_bits = self.context_bits[-self.max_context_bits:]

    def predict_continuation(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0
    ) -> str:
        """
        Genera continuazione di un prompt

        Args:
            prompt: Testo di input
            max_length: Lunghezza massima output in caratteri
            temperature: Controllo randomness (non implementato per ora)

        Returns:
            Testo generato
        """
        prompt_bits = self.translator.encode(prompt)

        start_coord = tuple([0] * self.map.dim)

        try:
            final_states = propagate(
                self.map,
                prompt_bits,
                start_coord,
                max_steps=self.config.lattice.max_steps,
                beam_width=self.config.lattice.beam_width
            )

            if not final_states:
                return "[Errore: nessuno stato finale]"

            output_bits = final_states[0].state

            extra_bits_needed = max_length * 8
            if len(output_bits) < extra_bits_needed:
                output_bits = np.pad(
                    output_bits,
                    (0, extra_bits_needed - len(output_bits)),
                    mode='constant'
                )

            output_text = self.translator.decode(output_bits)

            output_text = output_text[:max_length]

            return output_text

        except Exception as e:
            return f"[Errore durante inferenza: {e}]"

    def complete_masked_text(self, text: str, mask_char: str = '_') -> str:
        """
        Completa testo con caratteri mascherati

        Args:
            text: Testo con mask_char al posto di caratteri mancanti
            mask_char: Carattere che indica posizione mascherata

        Returns:
            Testo con caratteri predetti
        """
        full_bits = self.translator.encode(text.replace(mask_char, 'X'))
        mask_positions = np.array([c == mask_char for c in text], dtype=bool)

        input_bits = full_bits.copy()
        char_size = 8

        for i, is_masked in enumerate(mask_positions):
            if is_masked:
                bit_start = i * char_size
                bit_end = bit_start + char_size
                if bit_end <= len(input_bits):
                    input_bits[bit_start:bit_end] = 0

        start_coord = tuple([0] * self.map.dim)

        try:
            final_states = propagate(
                self.map,
                input_bits,
                start_coord,
                max_steps=self.config.lattice.max_steps,
                beam_width=self.config.lattice.beam_width
            )

            if not final_states:
                return text

            predicted_bits = final_states[0].state

            if len(predicted_bits) != len(full_bits):
                predicted_bits = np.resize(predicted_bits, len(full_bits))

            predicted_text = self.translator.decode(predicted_bits)

            return predicted_text[:len(text)]

        except Exception as e:
            return f"[Errore: {e}]"

    def chat(self, user_input: str, max_response_length: Optional[int] = None) -> str:
        """
        Modalità chat conversazionale

        Args:
            user_input: Input dell'utente
            max_response_length: Lunghezza massima risposta

        Returns:
            Risposta generata
        """
        if max_response_length is None:
            max_response_length = self.config.inference.max_response_length

        self.add_to_context(f"User: {user_input}\n")

        context_text = self.translator.decode(self.context_bits)
        prompt = context_text + "EVE: "

        response = self.predict_continuation(
            prompt,
            max_length=max_response_length,
            temperature=self.config.inference.temperature
        )

        self.add_to_context(f"EVE: {response}\n")

        return response


class EVEChatSession:
    """
    Sessione di chat interattiva con EVE
    """

    def __init__(self, inference: EVEInference):
        self.inference = inference
        self.history = []

    def send_message(self, message: str) -> str:
        """Invia messaggio e ottieni risposta"""
        response = self.inference.chat(message)
        self.history.append({
            'user': message,
            'eve': response
        })
        return response

    def display_history(self):
        """Mostra cronologia conversazione"""
        print("\n" + "═" * 60)
        print("CHAT HISTORY")
        print("═" * 60)
        for i, exchange in enumerate(self.history):
            print(f"\n[{i+1}] User: {exchange['user']}")
            print(f"    EVE:  {exchange['eve']}")
        print("═" * 60 + "\n")


def load_trained_model(checkpoint_path: str) -> Tuple[EVEInference, EVEConfig]:
    """
    Carica modello trained da checkpoint

    Args:
        checkpoint_path: Path al file checkpoint

    Returns:
        (EVEInference, EVEConfig)
    """
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    lattice_map = checkpoint['best_map']
    config = checkpoint['config']

    inference = EVEInference(lattice_map, config)

    return inference, config


if __name__ == "__main__":
    print("Test EVE Inference")
    print("─" * 60)

    from brain import LatticeMap
    from config import EVEConfig

    config = EVEConfig()
    test_map = LatticeMap(size=4, dim=2)

    inference = EVEInference(test_map, config)

    print("Test 1: Predict continuation")
    result = inference.predict_continuation("Hello ", max_length=20)
    print(f"Input:  'Hello '")
    print(f"Output: '{result}'")

    print("\nTest 2: Complete masked text")
    masked = "Hel__ wor__"
    result = inference.complete_masked_text(masked, mask_char='_')
    print(f"Input:  '{masked}'")
    print(f"Output: '{result}'")

    print("\nTest 3: Chat session")
    session = EVEChatSession(inference)
    response = session.send_message("Hello EVE!")
    print(f"User: Hello EVE!")
    print(f"EVE:  {response}")

    print("\n[Nota: Output sarà random senza training]")
