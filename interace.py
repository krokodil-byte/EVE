# interface.py
import abc
import numpy as np
from typing import Any, Dict, Tuple, Callable

BitArray = np.ndarray  # shape (N,), dtype=uint8


# =========================
#  TRANSLATOR BASE
# =========================

class Translator(abc.ABC):
    """
    Qualsiasi medium → bitstring e viceversa.
    """
    @abc.abstractmethod
    def encode(self, obj: Any) -> BitArray:
        ...

    @abc.abstractmethod
    def decode(self, bits: BitArray) -> Any:
        ...


class ByteTranslator(Translator):
    """
    bytes <-> bitstring.
    Puoi usare questo come mattoncino per testo, immagini, ecc.
    """
    def encode(self, data: bytes) -> BitArray:
        bits = []
        for b in data:
            for i in range(8):
                bits.append((b >> (7 - i)) & 1)
        return np.array(bits, dtype=np.uint8)

    def decode(self, bits: BitArray) -> bytes:
        bits = bits.astype(np.uint8)
        assert bits.size % 8 == 0
        out = bytearray()
        for i in range(0, bits.size, 8):
            v = 0
            for j in range(8):
                v = (v << 1) | int(bits[i + j])
            out.append(v)
        return bytes(out)


class TextTranslator(Translator):
    """
    Testo utf-8 <-> bitstring attraverso ByteTranslator.
    """
    def __init__(self):
        self.byte_tr = ByteTranslator()

    def encode(self, text: str) -> BitArray:
        return self.byte_tr.encode(text.encode("utf-8"))

    def decode(self, bits: BitArray) -> str:
        data = self.byte_tr.decode(bits)
        return data.decode("utf-8", errors="ignore")


# Placeholder per flussi (microfono, rete, ecc.)
# Qui metti solo le interfacce, l'implementazione concreta la fai su misura.
class StreamTranslator(Translator):
    """
    Esempio: wrapper per flussi (audio, sensori, rete).
    Per ora è solo un placeholder.
    """
    def encode(self, obj: Any) -> BitArray:
        raise NotImplementedError("Implementare basato sul tipo di stream.")

    def decode(self, bits: BitArray) -> Any:
        raise NotImplementedError("Dipende dal tipo di stream.")


# =========================
#  FAST REWARD CONNECTORS
# =========================

class RewardConnector(abc.ABC):
    """
    Interfaccia generica:
    prende input originali, output della mappa, contesto
    e restituisce uno scalar reward.
    """
    @abc.abstractmethod
    def __call__(
        self,
        original: Any,
        output_bits: BitArray,
        context: Dict[str, Any]
    ) -> float:
        ...


class BitReconstructionReward(RewardConnector):
    """
    Ricostruttore di file corrotti:
    - original_bits: contesto["target_bits"]
    - mask: contesto["mask"] (True sui bit nascosti)
    Reward = frazione di bit nascosti ricostruiti correttamente.
    """
    def __call__(
        self,
        original: Any,
        output_bits: BitArray,
        context: Dict[str, Any]
    ) -> float:
        target_bits: BitArray = context["target_bits"]
        mask: np.ndarray = context["mask"]

        output_bits = output_bits.astype(np.uint8)
        target_bits = target_bits.astype(np.uint8)

        if mask.sum() == 0:
            return 0.0

        pred_masked = output_bits[mask]
        target_masked = target_bits[mask]

        return float((pred_masked == target_masked).mean())


class ExternalRewardConnector(RewardConnector):
    """
    Fast connector generico:
    gli passi una funzione Python che valuta il risultato reale
    (file ricostruito, testo generato, azioni su ambiente, feedback umano, ecc.)
    """
    def __init__(self, fn: Callable[[Any, BitArray, Dict[str, Any]], float]):
        """
        fn(original, output_bits, context) -> reward
        """
        self.fn = fn

    def __call__(
        self,
        original: Any,
        output_bits: BitArray,
        context: Dict[str, Any]
    ) -> float:
        return float(self.fn(original, output_bits, context))
