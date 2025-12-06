"""
EVE Data Loader
Carica dataset arbitrari e li converte in bit streams
"""

import os
import json
import csv
import numpy as np
from typing import List, Iterator, Optional, Tuple
from pathlib import Path
from interace import TextTranslator, ByteTranslator


class DataLoader:
    """
    Caricatore universale di dataset
    Supporta: .txt, .json, .csv, .md, binari
    """

    def __init__(self, chunk_size: int = 512):
        """
        Args:
            chunk_size: Dimensione dei chunk in caratteri/byte
        """
        self.chunk_size = chunk_size
        self.text_translator = TextTranslator()
        self.byte_translator = ByteTranslator()

    def load_text_file(self, path: str) -> List[str]:
        """Carica file di testo e divide in chunk"""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        chunks = []
        for i in range(0, len(content), self.chunk_size):
            chunk = content[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def load_json_file(self, path: str) -> List[str]:
        """Carica file JSON e converte in stringhe"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        chunks = []

        def process_value(val):
            """Processa ricorsivamente valori JSON"""
            if isinstance(val, dict):
                for v in val.values():
                    process_value(v)
            elif isinstance(val, list):
                for v in val:
                    process_value(v)
            elif isinstance(val, str):
                chunks.append(val)
            else:
                chunks.append(str(val))

        process_value(data)
        return [c for c in chunks if len(c) > 10]

    def load_csv_file(self, path: str) -> List[str]:
        """Carica file CSV e concatena righe"""
        chunks = []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            for row in reader:
                row_text = ' '.join(row)
                if row_text.strip():
                    chunks.append(row_text)

        return chunks

    def load_binary_file(self, path: str) -> List[bytes]:
        """Carica file binario in chunk"""
        chunks = []
        with open(path, 'rb') as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                chunks.append(chunk)

        return chunks

    def load_directory(self, path: str, extensions: Optional[List[str]] = None) -> List[str]:
        """
        Carica tutti i file da una directory

        Args:
            path: Path della directory
            extensions: Lista di estensioni da includere (es. ['.txt', '.md'])
        """
        all_chunks = []
        path_obj = Path(path)

        if extensions is None:
            extensions = ['.txt', '.md', '.json', '.csv']

        for file_path in path_obj.rglob('*'):
            if file_path.is_file() and file_path.suffix in extensions:
                try:
                    chunks = self.load(str(file_path))
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"[Warning] Impossibile caricare {file_path}: {e}")

        return all_chunks

    def load(self, path: str) -> List[str]:
        """
        Auto-detect tipo file e carica appropriatamente

        Args:
            path: Path del file o directory

        Returns:
            Lista di chunk (stringhe)
        """
        path_obj = Path(path)

        if path_obj.is_dir():
            return self.load_directory(path)

        ext = path_obj.suffix.lower()

        if ext in ['.txt', '.md', '.py', '.js', '.html', '.css', '.c', '.cpp', '.java']:
            return self.load_text_file(path)
        elif ext == '.json':
            return self.load_json_file(path)
        elif ext == '.csv':
            return self.load_csv_file(path)
        else:
            # Prova come testo, altrimenti binario
            try:
                return self.load_text_file(path)
            except:
                binary_chunks = self.load_binary_file(path)
                return [chunk.decode('utf-8', errors='ignore') for chunk in binary_chunks]


class BitStreamDataset:
    """
    Dataset che converte dati in bit stream e genera batch
    """

    def __init__(
        self,
        data_source: str,
        chunk_size: int = 512,
        mask_ratio: float = 0.3,
        seed: Optional[int] = None
    ):
        """
        Args:
            data_source: Path a file o directory
            chunk_size: Dimensione chunk in caratteri
            mask_ratio: Percentuale di bit da mascherare
            seed: Seed per riproducibilità
        """
        self.data_source = data_source
        self.mask_ratio = mask_ratio
        self.loader = DataLoader(chunk_size=chunk_size)
        self.translator = TextTranslator()
        self.rng = np.random.RandomState(seed)

        print(f"[DataLoader] Caricamento da: {data_source}")
        self.chunks = self.loader.load(data_source)
        print(f"[DataLoader] Caricati {len(self.chunks)} chunk")

        if not self.chunks:
            raise ValueError(f"Nessun dato trovato in {data_source}")

    def __len__(self) -> int:
        return len(self.chunks)

    def get_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Ottiene un sample con bit mascherati

        Args:
            idx: Indice del chunk

        Returns:
            (input_bits, target_bits, mask_positions)
            - input_bits: bit con alcuni mascherati (settati a 0)
            - target_bits: bit originali completi
            - mask_positions: array booleano delle posizioni mascherate
        """
        chunk = self.chunks[idx % len(self.chunks)]

        target_bits = self.translator.encode(chunk)

        n_bits = len(target_bits)
        n_masked = int(n_bits * self.mask_ratio)

        mask_positions = np.zeros(n_bits, dtype=bool)
        masked_indices = self.rng.choice(n_bits, size=n_masked, replace=False)
        mask_positions[masked_indices] = True

        input_bits = target_bits.copy()
        input_bits[mask_positions] = 0

        return input_bits, target_bits, mask_positions

    def get_batch(self, batch_size: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Genera batch random di sample

        Args:
            batch_size: Numero di sample

        Returns:
            Lista di (input_bits, target_bits, mask_positions)
        """
        indices = self.rng.choice(len(self.chunks), size=batch_size, replace=True)
        return [self.get_sample(idx) for idx in indices]

    def iterate_batches(self, batch_size: int) -> Iterator[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """
        Iteratore infinito di batch
        """
        while True:
            yield self.get_batch(batch_size)


if __name__ == "__main__":
    # Test data loader
    print("Test DataLoader:")

    test_data = "Questo è un test per EVE. L'intelligenza artificiale del futuro!"

    with open("/tmp/test.txt", "w") as f:
        f.write(test_data * 10)

    dataset = BitStreamDataset(
        data_source="/tmp/test.txt",
        chunk_size=50,
        mask_ratio=0.3,
        seed=42
    )

    print(f"\nDataset size: {len(dataset)} chunks")

    input_bits, target_bits, mask = dataset.get_sample(0)
    print(f"\nSample 0:")
    print(f"  Target bits length: {len(target_bits)}")
    print(f"  Masked positions: {mask.sum()} / {len(mask)}")
    print(f"  Target: {target_bits[:50]}")
    print(f"  Input:  {input_bits[:50]}")
    print(f"  Mask:   {mask[:50].astype(int)}")

    batch = dataset.get_batch(3)
    print(f"\nBatch of 3 samples loaded successfully")
