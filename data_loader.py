"""
EVE Data Loader
Carica dataset arbitrari e li converte in bit streams
"""

import os
import json
import csv
import gzip
import bz2
import xml.etree.ElementTree as ET
import numpy as np
from typing import List, Iterator, Optional, Tuple, Dict, Any
from pathlib import Path
from interace import TextTranslator, ByteTranslator

# Import opzionali per Parquet
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("[Warning] pandas non disponibile. Installa con: pip install pandas pyarrow")


class DataLoader:
    """
    Caricatore universale di dataset
    Supporta: .txt, .json, .csv, .md, .xml, .parquet, binari
    Supporta file compressi: .gz, .bz2
    Supporta dump Wikipedia XML
    Supporta dataset conversazionali (Parquet con colonne chat)
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

    def load_xml_file(self, path: str, text_tags: Optional[List[str]] = None) -> List[str]:
        """
        Carica file XML ed estrae testo

        Args:
            path: Path al file XML
            text_tags: Tag da cui estrarre testo (default: cerca tag comuni)

        Returns:
            Lista di chunk di testo estratti
        """
        if text_tags is None:
            # Tag comuni per Wikipedia e altri XML
            text_tags = ['text', 'p', 'content', 'body', 'page']

        chunks = []

        try:
            # Prova a parsare come XML normale
            tree = ET.parse(path)
            root = tree.getroot()

            # Estrai testo da tutti i tag specificati
            for tag_name in text_tags:
                for element in root.iter(tag_name):
                    if element.text and element.text.strip():
                        text = element.text.strip()
                        # Chunking del testo estratto
                        for i in range(0, len(text), self.chunk_size):
                            chunk = text[i:i + self.chunk_size]
                            if chunk.strip():
                                chunks.append(chunk)

        except ET.ParseError:
            # Se parsing fallisce, tratta come testo
            print(f"[Warning] XML parsing fallito per {path}, carico come testo")
            return self.load_text_file(path)

        return chunks

    def load_wikipedia_xml(self, path: str) -> List[str]:
        """
        Carica dump XML di Wikipedia
        Formato: <page><title>...</title><text>...</text></page>

        Args:
            path: Path al dump Wikipedia (.xml o .xml.gz o .xml.bz2)

        Returns:
            Lista di chunk di testo da articoli
        """
        chunks = []

        # Apri file (gestendo compressione)
        if path.endswith('.gz'):
            file_obj = gzip.open(path, 'rt', encoding='utf-8', errors='ignore')
        elif path.endswith('.bz2'):
            file_obj = bz2.open(path, 'rt', encoding='utf-8', errors='ignore')
        else:
            file_obj = open(path, 'r', encoding='utf-8', errors='ignore')

        try:
            # Parsing incrementale per file grandi
            current_page_text = None
            in_text_tag = False
            text_buffer = []

            for line in file_obj:
                # Cerca tag <text>
                if '<text' in line:
                    in_text_tag = True
                    # Estrai contenuto sulla stessa riga se presente
                    start = line.find('>') + 1
                    end = line.find('</text>')
                    if end != -1:
                        text_buffer.append(line[start:end])
                        in_text_tag = False
                    elif start > 0:
                        text_buffer.append(line[start:])
                elif '</text>' in line:
                    end = line.find('</text>')
                    text_buffer.append(line[:end])
                    in_text_tag = False

                    # Processa testo accumulato
                    full_text = ''.join(text_buffer).strip()
                    if full_text and len(full_text) > 50:
                        # Chunking
                        for i in range(0, len(full_text), self.chunk_size):
                            chunk = full_text[i:i + self.chunk_size]
                            if chunk.strip():
                                chunks.append(chunk)

                    text_buffer = []
                elif in_text_tag:
                    text_buffer.append(line)

        finally:
            file_obj.close()

        print(f"[Wikipedia] Estratti {len(chunks)} chunk da {path}")
        return chunks

    def load_compressed_file(self, path: str) -> List[str]:
        """
        Carica file compresso (.gz, .bz2)

        Args:
            path: Path al file compresso

        Returns:
            Lista di chunk
        """
        # Determina tipo compressione
        if path.endswith('.gz'):
            open_fn = gzip.open
        elif path.endswith('.bz2'):
            open_fn = bz2.open
        else:
            return self.load_text_file(path)

        # Leggi contenuto decompresso
        with open_fn(path, 'rt', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Chunking
        chunks = []
        for i in range(0, len(content), self.chunk_size):
            chunk = content[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def load_parquet_file(
        self,
        path: str,
        conversation_columns: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """
        Carica file Parquet con dataset conversazionali

        Args:
            path: Path al file .parquet
            conversation_columns: Mapping delle colonne (default: auto-detect)
                Esempi:
                - {'user': 'prompt', 'assistant': 'response'}
                - {'user': 'question', 'assistant': 'answer'}
                - {'conversation': 'messages'}  # Lista di turni

        Returns:
            Lista di chunk con conversazioni formattate
        """
        if not PANDAS_AVAILABLE:
            raise ImportError(
                "pandas/pyarrow richiesti per Parquet. "
                "Installa con: pip install pandas pyarrow"
            )

        # Carica Parquet
        df = pd.read_parquet(path)
        chunks = []

        print(f"[Parquet] Colonne disponibili: {list(df.columns)}")

        # Auto-detect schema se non specificato
        if conversation_columns is None:
            conversation_columns = self._detect_conversation_schema(df)

        # Caso 1: Colonne separate user/assistant
        if 'user' in conversation_columns and 'assistant' in conversation_columns:
            user_col = conversation_columns['user']
            assistant_col = conversation_columns['assistant']

            for _, row in df.iterrows():
                user_text = str(row[user_col]) if pd.notna(row[user_col]) else ""
                assistant_text = str(row[assistant_col]) if pd.notna(row[assistant_col]) else ""

                if user_text.strip() and assistant_text.strip():
                    # Formato conversazionale
                    conversation = f"User: {user_text}\nAssistant: {assistant_text}"

                    # Chunking se troppo lungo
                    if len(conversation) <= self.chunk_size:
                        chunks.append(conversation)
                    else:
                        for i in range(0, len(conversation), self.chunk_size):
                            chunk = conversation[i:i + self.chunk_size]
                            if chunk.strip():
                                chunks.append(chunk)

        # Caso 2: Colonna con lista di messaggi
        elif 'conversation' in conversation_columns:
            conv_col = conversation_columns['conversation']

            for _, row in df.iterrows():
                messages = row[conv_col]

                # Converti numpy array in lista se necessario
                if hasattr(messages, 'tolist'):
                    messages = messages.tolist()

                if not isinstance(messages, list):
                    continue

                # Costruisci conversazione multi-turn
                conversation_lines = []
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get('role', msg.get('from', 'unknown'))
                        content = msg.get('content', msg.get('value', ''))

                        # Normalizza role
                        if role in ['user', 'human', 'question']:
                            role = 'User'
                        elif role in ['assistant', 'gpt', 'answer', 'bot']:
                            role = 'Assistant'
                        elif role == 'system':
                            role = 'System'

                        conversation_lines.append(f"{role}: {content}")

                if conversation_lines:
                    full_conversation = '\n'.join(conversation_lines)

                    # Chunking
                    if len(full_conversation) <= self.chunk_size:
                        chunks.append(full_conversation)
                    else:
                        for i in range(0, len(full_conversation), self.chunk_size):
                            chunk = full_conversation[i:i + self.chunk_size]
                            if chunk.strip():
                                chunks.append(chunk)

        # Caso 3: Colonna singola di testo
        elif 'text' in conversation_columns:
            text_col = conversation_columns['text']

            for _, row in df.iterrows():
                text = str(row[text_col]) if pd.notna(row[text_col]) else ""

                if text.strip():
                    # Chunking
                    for i in range(0, len(text), self.chunk_size):
                        chunk = text[i:i + self.chunk_size]
                        if chunk.strip():
                            chunks.append(chunk)

        print(f"[Parquet] Estratti {len(chunks)} chunk conversazionali")
        return chunks

    def _detect_conversation_schema(self, df: 'pd.DataFrame') -> Dict[str, str]:
        """
        Auto-detect schema colonne conversazionali

        Returns:
            Mapping colonne rilevate
        """
        cols = [c.lower() for c in df.columns]

        # Pattern comuni per colonne user/assistant
        user_patterns = ['prompt', 'user', 'question', 'input', 'human', 'instruction']
        assistant_patterns = ['response', 'assistant', 'answer', 'output', 'gpt', 'completion']

        user_col = None
        assistant_col = None

        for pattern in user_patterns:
            for col in df.columns:
                if pattern in col.lower():
                    user_col = col
                    break
            if user_col:
                break

        for pattern in assistant_patterns:
            for col in df.columns:
                if pattern in col.lower():
                    assistant_col = col
                    break
            if assistant_col:
                break

        if user_col and assistant_col:
            print(f"[Parquet] Schema rilevato: user='{user_col}', assistant='{assistant_col}'")
            return {'user': user_col, 'assistant': assistant_col}

        # Cerca colonna conversazione/messaggi
        conv_patterns = ['conversation', 'messages', 'dialog', 'chat']
        for pattern in conv_patterns:
            for col in df.columns:
                if pattern in col.lower():
                    print(f"[Parquet] Schema rilevato: conversation='{col}'")
                    return {'conversation': col}

        # Fallback: cerca colonna "text"
        for col in df.columns:
            if 'text' in col.lower():
                print(f"[Parquet] Schema rilevato: text='{col}'")
                return {'text': col}

        # Default: usa prima colonna
        print(f"[Parquet] Schema default: text='{df.columns[0]}'")
        return {'text': df.columns[0]}

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
            extensions = ['.txt', '.md', '.json', '.csv', '.xml', '.parquet', '.gz', '.bz2']

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

        # Gestisci estensioni composte (es. .xml.gz, .xml.bz2)
        full_name = path_obj.name.lower()

        # Dump Wikipedia (rileva pattern comuni)
        if 'wiki' in full_name and ('.xml' in full_name):
            return self.load_wikipedia_xml(path)

        # File compressi
        if full_name.endswith('.gz') or full_name.endswith('.bz2'):
            # Se è XML compresso, usa parser Wikipedia
            if '.xml' in full_name:
                return self.load_wikipedia_xml(path)
            else:
                return self.load_compressed_file(path)

        # Estensione singola
        ext = path_obj.suffix.lower()

        if ext == '.xml':
            # Prova prima come Wikipedia, poi XML generico
            try:
                return self.load_wikipedia_xml(path)
            except:
                return self.load_xml_file(path)
        elif ext == '.parquet':
            return self.load_parquet_file(path)
        elif ext in ['.txt', '.md', '.py', '.js', '.html', '.css', '.c', '.cpp', '.java']:
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
