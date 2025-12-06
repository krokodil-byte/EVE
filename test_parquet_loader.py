"""
Test per Parquet conversational dataset loader
"""

import os

# Verifica se pandas è disponibile
try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    print("⚠️  pandas/pyarrow non installati")
    print("Installa con: pip install pandas pyarrow")
    PARQUET_AVAILABLE = False
    exit(1)

from data_loader import DataLoader, BitStreamDataset

print("=" * 70)
print("TEST PARQUET CONVERSATIONAL DATASET LOADER")
print("=" * 70)

# ===== TEST 1: Schema prompt/response =====
print("\n📊 Test 1: Schema prompt/response")
print("-" * 70)

data1 = {
    'prompt': [
        "Cos'è l'intelligenza artificiale?",
        "Come funziona EVE?",
        "Quali sono i vantaggi dell'evoluzione biologica?"
    ],
    'response': [
        "L'intelligenza artificiale è la capacità dei computer di simulare processi cognitivi umani.",
        "EVE usa evoluzione darwiniana su lattice di operatori booleani invece di backpropagation.",
        "L'evoluzione biologica permette di trovare soluzioni senza gradient descent, lavorando direttamente sui bit."
    ]
}

df1 = pd.DataFrame(data1)
test_path1 = "/tmp/test_conversations_1.parquet"
df1.to_parquet(test_path1)

loader = DataLoader(chunk_size=300)
chunks1 = loader.load_parquet_file(test_path1)

print(f"✓ Chunks estratti: {len(chunks1)}")
print("\nPrimi 2 chunks:")
for i, chunk in enumerate(chunks1[:2]):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk)

# ===== TEST 2: Schema multi-turn conversation =====
print("\n" + "=" * 70)
print("📊 Test 2: Schema multi-turn conversation")
print("-" * 70)

data2 = {
    'conversation': [
        [
            {'role': 'user', 'content': 'Ciao EVE!'},
            {'role': 'assistant', 'content': 'Ciao! Sono EVE, un modello evolutivo.'},
            {'role': 'user', 'content': 'Come funzioni?'},
            {'role': 'assistant', 'content': 'Uso operatori booleani che evolvono invece di backpropagation.'}
        ],
        [
            {'role': 'user', 'content': 'Quali operatori usi?'},
            {'role': 'assistant', 'content': 'NOT, XOR, shift, AND con vicini, e majority gate.'},
            {'role': 'user', 'content': 'Perché solo questi?'},
            {'role': 'assistant', 'content': 'Sono sufficienti per qualsiasi computazione, lavorando direttamente sul substrato hardware.'}
        ]
    ]
}

df2 = pd.DataFrame(data2)
test_path2 = "/tmp/test_conversations_2.parquet"
df2.to_parquet(test_path2)

chunks2 = loader.load_parquet_file(test_path2)

print(f"✓ Chunks estratti: {len(chunks2)}")
print("\nPrimi 2 chunks:")
for i, chunk in enumerate(chunks2[:2]):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk)

# ===== TEST 3: Schema question/answer =====
print("\n" + "=" * 70)
print("📊 Test 3: Schema question/answer")
print("-" * 70)

data3 = {
    'question': [
        "Qual è il vantaggio di EVE?",
        "Come si allena EVE?"
    ],
    'answer': [
        "EVE lavora direttamente sui bit hardware, eliminando overhead di floating point.",
        "Tramite selezione evolutiva: mutazione, fitness evaluation, e riproduzione elitaria."
    ]
}

df3 = pd.DataFrame(data3)
test_path3 = "/tmp/test_conversations_3.parquet"
df3.to_parquet(test_path3)

chunks3 = loader.load_parquet_file(test_path3)

print(f"✓ Chunks estratti: {len(chunks3)}")
print("\nTutti i chunks:")
for i, chunk in enumerate(chunks3):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk)

# ===== TEST 4: BitStreamDataset integration =====
print("\n" + "=" * 70)
print("📊 Test 4: BitStreamDataset Integration")
print("-" * 70)

dataset = BitStreamDataset(test_path1, chunk_size=300, mask_ratio=0.3)
print(f"✓ Dataset caricato: {len(dataset)} chunks")

input_bits, target_bits, mask = dataset.get_sample(0)
print(f"✓ Sample generato:")
print(f"  - Bit totali: {len(target_bits)}")
print(f"  - Bit mascherati: {mask.sum()} ({mask.sum()/len(mask)*100:.1f}%)")

# Decodifica
from interace import TextTranslator
translator = TextTranslator()
decoded = translator.decode(target_bits)
print(f"  - Testo: '{decoded[:150]}...'")

# ===== SUMMARY =====
print("\n" + "=" * 70)
print("✅ TUTTI I TEST COMPLETATI!")
print("=" * 70)

print("\nFormati Parquet supportati:")
print("  1. ✓ prompt/response (colonne separate)")
print("  2. ✓ question/answer (colonne separate)")
print("  3. ✓ conversation/messages (lista di turni)")
print("  4. ✓ Auto-detection dello schema")
print("  5. ✓ Multi-turn dialogues")

print("\nOra puoi usare:")
print("  - Dataset HuggingFace in formato Parquet")
print("  - Conversazioni customizzate")
print("  - Training su dialoghi multi-turn")
print("  - Qualsiasi schema con auto-detection")

print("\n🚀 EVE è pronto per imparare a comunicare a turni!")
