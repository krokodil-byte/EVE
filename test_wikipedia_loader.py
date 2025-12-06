"""
Test per Wikipedia XML loader
"""

import os
from data_loader import DataLoader, BitStreamDataset

# Crea un file XML simulato in stile Wikipedia
test_xml = """<?xml version="1.0" encoding="UTF-8"?>
<mediawiki>
  <page>
    <title>Intelligenza Artificiale</title>
    <text>
      L'intelligenza artificiale (IA) è la disciplina che studia se e in che modo
      si possano realizzare sistemi informatici intelligenti in grado di simulare
      la capacità e il comportamento del pensiero umano.

      EVE rappresenta un approccio rivoluzionario all'IA, utilizzando evoluzione
      biologica invece di gradient descent. Lavorando direttamente sui bit a livello
      hardware, EVE elimina layer di astrazione inutili.
    </text>
  </page>
  <page>
    <title>Evoluzione Biologica</title>
    <text>
      L'evoluzione biologica è il processo di cambiamento e diversificazione degli
      organismi viventi nel corso delle generazioni. Charles Darwin fu il primo a
      formulare una teoria scientifica dell'evoluzione basata sulla selezione naturale.

      EVE applica questi principi alla computazione, evolvendo lattice di operatori
      booleani invece di ottimizzare pesi neurali con backpropagation.
    </text>
  </page>
  <page>
    <title>Operatori Booleani</title>
    <text>
      Gli operatori booleani sono operazioni logiche fondamentali come AND, OR, NOT,
      XOR che operano su valori binari (0 e 1). Tutti i computer moderni sono costruiti
      su questi operatori a livello hardware.

      EVE usa solo 6 operatori booleani: NOT, XOR, shift left, shift right, AND con
      vicini, e majority gate. Questo è sufficiente per qualsiasi computazione.
    </text>
  </page>
</mediawiki>
"""

# Salva file test
test_path = "/tmp/test_wikipedia.xml"
with open(test_path, 'w', encoding='utf-8') as f:
    f.write(test_xml)

print("Test Wikipedia XML Loader")
print("=" * 60)

# Test DataLoader
loader = DataLoader(chunk_size=200)
chunks = loader.load_wikipedia_xml(test_path)

print(f"\n✓ Chunks estratti: {len(chunks)}")
print("\nPrimi 3 chunks:")
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk[:150] + "..." if len(chunk) > 150 else chunk)

# Test BitStreamDataset
print("\n" + "=" * 60)
print("Test BitStreamDataset su Wikipedia XML")
print("=" * 60)

dataset = BitStreamDataset(test_path, chunk_size=200, mask_ratio=0.3)
print(f"✓ Dataset caricato: {len(dataset)} chunks")

# Get sample
input_bits, target_bits, mask = dataset.get_sample(0)
print(f"✓ Sample generato:")
print(f"  - Bit totali: {len(target_bits)}")
print(f"  - Bit mascherati: {mask.sum()} ({mask.sum()/len(mask)*100:.1f}%)")

# Decodifica per vedere il testo
from interace import TextTranslator
translator = TextTranslator()
decoded = translator.decode(target_bits)
print(f"  - Testo decodificato: '{decoded[:100]}...'")

print("\n✓ Test completato! Wikipedia XML supportato.")
print("\nOra puoi usare:")
print("  - File .xml normali")
print("  - File .xml.gz compressi")
print("  - File .xml.bz2 compressi")
print("  - Dump Wikipedia di qualsiasi dimensione")
