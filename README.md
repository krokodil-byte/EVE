# EVE - Evolutionary Intelligence Through Bit Evolution

**Un modello di AI rivoluzionario basato su evoluzione biologica invece che gradient descent.**

## 🧬 Filosofia

EVE risponde a una domanda fondamentale:

> "Se i computer a livello hardware usano SOLO operatori booleani, perché non lavorare direttamente su quel substrato invece di attraverso layer di astrazione (floating point, backprop, GPU)?"

EVE elimina:
- ❌ Floating point arithmetic
- ❌ Backpropagation
- ❌ Gradient descent
- ❌ Layer feedforward

E usa invece:
- ✅ Operatori booleani discreti (NOT, XOR, AND, shift, majority)
- ✅ Evoluzione darwiniana (mutazione + selezione)
- ✅ Propagazione spaziale attraverso lattice toroidale
- ✅ Computazione emergente da regole semplici

## 🏗️ Architettura

```
┌─────────────────────────────────────────────────────┐
│                   EVE SYSTEM                        │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────┐      ┌──────────────┐           │
│  │   Dataset    │─────>│ BitStream    │           │
│  │   Loader     │      │  Converter   │           │
│  └──────────────┘      └──────────────┘           │
│         │                      │                   │
│         v                      v                   │
│  ┌──────────────────────────────────┐             │
│  │    Lattice Map (D-dimensional)   │             │
│  │  ┌───┬───┬───┬───┬───┬───┐      │             │
│  │  │OP │OP │OP │OP │OP │OP │      │             │
│  │  ├───┼───┼───┼───┼───┼───┤      │             │
│  │  │OP │OP │OP │OP │OP │OP │      │  Operators: │
│  │  ├───┼───┼───┼───┼───┼───┤      │  - NOT      │
│  │  │OP │OP │OP │OP │OP │OP │      │  - XOR      │
│  │  └───┴───┴───┴───┴───┴───┘      │  - AND      │
│  └──────────────────────────────────┘  - SHIFT    │
│         │                               - MAJORITY │
│         v                                          │
│  ┌──────────────────────────────────┐             │
│  │   Beam Search Propagation        │             │
│  │   (max_steps, beam_width)        │             │
│  └──────────────────────────────────┘             │
│         │                                          │
│         v                                          │
│  ┌──────────────────────────────────┐             │
│  │  Evolutionary Selection          │             │
│  │  - Fitness evaluation            │             │
│  │  - Elite preservation            │             │
│  │  - Mutation & reproduction       │             │
│  └──────────────────────────────────┘             │
└─────────────────────────────────────────────────────┘
```

## 📁 Struttura Progetto

```
EVE/
├── brain.py              # Lattice map, operatori, propagazione
├── interace.py           # Translators (text/bytes ↔ bits)
├── executor.py           # Sistema evolutivo base
├── config.py             # Sistema configurazione
├── reward_system.py      # Reward predittivo + metriche
├── data_loader.py        # Caricamento dataset universale
├── train.py              # Training evolutivo
├── inference.py          # Chat e inferenza
├── tui.py                # Text User Interface
└── README.md             # Questo file
```

## 🚀 Quick Start

### 1. Avvia la TUI

```bash
python tui.py
```

### 2. Configura Settings (opzionale)

Menu: `3. Configure settings`

Imposta:
- **Lattice**: dimensioni, beam width, max steps
- **Evolution**: population size, mutation rate, generations
- **Training**: mask ratio, batch size
- **Inference**: max response length, context window

### 3. Train un Modello

Menu: `1. Train new model`

- Fornisci path a dataset (file .txt, .md, .json, .csv o directory)
- Il sistema:
  1. Carica il dataset
  2. Converte testo in bit
  3. Maschera bit random (es. 30%)
  4. Evolve popolazione di lattice per predire bit mascherati
  5. Salva checkpoint in `./checkpoints/`

**Esempio:**

```bash
Dataset path: ./my_dataset.txt
Generations [100]: 50
```

### 4. Chat con Modello Trained

Menu: `2. Chat with trained model`

- Seleziona checkpoint da `./checkpoints/`
- Inizia conversazione
- Scrivi `exit` o `quit` per terminare

## 🎯 Come Funziona il Training

### Predictive Training

1. **Carica dataset** (qualsiasi testo/dati)
2. **Converti in bit stream** (UTF-8 encoding)
3. **Maschera bit random** (es. 30%)
   ```
   Original: [1,0,1,1,0,1,0,0,1,1,...]
   Masked:   [1,0,0,0,0,1,0,0,0,0,...]
              Mask:    [0,0,1,1,0,0,0,0,1,1,...]
   ```
4. **Propaga attraverso lattice**
   - Input bits → lattice → operatori booleani → output bits
5. **Valuta fitness** (accuracy su bit mascherati)
6. **Evoluzione**
   - Preserva elite (top 25%)
   - Muta e riproduci
   - Nuova generazione

### Metriche

- **Accuracy**: % bit corretti
- **Precision**: vero positivi / (vero positivi + falsi positivi)
- **Recall**: vero positivi / (vero positivi + falsi negativi)
- **F1 Score**: media armonica precision/recall
- **Hamming Distance**: distanza bit-level

## 💡 Esempi di Uso

### Training su Codice Sorgente

```bash
# Carica tutto il codice Python da una directory
Dataset path: ./my_project/
Mask ratio: 0.25
Generations: 100
```

### Training su Letteratura

```bash
# Carica romanzo o documenti
Dataset path: ./books/
Mask ratio: 0.30
Generations: 200
```

### Training su Dati Strutturati

```bash
# JSON, CSV, etc
Dataset path: ./data.json
Mask ratio: 0.20
Generations: 150
```

### Training su Wikipedia

```bash
# Dump XML di Wikipedia (anche compressi!)
Dataset path: ./itwiki-latest-pages-articles.xml.bz2
# oppure
Dataset path: ./enwiki-latest-pages-articles.xml.gz
Mask ratio: 0.25
Generations: 500
```

**Formati supportati:**
- `.xml` - XML semplice
- `.xml.gz` - XML compresso gzip
- `.xml.bz2` - XML compresso bzip2 (formato standard Wikipedia)
- Qualsiasi file con "wiki" nel nome viene automaticamente parsato come dump Wikipedia

### Training su Dataset Conversazionali (Parquet)

```bash
# Dataset conversazionali (HuggingFace, custom)
Dataset path: ./conversations.parquet
Mask ratio: 0.30
Generations: 300
```

**Schema supportati (auto-detection):**
- `prompt` / `response` - Colonne separate user/assistant
- `question` / `answer` - Schema Q&A
- `conversation` / `messages` - Lista di turni multi-turn
- Qualsiasi schema con pattern comuni (auto-rilevato)

**Output formattato:**
```
User: Come funziona EVE?
Assistant: EVE usa evoluzione biologica invece di gradient descent.
User: Quali operatori usa?
Assistant: NOT, XOR, AND, shift, e majority gate.
```

EVE impara la struttura del dialogo a turni direttamente a livello di bit!

## ⚙️ Configurazione Avanzata

### File `eve_config.json`

```json
{
  "lattice": {
    "dimensions": 2,
    "size_per_dim": 8,
    "beam_width": 4,
    "max_steps": 16
  },
  "evolution": {
    "population_size": 32,
    "elite_ratio": 0.25,
    "mutation_rate": 0.1,
    "generations": 100
  },
  "training": {
    "mask_ratio": 0.3,
    "batch_size": 16,
    "save_interval": 10,
    "checkpoint_dir": "./checkpoints"
  },
  "inference": {
    "max_response_length": 512,
    "temperature": 1.0,
    "context_window": 1024
  }
}
```

Salva/carica con TUI (opzioni 5 e 6).

## 🔬 Approccio Scientifico

### Domande di Ricerca

1. **Può l'evoluzione competere con gradient descent?**
   - Velocità di convergenza
   - Sample efficiency
   - Generalizzazione

2. **Efficienza computazionale?**
   - Operazioni booleane vs float
   - Energia consumata
   - Throughput

3. **Espressività del substrato?**
   - Quali task può apprendere?
   - Limiti teorici
   - Scaling behavior

### Confronto con Neural Networks

| Metrica | Neural Networks | EVE |
|---------|----------------|-----|
| Operazioni base | Float multiply-add | Boolean ops |
| Ottimizzazione | Gradient descent | Evolution |
| Interpretabilità | Bassa (black box) | Alta (trace bit-by-bit) |
| Hardware | GPU (float heavy) | FPGA/ASIC (bool native) |
| Energia/op | Alta | Potenzialmente bassa |
| Training parallelization | Batch gradients | Population evaluation |

## 🎨 Estensioni Possibili

### 1. Operatori Personalizzati

Aggiungi nuovi operatori in `brain.py`:

```python
def op_custom(state, neighbors):
    # Tua logica booleana
    return new_state
```

### 2. Reward Multi-Obiettivo

Modifica `reward_system.py`:

```python
fitness = (
    0.5 * accuracy +
    0.3 * diversity +
    0.2 * efficiency
)
```

### 3. Adaptive Difficulty

Aumenta `mask_ratio` durante training se accuracy > threshold.

### 4. Hierarchical Lattices

Lattice nested per processare informazione a scale diverse.

## 🐛 Troubleshooting

### "No data loaded"

Verifica che il path del dataset sia corretto e contenga file supportati (.txt, .json, .csv, .md).

### "Training very slow"

Riduci:
- `population_size`
- `lattice.size_per_dim`
- `batch_size`

### "Output random/garbage"

Normale per modelli non trained. Serve training sostanziale (100+ generazioni) su dataset consistente.

### "Memory error"

Riduci:
- `chunk_size` in DataLoader
- `context_window` in inference
- `population_size`

## 📊 Benchmark (Da Fare)

TODO: Aggiungere benchmark su:
- [ ] Text reconstruction accuracy
- [ ] Training time vs NN equivalente
- [ ] Energy consumption
- [ ] Generalization su unseen data
- [ ] Scaling laws (population size, lattice size)

## 🤝 Contributi

Questo è un progetto di ricerca sperimentale. Idee:

1. Nuovi operatori booleani
2. Strategie evolutive avanzate (coevolution, novelty search)
3. Visualizzazione lattice durante evoluzione
4. Hardware acceleration (FPGA implementation)
5. Benchmark rigorosi vs state-of-the-art

## 📄 Licenza

[Specifica licenza]

**EVE**: L'AI che evolve invece di backpropagare.
