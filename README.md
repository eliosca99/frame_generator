# Frame Generator

Progetto per il laboratorio di Parallel Computing. Il programma esegue la generazione di frame intermedi usando 3 algoritmi:
LERP: P(out) = (1 - t) * P1 + t * P2
L'obiettivo del laboratorio è implementare la versione sequenziale e parallela con CUDA e effettuare dei benchmark per
confrontare le due versioni.

## 📋 Indice
- [Prerequisiti](#-prerequisiti)
- [Compilazione](#-compilazione)
- [Utilizzo](#-utilizzo)
- [Struttura del Progetto](#-struttura-del-progetto)
- [Confronto Prestazionale](#-confronto-prestazionale)
- [Autori](#%EF%B8%8F-autori)

## 🛠️ Prerequisiti

Prima di iniziare, assicurati di aver installato i seguenti strumenti:
- Compilatore C++ compatibile con lo standard richiesto (es. GCC, Clang)
- [CMake](https://cmake.org/) (versione X.X o superiore)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (necessario per la compilazione dei moduli in `src/parallel/`)
- Libreria `libpng` (per il caricamento e salvataggio dei frame)

## 🚀 Compilazione

```bash
# Entra nella directory del progetto
cd frame_generator

# Crea una directory di build e spostati al suo interno
mkdir build
cd build

# Configura il progetto tramite CMake
cmake ..

# Compila l'eseguibile
make
```

## 💻 Utilizzo

```bash
# Esempio di esecuzione
./frame_generator [opzioni...] <percorso_input> <percorso_output>
```

**Opzioni disponibili:**
* `-m sequential|parallel`: Sceglie l'implementazione da utilizzare.
* *(Aggiungi altre opzioni se necessarie...)*

## 📁 Struttura del Progetto

```text
frame_generator/
├── CMakeLists.txt        # Configurazione per la build con CMake
├── README.md             # Questo file
├── include/              # File di intestazione (.hpp, .cuh)
│   ├── parallel/         # Header per le funzioni CUDA
│   ├── sequential/       # Header per le funzioni CPU
│   └── utils/            # Utility varie (es. caricamento PNG)
├── src/                  # Codice sorgente (.cpp, .cu)
│   ├── main.cpp          # Entry-point dell'applicazione
│   ├── parallel/         # Moduli per l'elaborazione parallela su GPU
│   ├── sequential/       # Moduli per l'elaborazione sequenziale
│   └── utils/            # Implementazione delle utilities
└── input/                # Cartella per immagini/dati di input
```


## ✍️ Autori

* Elio Scaramuzzino - (https://github.com/eliosca99)
