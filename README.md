# Frame Generator

Progetto per il laboratorio di Parallel Computing. Il programma esegue la generazione di frame intermedi usando 2 algoritmi:
LERP: P(out) = (1 - t) * P1 + t * P2, ha bisogno solo di due frames
BICUBIC:

    Genera uno o più frame intermedi `I_τ` (con `τ ∈ [0,1]`, tra `t₀` e `t₁`) a partire da 4 frame `I₋₁, I₀, I₁, I₂`.

    - **Input**: i 4 frame, i vettori di moto `d₋₁, d₁, d₂` (spostamento di ogni punto di `I₀` verso `I₋₁`, `I₁`, `I₂`; `d₀ = 0`) e i valori di `τ`.
    - **Interpolazione cubica temporale**: per ogni pixel di output, la cubica Catmull-Rom applicata ai vettori (componenti x e y) dà lo spostamento al tempo `τ`: `D(τ) = C(d₋₁, 0, d₁, d₂; τ)`.
    - **Backward warping**: calcolo delle coordinate da leggere nei due frame sorgente, in generale frazionarie:
    - `(x₀', y₀') = (x, y) − D(τ)` in `I₀`
    - `(x₁', y₁') = (x, y) − D(τ) + d₁` in `I₁`
    - **Interpolazione bicubica spaziale**: il colore in `(x', y')` si ottiene dall'intorno 4×4 (16 pixel) con 4 cubiche lungo x e 1 lungo y. I bordi si gestiscono con clamp, e il risultato va limitato all'intervallo valido.
    - **Blending**: `I_τ(x, y) = (1 − τ)·S₀ + τ·S₁`, dove `S₀` e `S₁` sono i colori campionati in `I₀` e `I₁`.

    **Cubica Catmull-Rom** (`t ∈ [0,1]` tra `q₁` e `q₂`):
    `C(t) = ½·[ 2q₁ + (−q₀+q₂)t + (2q₀−5q₁+4q₂−q₃)t² + (−q₀+3q₁−3q₂+q₃)t³ ]`


Inoltre, per calcolare il movimento spaziale degli oggetti prima di applicare l'algoritmo, viene usato 
il block matching con metrica SAD per calcolare il matching tra i blocchi.
L'obiettivo del laboratorio è implementare la versione sequenziale e parallela con CUDA e effettuare dei benchmark per confrontare le due versioni.

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
