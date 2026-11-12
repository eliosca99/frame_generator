# Frame Generator — Motion-Compensated Frame Interpolation (Sequential vs. CUDA)

Progetto sviluppato per il laboratorio universitario di **Parallel Computing** (3 CFU).  
L'obiettivo centrale del progetto non è la sola validità algoritmica, bensì la **valutazione comparativa, l'analisi prestazionale e lo studio dei limiti di scalabilità** delle versioni parallele e ottimizzate su architettura **NVIDIA CUDA** rispetto a una baseline puramente sequenziale su CPU (Single-Core C++17).

---

## 📋 Indice
- [Panoramica Algoritmica](#-panoramica-algoritmica)
  - [1. LERP (Linear Interpolation)](#1-lerp-linear-interpolation)
  - [2. Block Matching (Motion Estimation)](#2-block-matching-motion-estimation)
  - [3. Bicubic Motion-Compensated Interpolation](#3-bicubic-motion-compensated-interpolation)
- [Varianti di Ottimizzazione CUDA](#-varianti-di-ottimizzazione-cuda)
  - [Varianti Block Matching](#varianti-block-matching)
  - [Varianti Bicubic Interpolation](#varianti-bicubic-interpolation)
- [Prerequisiti di Sistema](#%EF%B8%8F-prerequisiti-di-sistema)
- [Compilazione](#-compilazione)
- [Esecuzione e Benchmark](#-esecuzione-e-benchmark)
- [Struttura della Repository](#-struttura-della-repository)
- [Risultati dei Benchmark](#-risultati-dei-benchmark)
- [Autori](#%EF%B8%8F-autori)

---

## 🔬 Panoramica Algoritmica

Il framework implementa due differenti strategie per la generazione di $N$ fotogrammi intermedi uniformemente distribuiti nell'intervallo temporale $\tau \in (0, 1)$:

### 1. LERP (Linear Interpolation)
Operazione di blending lineare punto a punto tra due frame adiacenti ($F_1$ ed $F_2$):
$$P_{out}(x, y) = (1 - \tau) \cdot P_1(x, y) + \tau \cdot P_2(x, y)$$
* Algoritmo a bassissima intensità aritmetica ($FLOP/Byte$), puramente **memory-bound**.
* Utilizzato come baseline per quantificare il throughput effettivo della memoria e l'overhead di lancio dei kernel su GPU.

### 2. Block Matching (Motion Estimation)
Per compensare il movimento prima dell'interpolazione, il frame di riferimento $F_1$ viene suddiviso in macroblocchi di dimensione $B \times B$. Per ciascun blocco viene eseguita una ricerca all'interno di una finestra di raggio $W$ sul frame target $F_2$, minimizzando la metrica **SAD** (*Sum of Absolute Differences*):
$$SAD(dx, dy) = \sum_{i=0}^{B-1} \sum_{j=0}^{B-1} \sum_{c \in \{R,G,B\}} |F_1(x+j, y+i)[c] - F_2(x+dx+j, y+dy+i)[c]|$$
con $dx, dy \in [-W, W]$.
* La ricerca seleziona il vettore $(dx, dy)$ che minimizza la SAD; a parità di errore minimo, viene assegnata priorità al vettore nullo $(0,0)$.

### 3. Bicubic Motion-Compensated Interpolation
Genera frame intermedi ad alta qualità interpolando tra 4 fotogrammi temporali consecutivi ($F_0, F_1, F_2, F_3$) stimando il movimento tra $F_1$ ed $F_2$:
1. **Campi di moto**: Calcolo dei vettori di spostamento $d_0, d_2, d_3$ di ogni macroblocco da $F_1$ rispettivamente verso $F_0, F_2, F_3$ tramite Block Matching.
2. **Cubica temporale (Spline Catmull-Rom)**: Per ogni blocco al tempo $\tau$, lo spostamento continuo $D(\tau) = (D_x, D_y)$ è dato da:
   $$D(\tau) = C(d_0, 0, d_2, d_3; \tau) = w_0(\tau) \cdot d_0 + w_2(\tau) \cdot d_2 + w_3(\tau) \cdot d_3$$
   dove:
   $$w_0(\tau) = \frac{1}{2}(-\tau^3 + 2\tau^2 - \tau)$$
   $$w_2(\tau) = \frac{1}{2}(-3\tau^3 + 4\tau^2 + \tau)$$
   $$w_3(\tau) = \frac{1}{2}(\tau^3 - \tau^2)$$
3. **Backward Warping**: Per ogni pixel $(x, y)$, calcolo delle coordinate di campionamento (in generale sub-pixel frazionarie):
   $$(x_1', y_1') = (x, y) - D(\tau) \quad \text{in } F_1$$
   $$(x_2', y_2') = (x, y) - D(\tau) + d_2 \quad \text{in } F_2$$
4. **Interpolazione Spaziale Bicubica**: Ricostruzione del colore nei punti frazionari campionando l'intorno $4 \times 4$ (16 pixel) con kernel Catmull-Rom separabile (4 interpolazioni 1D orizzontali e 1 verticale) e clamping sui bordi.
5. **Alpha Blending finale**:
   $$F_\tau(x, y) = (1 - \tau) \cdot S_1(x_1', y_1') + \tau \cdot S_2(x_2', y_2')$$

---

## ⚡ Varianti di Ottimizzazione CUDA

Il cuore della sperimentazione è analizzare l'impatto della gerarchia di memoria (Global Memory, L1/L2 Cache, Shared Memory, Registri e Shuffle intra-warp) sulle prestazioni.

### Varianti Block Matching
* **`GlobalOnly`**: Tutti i pixel di $F_1$ e i candidati in $F_2$ vengono letti direttamente dalla memoria globale. I thread di un blocco CUDA si dividono i candidati $(dx, dy)$ e usano una riduzione con `atomicMin` a 64 bit (SAD compattato con priorità). Elevato traffico su cache L2 e Global Memory.
* **`SharedF1`**: Il macroblocco di riferimento di $F_1$ ($B \times B$) viene caricato cooperativamente nella memoria condivisa (`__shared__`) del blocco CUDA. Si eliminano le letture ridondanti di $F_1$ per tutti i candidati della search window.
* **`SharedF1F2`**: Sia il blocco di $F_1$ sia l'intera area di ricerca di $F_2$ ($(B + 2W) \times (B + 2W)$) vengono caricati in Shared Memory. Azzeramento completo degli accessi non coalescenti in RAM globale durante il calcolo SAD. Rappresenta la configurazione a prestazioni massime (speedup fino a $>90\times$).
* **`WarpPerCandidate`**: Parallelizzazione a grana fine in cui un intero warp (32 thread) valuta cooperativamente lo stesso candidato $(dx, dy)$ distribuendosi i pixel del blocco. La somma parziale del SAD viene ridotta a livello di registri tramite istruzioni `__shfl_xor_sync`, riducendo al minimo la contesa atomica.

### Varianti Bicubic Interpolation
* **`Baseline`**: Un thread CUDA per pixel di output. Ciascun thread calcola autonomamente la cubica temporale, i pesi spaziali frazionari, legge i 16 pixel da memoria globale per entrambi i frame e calcola il blend.
* **`Precomputed`**: Un primo kernel leggero (1 thread per macroblocco) precalcola gli offset interi e i pesi frazionari $4 \times \text{float}$ per l'intera griglia, memorizzandoli in un buffer compatto allineato a 128 bit (`float4`). Il kernel di rendering legge i pesi già pronti, riducendo la pressione sulla ALU.
* **`SharedTile`**: Un blocco CUDA è associato a un macroblocco dell'immagine. I thread caricano cooperativamente in Shared Memory la porzione di frame sorgente necessaria (comprensiva dell'halo di sicurezza di $+3$ pixel per il supporto bicubico $4 \times 4$), eliminando la latenza di lettura della Global Memory nel campionamento bicubico.

---

## 🛠️ Prerequisiti di Sistema

* **Compilatore C++**: GCC 12 (`g++-12`) o superiore con pieno supporto C++17.
* **Build System**: CMake (versione $\ge 3.10$).
* **GPU Computing**: NVIDIA CUDA Toolkit ($\ge 11.8$) e driver compatibili.
* **Librerie di sistema**:
  * `libpng` e relativi header (`libpng-dev`)
  * `zlib` (`zlib1g-dev`)

Su distribuzioni Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install cmake gcc-12 g++-12 libpng-dev zlib1g-dev nvidia-cuda-toolkit
```

---

## 🚀 Compilazione

Il progetto utilizza CMake per orchestrare la compilazione dei moduli C++ e dei moduli CUDA (`nvcc`):

```bash
# Clone e posizionamento nella directory del progetto
cd frame_generator

# Creazione della cartella di build
mkdir build && cd build

# Configurazione del progetto (Release mode abilitata)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Compilazione parallela di tutti i target
make -j$(nproc)
```

La compilazione produce due eseguibili in `build/`:
* `frame_generator`: Il benchmarker principale che esegue il confronto sequenziale vs CUDA su tutto lo spazio dei parametri e salva i risultati in formato CSV.
* `test_main`: Eseguibile stand-alone per la sola validazione sequenziale e il salvataggio effettivo dei frame PNG intermedi su disco.

---

## 💻 Esecuzione e Benchmark

### 1. Esecuzione Benchmark Completo (`frame_generator`)
L'eseguibile accetta diversi flag per personalizzare l'esecuzione e limitare i test a specifici algoritmi:

```bash
./frame_generator [OPZIONI]
```

**Opzioni CLI disponibili:**
* `--input <DIR>`: Directory contenente i 4 frame di input (`frame_0.png`, `frame_1.png`, `frame_2.png`, `frame_3.png`). Default: `input/` o `../input/`.
* `--csv <FILE>`: Percorso del file CSV su cui esportare i dati di benchmark. Default: `../benchmark_results.csv`.
* `--reps <N>`: Numero di ripetizioni per ciascuna configurazione CUDA (default: `10`).
* `--reps-seq <N>`: Numero di ripetizioni per ciascuna configurazione sequenziale CPU (default: `3`).
* `--quick`: Modalità rapida con sottogruppo ridotto di parametri (utile per collaudo veloce).
* `--no-lerp`: Disabilita i test di LERP.
* `--no-bm`: Disabilita i test di Block Matching standalone.
* `--no-bicubic`: Disabilita i test dell'algoritmo Bicubic completo.

**Esempio di esecuzione completa:**
```bash
./frame_generator --input ../input --csv ../benchmark_results.csv --reps 10 --reps-seq 3
```

### 2. Generazione e Salvataggio Immagini (`test_main`)
Esegue la pipeline sequenziale su CPU e scrive le sequenze di immagini interpolate direttamente nella directory `output/`:
```bash
./test_main
```

---

## 📁 Struttura della Repository

```text
frame_generator/
├── CMakeLists.txt                    # Regole di compilazione CMake (C++17 + CUDA)
├── README.md                         # Documentazione tecnica del repository
├── benchmark_results.csv             # Risultati numerici dei benchmark (tempi, speedup, errori)
├── include/
│   ├── frame.hpp                     # Definizioni struct Pixel, Frame e FrameSequence
│   ├── parallel/
│   │   ├── block_matching_cuda.hpp   # Interfacce kernel CUDA per motion estimation
│   │   ├── cuda_utils.cuh            # Macro CUDA_CHECK, DeviceBuffer RAII e GpuTimer
│   │   └── interpolation_cuda.hpp    # Interfacce kernel CUDA per LERP e Bicubic
│   ├── sequential/
│   │   ├── block_matching.hpp        # Dichiarazione Block Matching CPU sequenziale
│   │   └── interpolation.hpp         # Dichiarazione LERP e Bicubic sequenziali
│   └── utils/
│       ├── bicubic_math.hpp          # Matrici Catmull-Rom e funzioni inline __device__
│       ├── png_utils.hpp             # Wrapper I/O basato su libpng
│       └── utils.hpp                 # Struttura int2 e utility di calcolo
├── src/
│   ├── main.cpp                      # Harness principale di benchmark e logging su CSV
│   ├── test_main.cpp                 # Entry point di test per la generazione PNG su disco
│   ├── parallel/
│   │   ├── block_matching_cuda.cu    # Kernel CUDA (GlobalOnly, SharedF1, SharedF1F2, Warp)
│   │   └── interpolation_cuda.cu     # Kernel CUDA (Baseline, Precomputed, SharedTile)
│   ├── sequential/
│   │   ├── block_matching.cpp        # Baseline CPU con early-termination SAD
│   │   └── interpolation.cpp         # Baseline CPU LERP (shift a interi) e Bicubic
│   └── utils/
│       ├── png_utils.cpp             # Implementazione decodifica/codifica PNG 8-bit RGBA
│       └── utils.cpp                 # Generatore di passi temporali
└── input/                            # Cartella contenente i fotogrammi di test (4 frame)
```

---

## 📊 Risultati dei Benchmark

I test misurano la media temporale, la deviazione standard, il tempo minimo e la fedeltà numerica calcolando:
* **Speedup**: $S = \frac{T_{seq}}{T_{CUDA}}$.
* **Massima discrepanza pixel (`max_diff`)**: $\max |P_{seq} - P_{CUDA}|$ su tutti i canali RGBA.
* **Pixel discrepanti (`px_diff_gt1`)**: Conteggio dei pixel con differenza assoluta $> 1$ LSB (dovuta a minime differenze di precisione floating-point tra FPU host e device).

I parametri esplorati includono:
* Dimensione blocco di matching: $B \in \{16, 32, 50, 64, 96, 128\}$.
* Finestra di ricerca: $W \in \{8, 16, 24, 32, 64\}$.
* Dimensioni blocco CUDA 1D: $64, 128, 256, 512, 1024$.
* Dimensioni blocco CUDA 2D: $32 \times 4, 32 \times 8, 32 \times 16, 32 \times 32, 16 \times 8, 16 \times 16, 16 \times 32, 64 \times 4, 64 \times 8, 8 \times 32$.

---

## ✍️ Autori

* **Elio Scaramuzzino** — Ingegneria Informatica, Università degli Studi di Firenze  
  GitHub: [@eliosca99](https://github.com/eliosca99)