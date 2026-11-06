// Benchmark: confronto tra versione sequenziale e CUDA degli algoritmi LERP e BICUBIC.
// Ogni combinazione di parametri viene misurata più volte (media, deviazione standard, minimo).
// I parametri personalizzabili sono i vettori nella sezione "SPAZIO DEI PARAMETRI".
//
// Uso:
//   ./frame_generator [--input DIR] [--csv FILE] [--reps N] [--reps-seq N] [--quick] [--no-lerp]

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "../include/frame.hpp"
#include "../include/utils/png_utils.hpp"
#include "../include/sequential/interpolation.hpp"
#include "../include/parallel/interpolation_cuda.hpp"
#include "../include/parallel/cuda_utils.cuh"

namespace fs = std::filesystem;
using namespace framegen;
using clk = std::chrono::steady_clock;

// ----------------------------------------------------------------------------
// statistiche
// ----------------------------------------------------------------------------
struct Stats {
    double mean = 0, stddev = 0, min = 0;
};

// esegue f() "warm" volte senza misurare, poi "reps" volte misurando il tempo (ms)
template <class F>
Stats measure(int warm, int reps, F&& f) {
    for (int i = 0; i < warm; i++) f();

    std::vector<double> t;
    t.reserve(reps);
    for (int i = 0; i < reps; i++) {
        auto a = clk::now();
        f();
        auto b = clk::now();
        t.push_back(std::chrono::duration<double, std::milli>(b - a).count());
    }

    Stats s;
    for (double v : t) s.mean += v;
    s.mean /= t.size();
    for (double v : t) s.stddev += (v - s.mean) * (v - s.mean);
    s.stddev = t.size() > 1 ? std::sqrt(s.stddev / (t.size() - 1)) : 0.0;
    s.min = *std::min_element(t.begin(), t.end());
    return s;
}

// ----------------------------------------------------------------------------
// confronto tra due sequenze di frame
// ----------------------------------------------------------------------------
struct Diff {
    int maxDiff = 0;      // massima differenza assoluta su un singolo canale
    size_t overOne = 0;   // numero di pixel con almeno un canale che differisce di più di 1
    bool ok = true;       // false se numero/dimensione dei frame non coincidono
};

Diff compare(const FrameSequence& a, const FrameSequence& b) {
    Diff d;
    if (a.size() != b.size()) { d.ok = false; return d; }
    for (size_t k = 0; k < a.size(); k++) {
        if (a[k].width != b[k].width || a[k].height != b[k].height ||
            a[k].data.size() != b[k].data.size()) {
            d.ok = false;
            return d;
        }
        for (size_t i = 0; i < a[k].data.size(); i++) {
            const Pixel& p = a[k].data[i];
            const Pixel& q = b[k].data[i];
            int m = std::max({std::abs(p.r - q.r), std::abs(p.g - q.g),
                              std::abs(p.b - q.b), std::abs(p.a - q.a)});
            d.maxDiff = std::max(d.maxDiff, m);
            if (m > 1) d.overOne++;
        }
    }
    return d;
}

// ----------------------------------------------------------------------------
// opzioni da riga di comando
// ----------------------------------------------------------------------------
struct Options {
    fs::path inputDir = fs::exists("input") ? "input" : "../input";
    std::string csvPath = "../benchmark_results.csv";
    int repsCuda = 10;    // ripetizioni per la versione CUDA
    int repsSeq = 3;      // ripetizioni per la versione sequenziale (molto più lenta)
    bool quick = false;   // spazio dei parametri ridotto
    bool lerp = true;     // esegue anche il benchmark del LERP
};

Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "Manca il valore per " << a << std::endl; std::exit(1); }
            return argv[++i];
        };
        if (a == "--input") o.inputDir = next();
        else if (a == "--csv") o.csvPath = next();
        else if (a == "--reps") o.repsCuda = std::max(1, std::atoi(next().c_str()));
        else if (a == "--reps-seq") o.repsSeq = std::max(1, std::atoi(next().c_str()));
        else if (a == "--quick") o.quick = true;
        else if (a == "--no-lerp") o.lerp = false;
        else { std::cerr << "Opzione sconosciuta: " << a << std::endl; std::exit(1); }
    }
    return o;
}

// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    const Options opt = parse_args(argc, argv);

    // ---- caricamento dei 4 frame ----
    const std::array<const char*, 4> files = {"bianca.png", "bianca_2.png", "nera.png", "nera_2.png"};
    FrameSequence input;
    for (const char* f : files) {
        std::optional<Frame> fr = utils::load_png(opt.inputDir / f);
        if (!fr) {
            std::cerr << "Impossibile caricare " << (opt.inputDir / f) << std::endl;
            return -1;
        }
        input.push_back(*fr);
    }
    const int W = input[0].width, H = input[0].height;
    for (const Frame& f : input) {
        if (f.width != W || f.height != H) {
            std::cerr << "I frame di input hanno dimensioni diverse!" << std::endl;
            return -1;
        }
    }
    std::cout << "Frame caricati: " << W << "x" << H << std::endl;

    int devCount = 0;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount == 0) {
        std::cerr << "Nessun dispositivo CUDA disponibile!" << std::endl;
        return -1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Ripetizioni: CUDA=" << opt.repsCuda << ", sequenziale=" << opt.repsSeq << std::endl;

    // ========================================================================
    // SPAZIO DEI PARAMETRI (modificabile)
    // ========================================================================
    std::vector<int> nFramesList      = {1, 4, 8};                  // frame intermedi da generare
    std::vector<int> blockSizes       = {16, 32, 50, 64, 96, 128};        // block_size del block matching
    std::vector<int> searchWindows    = {8, 16, 24, 32, 64};             // search_window_size
    std::vector<int> cudaBlockSizes   = {64, 128, 256, 512, 1024}; // LERP: thread per blocco CUDA (multipli di 32)
    // BICUBIC: forme (x, y) del blocco CUDA. Il prodotto x*y deve essere multiplo di 32 e <= 1024,
    // perché viene passato anche al block matching come numero di thread per blocco.
    std::vector<std::pair<int, int>> cudaBlocks2D = {
        {32, 4}, {32, 8}, {32, 16}, {32, 32},
        {16, 8}, {16, 16}, {16, 32},
        {64, 4}, {64, 8}, {8, 32}
    };

    std::vector<framegen::block_matching::BMVariant> variants = {
        framegen::block_matching::BMVariant::GlobalOnly,
        framegen::block_matching::BMVariant::SharedF1,
        framegen::block_matching::BMVariant::SharedF1F2,
        framegen::block_matching::BMVariant::WarpPerCandidate
    };

    if (opt.quick) {
        nFramesList    = {1};
        blockSizes     = {32, 50};
        searchWindows  = {8, 16};
        cudaBlockSizes = {128, 256, 512};
        cudaBlocks2D   = {{32, 4}, {32, 8}, {16, 16}, {32, 16}};
        variants       = {framegen::block_matching::BMVariant::GlobalOnly, framegen::block_matching::BMVariant::SharedF1F2};
    }
    // scarto le forme non valide
    cudaBlocks2D.erase(std::remove_if(cudaBlocks2D.begin(), cudaBlocks2D.end(),
        [](const std::pair<int, int>& p) {
            int t = p.first * p.second;
            return p.first <= 0 || p.second <= 0 || t < 32 || t > 1024 || t % 32 != 0;
        }), cudaBlocks2D.end());
    const int warmupCuda = 1;   // esecuzioni scartate prima di misurare (allocazioni, cache, clock GPU)
    const int warmupSeq = 0;

    // ---- file CSV ----
    std::ofstream csv(opt.csvPath);
    if (!csv) {
        std::cerr << "Impossibile aprire " << opt.csvPath << std::endl;
        return -1;
    }
    csv << "algoritmo,variant,nFrames,block_size,search_window,cuda_block_x,cuda_block_y,"
           "seq_ms_mean,seq_ms_std,cuda_ms_mean,cuda_ms_std,cuda_ms_min,speedup,max_diff,px_diff_gt1\n";

    warmup();  // inizializza il contesto CUDA fuori dalle misurazioni

    // ========================================================================
    // BENCHMARK BICUBIC
    // ========================================================================
    std::cout << "\n================ BICUBIC ================\n";

    struct Best { double speedup = 0; int n = 0, bs = 0, sw = 0, bx = 0, by = 0; framegen::block_matching::BMVariant variant; double cudaMs = 0, seqMs = 0; };
    Best globalBest;

    auto variant_to_string = [](framegen::block_matching::BMVariant v) {
        using framegen::block_matching::BMVariant;
        switch(v) {
            case BMVariant::GlobalOnly: return "GlobalOnly";
            case BMVariant::SharedF1: return "SharedF1";
            case BMVariant::SharedF1F2: return "SharedF1F2";
            case BMVariant::WarpPerCandidate: return "WarpPerCandidate";
            default: return "Unknown";
        }
    };

    for (int n : nFramesList) {
        for (int bs : blockSizes) {
            // combinazioni non valide: salto, la funzione stamperebbe solo un errore
            if (bs > std::min(W, H) || (size_t)bs * bs * sizeof(Pixel) > 48 * 1024) {
                std::cout << "\n[salto] block_size=" << bs << " non valido per questa immagine\n";
                continue;
            }
            for (int sw : searchWindows) {
                std::cout << "\n--- nFrames=" << n << "  block_size=" << bs
                          << "  search_window=" << sw << " ---\n";

                // 1. versione sequenziale: dipende solo da (n, bs, sw), la misuro una volta
                FrameSequence outSeq;
                Stats seq = measure(warmupSeq, opt.repsSeq, [&] {
                    outSeq.clear();
                    interpolation::bicubic_sequential(input, n, bs, sw, outSeq);
                });
                if ((int)outSeq.size() != n) {
                    std::cerr << "Sequenziale fallito, salto la configurazione\n";
                    continue;
                }
                std::printf("  sequenziale: %10.2f ms  (std %.2f)\n", seq.mean, seq.stddev);
                
                for (auto var : variants) {
                    std::cout << "\n  --- Variante: " << variant_to_string(var) << " ---\n";
                    std::printf("  %-8s %12s %10s %10s %10s %10s\n",
                                "blocco", "cuda [ms]", "std", "min", "speedup", "maxdiff");

                    // 2. versione CUDA: una misura per ogni forma (x, y) del blocco CUDA
                    Best best;
                    for (const auto& blk : cudaBlocks2D) {
                        const int bx = blk.first, by = blk.second;
                        char lbl[16];
                        std::snprintf(lbl, sizeof(lbl), "%dx%d", bx, by);
                        FrameSequence outCuda;
                        Stats cu = measure(warmupCuda, opt.repsCuda, [&] {
                            outCuda.clear();
                            interpolation::bicubic_cuda(input, n, bs, sw, outCuda, var, bx, by);
                        });
                        if ((int)outCuda.size() != n) {
                            std::printf("  %-8s (fallito)\n", lbl);
                            continue;
                        }

                        // 3. validazione dell'ultimo output CUDA rispetto al sequenziale
                        Diff d = compare(outSeq, outCuda);
                        double speedup = seq.mean / cu.mean;
                        std::printf("  %-8s %12.2f %10.2f %10.2f %9.2fx %10d%s\n",
                                    lbl, cu.mean, cu.stddev, cu.min, speedup, d.maxDiff,
                                    (!d.ok || d.maxDiff > 2) ? "  <-- DIFFERENZE!" : "");

                        csv << "bicubic," << variant_to_string(var) << "," << n << "," << bs << "," << sw << "," << bx << "," << by << ","
                            << seq.mean << "," << seq.stddev << ","
                            << cu.mean << "," << cu.stddev << "," << cu.min << ","
                            << speedup << "," << d.maxDiff << "," << d.overOne << "\n";

                        if (speedup > best.speedup) best = {speedup, n, bs, sw, bx, by, var, cu.mean, seq.mean};
                    }
                    if (best.speedup > 0) {
                        std::printf("  => miglior blocco per %s: %dx%d (speedup %.2fx)\n", 
                                    variant_to_string(var), best.bx, best.by, best.speedup);
                        if (best.speedup > globalBest.speedup) globalBest = best;
                    }
                }
            }
        }
    }

    // ========================================================================
    // BENCHMARK LERP (usa F1 e F2, cioè input[1] e input[2])
    // ========================================================================
    if (opt.lerp) {
        std::cout << "\n================ LERP ================\n";
        const FrameSequence pair = {input[1], input[2]};

        for (int n : nFramesList) {
            std::cout << "\n--- nFrames=" << n << " ---\n";

            FrameSequence outSeq;
            Stats seq = measure(warmupSeq, opt.repsCuda, [&] {   // il LERP è veloce: uso più ripetizioni
                outSeq.clear();
                interpolation::lerp_sequential(pair, n, outSeq);
            });
            std::printf("  sequenziale: %10.2f ms  (std %.2f)\n", seq.mean, seq.stddev);
            std::printf("  %-8s %12s %10s %10s %10s %10s\n",
                        "cudaBlk", "cuda [ms]", "std", "min", "speedup", "maxdiff");

            for (int cbs : cudaBlockSizes) {
                FrameSequence outCuda;
                Stats cu = measure(warmupCuda, opt.repsCuda, [&] {
                    outCuda.clear();
                    interpolation::lerp_cuda(pair, n, outCuda, cbs);
                });
                if ((int)outCuda.size() != n) {
                    std::printf("  %-8d (fallito)\n", cbs);
                    continue;
                }
                Diff d = compare(outSeq, outCuda);
                double speedup = seq.mean / cu.mean;
                std::printf("  %-8d %12.2f %10.2f %10.2f %9.2fx %10d%s\n",
                            cbs, cu.mean, cu.stddev, cu.min, speedup, d.maxDiff,
                            (!d.ok || d.maxDiff > 0) ? "  <-- DIFFERENZE!" : "");

                csv << "lerp," << n << ",,," << cbs << ",,"
                    << seq.mean << "," << seq.stddev << ","
                    << cu.mean << "," << cu.stddev << "," << cu.min << ","
                    << speedup << "," << d.maxDiff << "," << d.overOne << "\n";
            }
        }
    }

    // ---- riepilogo ----
    std::cout << "\n================ RIEPILOGO ================\n";
    if (globalBest.speedup > 0) {
        std::printf("Miglior speedup bicubic: %.2fx\n", globalBest.speedup);
        std::printf("  Variante: %s  nFrames=%d  block_size=%d  search_window=%d  blocco CUDA=%dx%d\n",
                    variant_to_string(globalBest.variant), globalBest.n, globalBest.bs, globalBest.sw, globalBest.bx, globalBest.by);
        std::printf("  sequenziale %.2f ms  vs  CUDA %.2f ms\n", globalBest.seqMs, globalBest.cudaMs);
    }
    std::cout << "Risultati completi in: " << opt.csvPath << std::endl;
    return 0;
}