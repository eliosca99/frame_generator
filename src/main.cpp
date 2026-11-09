// Benchmark completo: LERP, Block Matching Standalone e Bicubic (con tutte le varianti).
//
// Uso:
//   ./frame_generator [--input DIR] [--csv FILE] [--reps N] [--reps-seq N]
//                     [--quick] [--no-lerp] [--no-bm] [--no-bicubic]

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
#include <set>
#include <string>
#include <vector>

#include "../include/frame.hpp"
#include "../include/utils/png_utils.hpp"
#include "../include/sequential/block_matching.hpp"
#include "../include/sequential/interpolation.hpp"
#include "../include/parallel/block_matching_cuda.hpp"
#include "../include/parallel/interpolation_cuda.hpp"
#include "../include/parallel/cuda_utils.cuh"

namespace fs = std::filesystem;
using namespace framegen;
using framegen::block_matching::BMVariant;
using framegen::interpolation::BicubicVariant;
//using framegen::utils::int2;
using clk = std::chrono::steady_clock;

// ----------------------------------------------------------------------------
// Statistiche
// ----------------------------------------------------------------------------
struct Stats {
    double mean = 0, stddev = 0, min = 0;
};

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
// Funzioni di confronto
// ----------------------------------------------------------------------------
struct Diff {
    int maxDiff = 0;
    size_t overOne = 0;
    bool ok = true;
};

Diff compare_frames(const FrameSequence& a, const FrameSequence& b) {
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

Diff compare_mv(const std::vector<framegen::utils::int2>& a,
                const std::vector<framegen::utils::int2>& b) {
    Diff d;
    if (a.size() != b.size()) { d.ok = false; return d; }
    for (size_t i = 0; i < a.size(); i++) {
        int m = std::max(std::abs(a[i].x - b[i].x), std::abs(a[i].y - b[i].y));
        d.maxDiff = std::max(d.maxDiff, m);
        if (m > 0) d.overOne++;
    }
    return d;
}

// ----------------------------------------------------------------------------
// Helper stringhe enum
// ----------------------------------------------------------------------------
const char* bm_variant_to_string(BMVariant v) {
    switch (v) {
        case BMVariant::GlobalOnly:       return "GlobalOnly";
        case BMVariant::SharedF1:         return "SharedF1";
        case BMVariant::SharedF1F2:       return "SharedF1F2";
        case BMVariant::WarpPerCandidate: return "WarpPerCandidate";
        default:                          return "Unknown";
    }
}

const char* bicubic_variant_to_string(BicubicVariant v) {
    switch (v) {
        case BicubicVariant::Baseline:    return "Baseline";
        case BicubicVariant::Precomputed: return "Precomputed";
        case BicubicVariant::SharedTile:  return "SharedTile";
        default:                          return "Unknown";
    }
}

// ----------------------------------------------------------------------------
// Opzioni da riga di comando
// ----------------------------------------------------------------------------
struct Options {
    fs::path inputDir = fs::exists("input") ? "input" : "../input";
    std::string csvPath = "../benchmark_results.csv";
    int repsCuda = 10;
    int repsSeq = 3;
    bool quick = false;
    bool lerp = true;
    bool bm = true;
    bool bicubic = true;
};

Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Manca il valore per " << a << std::endl;
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--input") o.inputDir = next();
        else if (a == "--csv") o.csvPath = next();
        else if (a == "--reps") o.repsCuda = std::max(1, std::atoi(next().c_str()));
        else if (a == "--reps-seq") o.repsSeq = std::max(1, std::atoi(next().c_str()));
        else if (a == "--quick") o.quick = true;
        else if (a == "--no-lerp") o.lerp = false;
        else if (a == "--no-bm") o.bm = false;
        else if (a == "--no-bicubic") o.bicubic = false;
        else {
            std::cerr << "Opzione sconosciuta: " << a << std::endl;
            std::exit(1);
        }
    }
    return o;
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    const Options opt = parse_args(argc, argv);

    // ---- Caricamento dei 4 frame ----
    const std::array<const char*, 4> files = {"primo.png", "secondo.png", "terzo.png", "quarto.png"};
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
    // Spazio dei parametri
    // ========================================================================
    std::vector<int> nFramesList    = {1, 4, 8};
    std::vector<int> blockSizes     = {16, 32, 50, 64, 96, 128};
    std::vector<int> searchWindows  = {8, 16, 24, 32, 64};
    std::vector<int> cudaBlockSizes = {64, 128, 256, 512, 1024};

    std::vector<std::pair<int, int>> cudaBlocks2D = {
        {32, 4}, {32, 8}, {32, 16}, {32, 32},
        {16, 8}, {16, 16}, {16, 32},
        {64, 4}, {64, 8}, {8, 32}
    };

    std::vector<BMVariant> bmVariants = {
        BMVariant::GlobalOnly,
        BMVariant::SharedF1,
        BMVariant::SharedF1F2,
        BMVariant::WarpPerCandidate
    };

    std::vector<BicubicVariant> bicubicVariants = {
        BicubicVariant::Baseline,
        BicubicVariant::Precomputed,
        BicubicVariant::SharedTile
    };

    if (opt.quick) {
        nFramesList     = {1};
        blockSizes      = {32, 50};
        searchWindows   = {8, 16};
        cudaBlockSizes  = {128, 256};
        cudaBlocks2D    = {{32, 4}, {32, 8}, {16, 16}};
        bmVariants      = {BMVariant::GlobalOnly, BMVariant::SharedF1F2};
        bicubicVariants = {BicubicVariant::Baseline, BicubicVariant::SharedTile};
    }

    cudaBlocks2D.erase(std::remove_if(cudaBlocks2D.begin(), cudaBlocks2D.end(),
        [](const std::pair<int, int>& p) {
            int t = p.first * p.second;
            return p.first <= 0 || p.second <= 0 || t < 32 || t > 1024 || t % 32 != 0;
        }), cudaBlocks2D.end());

    const int warmupCuda = 1;
    const int warmupSeq  = 0;

    // ---- File CSV ----
    std::ofstream csv(opt.csvPath);
    if (!csv) {
        std::cerr << "Impossibile aprire " << opt.csvPath << std::endl;
        return -1;
    }
    csv << "algoritmo,bm_variant,bicubic_variant,nFrames,block_size,search_window,"
           "cuda_block_x,cuda_block_y,seq_ms_mean,seq_ms_std,cuda_ms_mean,cuda_ms_std,"
           "cuda_ms_min,speedup,max_diff,px_diff_gt1\n";

    warmup();

    // ========================================================================
    // 1. BENCHMARK LERP (Sequenziale vs CUDA)
    // ========================================================================
    if (opt.lerp) {
        std::cout << "\n================ LERP ================\n";
        const FrameSequence pair = {input[1], input[2]};

        for (int n : nFramesList) {
            std::cout << "\n--- nFrames=" << n << " ---\n";

            FrameSequence outSeq;
            Stats seq = measure(warmupSeq, opt.repsSeq, [&] {
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
                Diff d = compare_frames(outSeq, outCuda);
                double speedup = seq.mean / cu.mean;
                std::printf("  %-8d %12.2f %10.2f %10.2f %9.2fx %10d%s\n",
                            cbs, cu.mean, cu.stddev, cu.min, speedup, d.maxDiff,
                            (!d.ok || d.maxDiff > 0) ? "  <-- DIFFERENZE!" : "");

                csv << "lerp,-,-," << n << ",,," << cbs << ",1,"
                    << seq.mean << "," << seq.stddev << ","
                    << cu.mean << "," << cu.stddev << "," << cu.min << ","
                    << speedup << "," << d.maxDiff << "," << d.overOne << "\n";
            }
        }
    }

    // ========================================================================
    // 2. BENCHMARK BLOCK MATCHING STANDALONE (F1 -> F2)
    // ========================================================================
    if (opt.bm) {
        std::cout << "\n================ BLOCK MATCHING STANDALONE (F1 -> F2) ================\n";
        const Frame& F1 = input[1];
        const Frame& F2 = input[2];

        for (int bs : blockSizes) {
            if (bs <= 0 || bs > std::min(W, H) || (size_t)bs * bs * sizeof(Pixel) > 48 * 1024) {
                continue;
            }
            for (int sw : searchWindows) {
                std::cout << "\n--- block_size=" << bs << "  search_window=" << sw << " ---\n";

                std::optional<std::vector<framegen::utils::int2>> mvSeq;
                Stats seq = measure(warmupSeq, opt.repsSeq, [&] {
                    mvSeq = block_matching::block_matching(F1, F2, bs, sw);
                });
                if (!mvSeq) {
                    std::cerr << "Block Matching sequenziale fallito, salto\n";
                    continue;
                }
                std::printf("  sequenziale: %10.2f ms  (std %.2f)\n", seq.mean, seq.stddev);

                for (BMVariant bmVar : bmVariants) {
                    if (!block_matching::bm_variant_supported(bmVar, bs, sw)) {
                        std::printf("  [salto] BMVariant=%s non supportata (smem > 48KB)\n",
                                    bm_variant_to_string(bmVar));
                        continue;
                    }

                    std::cout << "\n  --- BMVariant: " << bm_variant_to_string(bmVar) << " ---\n";
                    std::printf("  %-8s %12s %10s %10s %10s %10s\n",
                                "cudaBlk", "cuda [ms]", "std", "min", "speedup", "maxdiff");

                    for (int cbs : cudaBlockSizes) {
                        std::optional<std::vector<framegen::utils::int2>> mvCuda;
                        Stats cu = measure(warmupCuda, opt.repsCuda, [&] {
                            mvCuda = block_matching::BM_cuda(F1, F2, bs, sw, bmVar, cbs);
                        });
                        if (!mvCuda || mvCuda->empty()) {
                            std::printf("  %-8d (fallito)\n", cbs);
                            continue;
                        }

                        Diff d = compare_mv(*mvSeq, *mvCuda);
                        double speedup = seq.mean / cu.mean;
                        std::printf("  %-8d %12.2f %10.2f %10.2f %9.2fx %10d%s\n",
                                    cbs, cu.mean, cu.stddev, cu.min, speedup, d.maxDiff,
                                    (!d.ok || d.maxDiff > 0) ? "  <-- DIFFERENZE!" : "");

                        csv << "block_matching," << bm_variant_to_string(bmVar) << ",-,1,"
                            << bs << "," << sw << "," << cbs << ",1,"
                            << seq.mean << "," << seq.stddev << ","
                            << cu.mean << "," << cu.stddev << "," << cu.min << ","
                            << speedup << "," << d.maxDiff << "," << d.overOne << "\n";
                    }
                }
            }
        }
    }

    // ========================================================================
    // 3. BENCHMARK BICUBIC COMPLETO (Tutte le combinazioni)
    // ========================================================================
    if (opt.bicubic) {
        std::cout << "\n================ BICUBIC COMPLETO ================\n";

        struct BestBicubic {
            double speedup = 0;
            int n = 0, bs = 0, sw = 0, bx = 0, by = 0;
            BMVariant bmVar = BMVariant::GlobalOnly;
            BicubicVariant bicVar = BicubicVariant::Baseline;
            double cudaMs = 0, seqMs = 0;
        };
        BestBicubic globalBestBicubic;

        for (int n : nFramesList) {
            for (int bs : blockSizes) {
                if (bs <= 0 || bs > std::min(W, H) || (size_t)bs * bs * sizeof(Pixel) > 48 * 1024) {
                    continue;
                }
                for (int sw : searchWindows) {
                    std::cout << "\n--- nFrames=" << n << "  block_size=" << bs
                              << "  search_window=" << sw << " ---\n";

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

                    for (BMVariant bmVar : bmVariants) {
                        if (!block_matching::bm_variant_supported(bmVar, bs, sw)) continue;

                        for (BicubicVariant bicVar : bicubicVariants) {
                            if (!interpolation::bicubic_variant_supported(bicVar, bs)) continue;

                            std::cout << "\n  --- BM: " << bm_variant_to_string(bmVar)
                                      << " | Bicubic: " << bicubic_variant_to_string(bicVar) << " ---\n";
                            std::printf("  %-8s %12s %10s %10s %10s %10s\n",
                                        "blocco", "cuda [ms]", "std", "min", "speedup", "maxdiff");

                            BestBicubic bestCombo;
                            std::set<int> seenThreads1D;

                            for (const auto& blk : cudaBlocks2D) {
                                const int bx = blk.first, by = blk.second;
                                if (bicVar == BicubicVariant::SharedTile) {
                                    if (!seenThreads1D.insert(bx * by).second) continue;
                                }

                                char lbl[16];
                                if (bicVar == BicubicVariant::SharedTile) {
                                    std::snprintf(lbl, sizeof(lbl), "%d(1D)", bx * by);
                                } else {
                                    std::snprintf(lbl, sizeof(lbl), "%dx%d", bx, by);
                                }

                                FrameSequence outCuda;
                                Stats cu = measure(warmupCuda, opt.repsCuda, [&] {
                                    outCuda.clear();
                                    interpolation::bicubic_cuda(input, n, bs, sw, outCuda,
                                                                bmVar, bx, by, bicVar);
                                });
                                if ((int)outCuda.size() != n) {
                                    std::printf("  %-8s (fallito)\n", lbl);
                                    continue;
                                }

                                Diff d = compare_frames(outSeq, outCuda);
                                double speedup = seq.mean / cu.mean;
                                std::printf("  %-8s %12.2f %10.2f %10.2f %9.2fx %10d%s\n",
                                            lbl, cu.mean, cu.stddev, cu.min, speedup, d.maxDiff,
                                            (!d.ok || d.maxDiff > 2) ? "  <-- DIFFERENZE!" : "");

                                csv << "bicubic," << bm_variant_to_string(bmVar) << ","
                                    << bicubic_variant_to_string(bicVar) << ","
                                    << n << "," << bs << "," << sw << "," << bx << "," << by << ","
                                    << seq.mean << "," << seq.stddev << ","
                                    << cu.mean << "," << cu.stddev << "," << cu.min << ","
                                    << speedup << "," << d.maxDiff << "," << d.overOne << "\n";

                                if (speedup > bestCombo.speedup) {
                                    bestCombo = {speedup, n, bs, sw, bx, by, bmVar, bicVar, cu.mean, seq.mean};
                                }
                            }

                            if (bestCombo.speedup > 0) {
                                std::printf("  => miglior blocco per [%s + %s]: %dx%d (speedup %.2fx)\n",
                                            bm_variant_to_string(bmVar), bicubic_variant_to_string(bicVar),
                                            bestCombo.bx, bestCombo.by, bestCombo.speedup);
                                if (bestCombo.speedup > globalBestBicubic.speedup) {
                                    globalBestBicubic = bestCombo;
                                }
                            }
                        }
                    }
                }
            }
        }

        if (globalBestBicubic.speedup > 0) {
            std::cout << "\n================ RIEPILOGO BICUBIC ================\n";
            std::printf("Miglior speedup BICUBIC: %.2fx\n", globalBestBicubic.speedup);
            std::printf("  BM=%s  Bicubic=%s  nFrames=%d  bs=%d  sw=%d  blocco CUDA=%dx%d\n",
                        bm_variant_to_string(globalBestBicubic.bmVar),
                        bicubic_variant_to_string(globalBestBicubic.bicVar),
                        globalBestBicubic.n, globalBestBicubic.bs, globalBestBicubic.sw,
                        globalBestBicubic.bx, globalBestBicubic.by);
            std::printf("  sequenziale %.2f ms  vs  CUDA %.2f ms\n",
                        globalBestBicubic.seqMs, globalBestBicubic.cudaMs);
        }
    }

    std::cout << "\nBenchmark completati con successo! Risultati salvati in: " << opt.csvPath << std::endl;
    return 0;
}