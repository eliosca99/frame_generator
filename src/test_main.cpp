// Genera i frame intermedi con la sola versione sequenziale (senza CUDA) e li salva come PNG.
// Ogni combinazione di parametri finisce in una cartella separata, il cui nome indica
// algoritmo e parametri, per esempio:
//   output/lerp_n3
//   output/bicubic_n3_bs32_sw64
// I frame intermedi sono sempre generati tra il secondo e il terzo frame di input.
//
// Uso: ./frame_generator_seq        (i parametri si cambiano nei vettori qui sotto)

#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "../include/frame.hpp"
#include "../include/utils/png_utils.hpp"
#include "../include/sequential/interpolation.hpp"

namespace fs = std::filesystem;
using namespace framegen;

namespace {
fs::path find_input_directory() {
    const std::array<fs::path, 3> candidates = {
        fs::path(TEST_MAIN_SOURCE_DIR) / "input",
        fs::path("input"),
        fs::path("../input")
    };
    const std::array<const char*, 4> files = {"0.png", "1.png", "2.png", "3.png"};

    for (const fs::path& candidate : candidates) {
        bool complete = true;
        for (const char* file : files) {
            if (!fs::is_regular_file(candidate / file)) {
                complete = false;
                break;
            }
        }
        if (complete) return candidate;
    }
    return {};
}
}

int main() {
    // ---- percorsi ----
    const fs::path inputDir = find_input_directory();
    const fs::path outputDir = "../output";
    const std::array<const char*, 4> files = {"0.png", "1.png", "2.png", "3.png"};

    if (inputDir.empty()) {
        std::cerr << "Impossibile trovare le immagini di input. Cercate in: input, "
                  << "../input" << " e " << TEST_MAIN_SOURCE_DIR << "/input" << std::endl;
        return 1;
    }

    // ---- parametri (modificabili) ----
    const std::vector<int> nFramesList   = {4, 8};            // frame intermedi da generare
    const std::vector<int> blockSizes    = {16, 32};  // block_size del block matching
    const std::vector<int> searchWindows = {72, 80};       // search_window del block matching

    // ---- caricamento dei 4 frame ----
    FrameSequence input;
    for (const char* f : files) {
        std::optional<Frame> fr = utils::load_png(inputDir / f);
        if (!fr) {
            std::cerr << "Impossibile caricare " << (inputDir / f) << std::endl;
            return 1;
        }
        input.push_back(*fr);
    }
    const int W = input[0].width, H = input[0].height;
    std::cout << "Frame caricati: " << W << "x" << H << std::endl;

    // ---- LERP: usa solo i frame 1 e 2 ----
    const FrameSequence pair = {input[1], input[2]};
    for (int n : nFramesList) {
        const std::string name = "lerp_n" + std::to_string(n);
        FrameSequence out;
        interpolation::lerp_sequential(pair, n, out);
        if ((int)out.size() != n || !utils::save_png_sequence(out, outputDir / name, "frame")) {
            std::cerr << "[" << name << "] fallito" << std::endl;
            continue;
        }
        std::cout << "[" << name << "] salvato" << std::endl;
    }

    // ---- BICUBIC: usa tutti e 4 i frame ----
    for (int n : nFramesList) {
        for (int bs : blockSizes) {
            if (bs > std::min(W, H)) continue;   // block_size non valido per questa immagine
            for (int sw : searchWindows) {
                const std::string name = "bicubic_n" + std::to_string(n)
                                       + "_bs" + std::to_string(bs)
                                       + "_sw" + std::to_string(sw);
                FrameSequence out;
                interpolation::bicubic_sequential(input, n, bs, sw, out);
                if ((int)out.size() != n || !utils::save_png_sequence(out, outputDir / name, "frame")) {
                    std::cerr << "[" << name << "] fallito" << std::endl;
                    continue;
                }
                std::cout << "[" << name << "] salvato" << std::endl;
            }
        }
    }
    return 0;
}