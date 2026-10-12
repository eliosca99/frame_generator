#include <iostream>
#include <filesystem>
#include <array>

#include "../include/utils/png_utils.hpp"
#include "../include/sequential/interpolation.hpp"
#include <optional>

int main() {
    namespace fs = std::filesystem;
    const fs::path base_dir = fs::exists("input") ? "input" : "../input";
    const std::array<const char*, 4> file_names = {
        "bianca.png", "bianca_2.png", "nera.png", "nera_2.png"
    };

    framegen::FrameSequence input;
    for (const char* file_name : file_names) {
        const std::optional<framegen::Frame> frame =
            framegen::utils::load_png(base_dir / file_name);
        if (!frame) {
            std::cerr << "Impossibile caricare il frame " << file_name << std::endl;
            return -1;
        }
        input.push_back(*frame);
    }

    std::cout << "Quattro frame caricati correttamente" << std::endl;

    framegen::FrameSequence output;
    constexpr int nFrames = 1;
    constexpr int blockSize = 50;
    constexpr int searchWindowSize = 20;
    framegen::interpolation::bicubic_sequential(
        input, nFrames, blockSize, searchWindowSize, output);

    const fs::path output_path =
        fs::exists("output") ? "output/bicubic_sequential" : "../output/bicubic_sequential";
    if (framegen::utils::save_png_sequence(output, output_path)) {
        std::cout << "Frame bicubici salvati correttamente in "
                  << output_path << std::endl;
    } else {
        std::cerr << "Impossibile salvare i frame in " << output_path << std::endl;
        return -1;
    }

    return 0;
}