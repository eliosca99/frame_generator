#include <iostream>
#include <filesystem>

#include "../include/utils/png_utils.hpp"
#include "../include/sequential/processing.hpp"
#include <optional>

int main() {
    namespace fs = std::filesystem;
    fs::path base_dir = "../input";
    fs::path file_name_1 = "bianca.png";
    fs::path file_name_2 = "nera.png";
    fs::path path_1 = base_dir / file_name_1;
    fs::path path_2 = base_dir / file_name_2;
    std::optional<framegen::Frame> loaded_frame_1 = framegen::utils::load_png(path_1);
    std::optional<framegen::Frame> loaded_frame_2 = framegen::utils::load_png(path_2);
    if (loaded_frame_1 && loaded_frame_2) {
        std::cout << "Frames caricati correttamente" << std::endl;

        framegen::FrameSequence seq;
        framegen::FrameSequence output;
        // loaded_frame_1 and loaded_frame_2 are optionals; push the contained Frame
        seq.push_back(*loaded_frame_1);
        seq.push_back(*loaded_frame_2);
        int nFrames = 10;
        framegen::processing::lerp_sequential(seq, nFrames, output);
        fs::path output_path = "../output/";
        if (framegen::utils::save_png_sequence(output, output_path)) {
            std::cout << "Frame salvato correttamente in " << output_path << std::endl;
        } else {
            std::cerr << "Impossibile salvare il frame in " << output_path << std::endl;
        }
    } else {
        std::cerr << "Frames non caricati" << std::endl;
        return -1;
    }

    return 0;
}