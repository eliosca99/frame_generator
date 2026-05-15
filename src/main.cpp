#include <iostream>
#include <filesystem>

#include "../include/utils/png_loader.hpp"

int main() {
    namespace fs = std::filesystem;
    fs::path base_dir = "../input";
    fs::path file_name = "test.png";
    fs::path path = base_dir / file_name;
    if(auto loaded_frame = framegen::utils::load_png(path)) {
        std::cout << "Frame caricato correttamente" << std::endl;
        std::cout << static_cast<int>(loaded_frame.value().format) << std::endl;
            std::cout << loaded_frame.value().width << std::endl;
            std::cout << loaded_frame.value().height << std::endl;
    } else {
        std::cerr << "Frame non caricato" << std::endl;
        return -1;
    }

    return 0;
}