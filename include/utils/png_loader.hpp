#ifndef PNG_LOADER_HPP
#define PNG_LOADER_HPP

// funzioni di utilità per caricare immagini PNG come Frame o sequenze di Frame

#include "../frame.hpp"
#include <optional>
#include <filesystem>

namespace framegen::utils { // namespace per le funzioni di utilità legate al frame generation

namespace fs = std::filesystem;

std::optional<Frame> load_png(const fs::path& path);
std::optional<FrameSequence> load_png_sequence(const fs::path& dir);

}//namespace framegen::utils

#endif