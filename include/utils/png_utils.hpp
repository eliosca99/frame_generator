#ifndef PNG_UTILS_HPP
#define PNG_UTILS_HPP

// funzioni di utilità per caricare e salvare immagini PNG come Frame o sequenze di Frame

#include "../frame.hpp"
#include <optional>
#include <filesystem>
#include <string>

namespace framegen::utils { // namespace per le funzioni di utilità legate al frame generation

namespace fs = std::filesystem;

std::optional<Frame> load_png(const fs::path& path);
std::optional<FrameSequence> load_png_sequence(const fs::path& dir);
bool save_png(const Frame& frame, const fs::path& path);
bool save_png_sequence(const FrameSequence& sequence, const fs::path& dir, const std::string& prefix = "frame");

}//namespace framegen::utils

#endif