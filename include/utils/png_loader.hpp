#ifndef PNG_LOADER_HPP
#define PNG_LOADER_HPP

#include "../frame.hpp"
#include <optional>
#include <filesystem>

namespace framegen::utils {

namespace fs = std::filesystem;

std::optional<Frame> load_png(const fs::path& path);
std::optional<FrameSequence> load_png_sequence(const fs::path& dir);

}//namespace framegen::utils

#endif