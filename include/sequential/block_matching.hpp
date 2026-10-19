#ifndef BLOCK_MATCHING_HPP 
#define BLOCK_MATCHING_HPP

#include "../frame.hpp"
#include "../utils/utils.hpp"
#include <vector>
#include <optional>

namespace framegen::block_matching {

    namespace utils = framegen::utils;
    std::optional<std::vector<utils::int2>> block_matching(const Frame& frame1, const Frame& frame2, int block_size, int search_window_size);

}
#endif