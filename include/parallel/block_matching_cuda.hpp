#ifndef BLOCK_MATCHING_CUDA_HPP
#define BLOCK_MATCHING_CUDA_HPP

#include <optional>
#include <vector>
#include "../frame.hpp"
#include "../utils/utils.hpp"

namespace framegen::block_matching_cuda {

    using framegen::utils::int2;
    std::optional<std::vector<int2>> BM_cuda(const Frame& frame1, const Frame& frame2, int block_size, int search_window_size);

} //namespace

#endif