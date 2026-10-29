#ifndef BLOCK_MATCHING_CUDA_HPP
#define BLOCK_MATCHING_CUDA_HPP

#include <optional>
#include <vector>
#include "../frame.hpp"
#include "../utils/utils.hpp"
#include "../parallel/cuda_utils.cuh"

namespace framegen::block_matching {

    using framegen::utils::int2;
    bool bm_params_valid(int W, int H, int block_size, int search_window_size, int cuda_block_size);
    std::optional<std::vector<int2>> BM_cuda(const Frame& frame1, const Frame& frame2, int block_size, int search_window_size, int cuda_block_size = 256);
    void BM_cuda_device(const Pixel* d_A, const Pixel* d_B, int2* d_out, int width, int height, int block_size, int search_window_size, int cuda_block_size = 256);
} //namespace

#endif