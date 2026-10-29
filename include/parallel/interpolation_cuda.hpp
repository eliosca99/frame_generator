#ifndef INTERPOLATION_CUDA_HPP
#define INTERPOLATION_CUDA_HPP

#include "../frame.hpp"

namespace framegen::interpolation {

void lerp_cuda(const FrameSequence& input, int nFrames, FrameSequence& output, int cudaBlockSize = 256);
void bicubic_cuda(const FrameSequence& input, int nFrames, int block_matching_block_size, int block_matching_search_window_size, FrameSequence& output, int cuda_block_x = 32, int cuda_block_y = 8);

}

#endif