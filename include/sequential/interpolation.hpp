#ifndef INTERPOLATION_HPP
#define INTERPOLATION_HPP

#include "../frame.hpp"

namespace framegen::interpolation {

    void lerp_sequential(const FrameSequence& input, int nFrames, FrameSequence& output);
    void bicubic_sequential(const FrameSequence& input, int nFrames, int block_matching_block_size, int block_matching_search_window_size, FrameSequence& output);

}// framegen::interpolation

#endif