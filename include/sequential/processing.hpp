#ifndef PROCESSING_HPP
#define PROCESSING_HPP

#include "../utils/png_utils.hpp"
#include "../frame.hpp"

namespace framegen::processing {

    void lerp_sequential(const FrameSequence& input, int nFrames, FrameSequence& output);
    void bicubic_sequential(const FrameSequence& input, int nFrames, FrameSequence& output);
    void block_matching_sequential(const FrameSequence& input, int nFrames, FrameSequence& output);

}// framegen::processing

#endif