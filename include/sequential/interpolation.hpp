#ifndef INTERPOLATION_HPP
#define INTERPOLATION_HPP

#include "../frame.hpp"

namespace framegen::interpolation {

    struct Color {
        float r, g, b, a;
    };

    void spatialWeights(float off, int& i, float w[4]);
    int clampi(int v, int lo, int hi);
    Color sample(const Pixel* frame, int width, int height, int x, int y,
                 const float wx[4], const float wy[4]);

    void lerp_sequential(const FrameSequence& input, int nFrames, FrameSequence& output);
    void bicubic_sequential(const FrameSequence& input, int nFrames, int block_matching_block_size, int block_matching_search_window_size, FrameSequence& output);

}// framegen::interpolation

#endif