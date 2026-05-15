#ifndef FRAME_HPP
#define FRAME_HPP

#include <cstdint>
#include <vector>

namespace framegen {

enum class PixelFormat: uint8_t {
    GRAY = 1,
    RGB = 3,
    RGBA = 4
};

struct Frame {
    std::vector<uint8_t> data;
    int width;
    int height;
    PixelFormat format;
};

using FrameSequence = std::vector<Frame>; // per usare una sequenza di frame, simile al typedef

}//namespace framegen
#endif