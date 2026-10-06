#ifndef BLOCK_MATCHING_HPP 
#define BLOCK_MATCHING_HPP

#include "../frame.hpp"
#include <vector>
#include <optional>

namespace framegen::block_matching {

    struct int2 {
        int x;
        int y;
    };

    inline int2 make_int2(int x, int y) {
        int2 v;
        v.x = x;
        v.y = y;
        return v;
    };

    std::optional<std::vector<int2>> block_matching(const Frame& frame1, const Frame& frame2, int block_size, int search_window_size);

}
#endif