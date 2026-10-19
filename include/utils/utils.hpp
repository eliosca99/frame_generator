#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>

namespace framegen::utils {

struct int2 {
        int x;
        int y;
    };

    inline int2 make_int2(int x, int y) {
        int2 v;
        v.x = x;
        v.y = y;
        return v;
    }

std::vector<float> make_times(int nFrames);

}// namespace framegen::utils

#endif