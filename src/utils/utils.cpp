#include "../include/utils/utils.hpp"

namespace framegen::utils {

std::vector<float> make_times(int nFrames) {
    std::vector<float> times;
    times.reserve(nFrames);
    float step = 1.0f / static_cast<float>(nFrames + 1);
    float cur = step;
    for (int i = 0; i < nFrames; i++) {
        times.push_back(cur);
        cur += step;
    }
    return times;
}

}// namespace framegen::utils