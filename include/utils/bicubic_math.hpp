#ifndef BICUBIC_MATH_HPP
#define BICUBIC_MATH_HPP

#include <cmath>
#include "../frame.hpp"

#ifdef __CUDACC__
#define FG_HD __host__ __device__
#else
#define FG_HD
#endif

namespace framegen::utils {

struct Color { float r, g, b, a; };

FG_HD inline void temporalWeights(float t, float& w0, float& w2, float& w3) {
    float t2 = t * t, t3 = t2 * t;
    w0 = 0.5f * (-t3 + 2.f*t2 - t);
    w2 = 0.5f * (-3.f*t3 + 4.f*t2 + t);
    w3 = 0.5f * (t3 - t2);
}

FG_HD inline void spatialWeights(float off, int& i, float w[4]) {
    i = (int)floorf(off);
    float f = off - i, f2 = f * f, f3 = f2 * f;
    w[0] = 0.5f * (-f3 + 2.f*f2 - f);
    w[1] = 0.5f * ( 3.f*f3 - 5.f*f2 + 2.f);
    w[2] = 0.5f * (-3.f*f3 + 4.f*f2 + f);
    w[3] = 0.5f * ( f3 - f2);
}

FG_HD inline int clampi(int v, int lo, int hi) {
    // funzione che esegue il clamping di v (indice del pixel nel blocco)
    return v < lo ? lo : (v > hi ? hi : v);
}

FG_HD inline Color sample(const Pixel* F, int W, int H, int X, int Y,
                          const float wx[4], const float wy[4]) {
    Color res = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int yy = clampi(Y - 1 + j, 0, H - 1);
        Color rowColor = {0.f, 0.f, 0.f, 0.f};
        // con questo for, applico i pesi ai 4 pixel della riga yy
        for (int i = 0; i < 4; i++) {
            int xx = clampi(X - 1 + i, 0, W - 1);
            Pixel p = F[yy * W + xx];
            rowColor.r += wx[i] * p.r;
            rowColor.g += wx[i] * p.g;
            rowColor.b += wx[i] * p.b;
            rowColor.a += wx[i] * p.a;
        }
        // finita la passata sulla riga, applico i pesi in yy; è una somma, quindi la faccio per ogni yy
        // e non tutta alla fine
        res.r += wy[j] * rowColor.r;
        res.g += wy[j] * rowColor.g;
        res.b += wy[j] * rowColor.b;
        res.a += wy[j] * rowColor.a;
    }
    return res;
}

} // namespace framegen::utils
#endif