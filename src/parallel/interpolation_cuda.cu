#include "../../include/parallel/interpolation_cuda.hpp" 
#include "../../include/utils/utils.hpp"
#include "../../include/utils/bicubic_math.hpp"
#include "../../include/frame.hpp"
#include "../../include/parallel/cuda_utils.cuh"

#include <iostream>

namespace framegen::interpolation {
using framegen::utils::int2;
using framegen::utils::Color;
using framegen::utils::spatialWeights;
using framegen::utils::sample;
using framegen::utils::clampi;



// --- UTILS ---
// parametri di un blocco di matching per un dato t (80 byte)
struct alignas(16) BlockParams {
    int ix1, iy1, ix2, iy2;       // parte intera degli offset di lettura in F1 e F2
    float4 wx1, wy1, wx2, wy2;    // pesi cubici spaziali
};

__device__ inline BlockParams make_params(int2 m0, int2 m2, int2 m3,
                                          float w0, float w2, float w3) {
    const float Dx = w0 * m0.x + w2 * m2.x + w3 * m3.x;
    const float Dy = w0 * m0.y + w2 * m2.y + w3 * m3.y;
    BlockParams P;
    float wx1[4], wy1[4], wx2[4], wy2[4];
    spatialWeights(-Dx,        P.ix1, wx1);
    spatialWeights(-Dy,        P.iy1, wy1);
    spatialWeights(m2.x - Dx,  P.ix2, wx2);
    spatialWeights(m2.y - Dy,  P.iy2, wy2);
    P.wx1 = make_float4(wx1[0], wx1[1], wx1[2], wx1[3]);
    P.wy1 = make_float4(wy1[0], wy1[1], wy1[2], wy1[3]);
    P.wx2 = make_float4(wx2[0], wx2[1], wx2[2], wx2[3]);
    P.wy2 = make_float4(wy2[0], wy2[1], wy2[2], wy2[3]);
    return P;
}
__device__ inline void unpack(const float4& v, float w[4]) {
    w[0] = v.x; w[1] = v.y; w[2] = v.z; w[3] = v.w;
}

__device__ inline Pixel blend(const Color& c1, const Color& c2, float t) {
    Pixel o;
    o.r = (uint8_t)(fminf(fmaxf((1.f - t) * c1.r + t * c2.r, 0.f), 255.f) + 0.5f);
    o.g = (uint8_t)(fminf(fmaxf((1.f - t) * c1.g + t * c2.g, 0.f), 255.f) + 0.5f);
    o.b = (uint8_t)(fminf(fmaxf((1.f - t) * c1.b + t * c2.b, 0.f), 255.f) + 0.5f);
    o.a = (uint8_t)(fminf(fmaxf((1.f - t) * c1.a + t * c2.a, 0.f), 255.f) + 0.5f);
    return o;
}

bool bicubic_variant_supported(BicubicVariant v, int bs) {
    if (v != BicubicVariant::SharedTile) return true;
    const size_t bytes = 2 * (size_t)(bs + 3) * (bs + 3) * sizeof(Pixel) + sizeof(BlockParams);
    return bytes <= 48 * 1024;
}


// --- KERNELS ---
__global__ void lerp_kernel(const Pixel* d_F1, const Pixel* d_F2, Pixel* d_out, int n, uint16_t T, uint16_t T_inv) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if(threadId < n) {
        Pixel p;
        Pixel p1 = d_F1[threadId];
        Pixel p2 = d_F2[threadId];
        p.r = static_cast<uint8_t>(((uint32_t)p1.r * T_inv + (uint32_t)p2.r * T + 128) >> 8);
        p.g = static_cast<uint8_t>(((uint32_t)p1.g * T_inv + (uint32_t)p2.g * T + 128) >> 8);
        p.b = static_cast<uint8_t>(((uint32_t)p1.b * T_inv + (uint32_t)p2.b * T + 128) >> 8);
        p.a = static_cast<uint8_t>(((uint32_t)p1.a * T_inv + (uint32_t)p2.a * T + 128) >> 8);
        d_out[threadId] = p;
    }
}// lerp_kernel

__global__ void bicubic_kernel(const Pixel* d_F1, const Pixel* d_F2, int2* d_d0, int2* d_d2, int2* d_d3, Pixel* d_out, float t, float w0, float w2, float w3, int W, int H, int block_matching_block_size, int grid_w) {
    // un thread per pixel di output.
    // ricavo prima il blocco di matching in cui il thread si trova
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if(x >= W || y >= H) return;
    int block = ((y / block_matching_block_size) * grid_w) + (x / block_matching_block_size);

    // calcolo D, che indica lo spostamento del blocco b da F1 a F2, usando i pesi di Catmull-Rom
    float Dx = w0 * d_d0[block].x + w2 * d_d2[block].x + w3 * d_d3[block].x;
    float Dy = w0 * d_d0[block].y + w2 * d_d2[block].y + w3 * d_d3[block].y;

    // calcolo la parte intera degli offset di F1 e F2, e con la parte frazionaria calcolo i pesi spaziali, sia su F1 che su F2
    int ixF1, iyF1, ixF2, iyF2;
    float wxF1[4], wyF1[4], wxF2[4], wyF2[4];

    framegen::utils::spatialWeights(     -Dx, ixF1, wxF1);
    framegen::utils::spatialWeights(     -Dy, iyF1, wyF1);
    framegen::utils::spatialWeights(d_d2[block].x - Dx, ixF2, wxF2);
    framegen::utils::spatialWeights(d_d2[block].y - Dy, iyF2, wyF2);

    framegen::utils::Color c1 = framegen::utils::sample(d_F1, W, H, x + ixF1, y + iyF1, wxF1, wyF1);
    framegen::utils::Color c2 = framegen::utils::sample(d_F2, W, H, x + ixF2, y + iyF2, wxF2, wyF2);
    // campionati i due pixel, eseguo il blending dei colori
    float r = ((1.f - t) * c1.r) + (t * c2.r);
    float g = ((1.f - t) * c1.g) + (t * c2.g);
    float b = ((1.f - t) * c1.b) + (t * c2.b);
    float a = ((1.f - t) * c1.a) + (t * c2.a);
    
    Pixel o;
    o.r = (uint8_t)(fminf(fmaxf(r, 0.f), 255.f) + 0.5f);
    o.g = (uint8_t)(fminf(fmaxf(g, 0.f), 255.f) + 0.5f);
    o.b = (uint8_t)(fminf(fmaxf(b, 0.f), 255.f) + 0.5f);
    o.a = (uint8_t)(fminf(fmaxf(a, 0.f), 255.f) + 0.5f);
    d_out[y * W + x] = o;

}// bicubic_kernel

// un thread per blocco di matching
__global__ void bicubic_params_kernel(const int2* d_d0, const int2* d_d2, const int2* d_d3,
                                      BlockParams* d_params, int n_blocks,
                                      float w0, float w2, float w3) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    d_params[b] = make_params(d_d0[b], d_d2[b], d_d3[b], w0, w2, w3);
}// bicubic_params_kernel

// un thread per pixel, legge i parametri già pronti
__global__ void bicubic_kernel_pre(const Pixel* d_F1, const Pixel* d_F2,
                                   const BlockParams* d_params, Pixel* d_out,
                                   float t, int W, int H, int bs, int grid_w) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const int b = (y / bs) * grid_w + (x / bs);
    const BlockParams P = d_params[b];     // 80 byte, letture a 128 bit

    float wx1[4], wy1[4], wx2[4], wy2[4];
    unpack(P.wx1, wx1); unpack(P.wy1, wy1);
    unpack(P.wx2, wx2); unpack(P.wy2, wy2);

    const Color c1 = sample(d_F1, W, H, x + P.ix1, y + P.iy1, wx1, wy1);
    const Color c2 = sample(d_F2, W, H, x + P.ix2, y + P.iy2, wx2, wy2);
    d_out[y * W + x] = blend(c1, c2, t);
}// bicubic_kernel_pre

__global__ void bicubic_kernel_tile(const Pixel* d_F1, const Pixel* d_F2,
                                    const int2* d_d0, const int2* d_d2, const int2* d_d3,
                                    Pixel* d_out, float t, float w0, float w2, float w3,
                                    int W, int H, int bs) {
    extern __shared__ Pixel smem[];
    __shared__ BlockParams P;

    const int grid_c = blockIdx.x, grid_r = blockIdx.y;
    const int b  = grid_r * gridDim.x + grid_c;
    const int x0 = grid_c * bs, y0 = grid_r * bs;
    const int bw = min(bs, W - x0), bh = min(bs, H - y0);   // blocchi parziali sui bordi
    const int tw = bw + 3, th = bh + 3;                      // finestra sorgente: +1 a sinistra/sopra, +2 a destra/sotto

    // 1. parametri del blocco: li calcola un solo thread
    if (threadIdx.x == 0) P = make_params(d_d0[b], d_d2[b], d_d3[b], w0, w2, w3);
    __syncthreads();

    // 2. caricamento cooperativo delle due finestre, con il clamp applicato qui
    Pixel* tile1 = smem;
    Pixel* tile2 = smem + tw * th;
    const int sx1 = x0 + P.ix1 - 1, sy1 = y0 + P.iy1 - 1;
    const int sx2 = x0 + P.ix2 - 1, sy2 = y0 + P.iy2 - 1;
    for (int i = threadIdx.x; i < tw * th; i += blockDim.x) {
        const int r = i / tw, c = i % tw;
        tile1[i] = d_F1[clampi(sy1 + r, 0, H - 1) * W + clampi(sx1 + c, 0, W - 1)];
        tile2[i] = d_F2[clampi(sy2 + r, 0, H - 1) * W + clampi(sx2 + c, 0, W - 1)];
    }
    __syncthreads();

    float wx1[4], wy1[4], wx2[4], wy2[4];
    unpack(P.wx1, wx1); unpack(P.wy1, wy1);
    unpack(P.wx2, wx2); unpack(P.wy2, wy2);

    // 3. ogni thread elabora uno o più pixel del blocco
    for (int p = threadIdx.x; p < bw * bh; p += blockDim.x) {
        const int lx = p % bw, ly = p / bw;
        // coordinate locali: +1 perché sample() legge da X-1 a X+2
        const Color c1 = sample(tile1, tw, th, lx + 1, ly + 1, wx1, wy1);
        const Color c2 = sample(tile2, tw, th, lx + 1, ly + 1, wx2, wy2);
        d_out[(y0 + ly) * W + (x0 + lx)] = blend(c1, c2, t);
    }
}// bicubic_kernel_tile



// --- ALGORITHMS ---
void lerp_cuda(const FrameSequence& input, int nFrames, FrameSequence& output, int cudaBlockSize) {
    // implementazione parallela dell'interpolazione con algoritmo lerp. Con il parametro nFrames, si indicano
    // quanti frame intermedi generare e verranno distribuiti in modo uniforme tra 0.0 e 1.0.

    // check di errori nei parametri
    if(input.size() != 2) {
        std::cerr << "Numero di Frames in input diverso da 2!" << std::endl;
        return;
    }
    if(nFrames < 1) {
        std::cerr << "Numero di Frames intermedi richiesti minore di 1!" << std::endl;
        return;
    }
    const Frame& F1 = input.at(0);
    const Frame& F2 = input.at(1);
    if(F1.height != F2.height || F1.width != F2.width) {
        std::cerr << "Frames in input di dimensione diversa!" << std::endl;
        return;
    }
    size_t expected_size = static_cast<size_t>(F1.height) * static_cast<size_t>(F1.width);
    if(F1.data.size() != expected_size || F2.data.size() != expected_size) {
        std::cerr << "Frames in input con dati non validi!" << std::endl;
        return;
    }

    if(cudaBlockSize < 32 || cudaBlockSize > 1024 || cudaBlockSize % 32 != 0) {
        std::cerr << "cudaBlockSize non valido!" << std::endl;
        return;
    }
    
    size_t n = F1.height * F1.width;
    int gridSize = (n + cudaBlockSize - 1) / cudaBlockSize;

    try {
        DeviceBuffer<Pixel> d_F1(n);
        DeviceBuffer<Pixel> d_F2(n);
        DeviceBuffer<Pixel> d_out(n);
        d_F1.upload(F1.data.data());
        d_F2.upload(F2.data.data());

        auto times = framegen::utils::make_times(nFrames);
        for (float t : times) {
            Frame out;
            out.height = F1.height;
            out.width = F1.width;
            out.data.resize(static_cast<size_t>(n));

            uint16_t T = (uint16_t)(t * 256 + 0.5);
            if(T > 256) T = 256;
            uint16_t T_inv = 256 - T;

            lerp_kernel<<<gridSize, cudaBlockSize>>>(d_F1.get(), d_F2.get(), d_out.get(), n, T, T_inv);
            CUDA_CHECK(cudaGetLastError());

            d_out.download(out.data.data());
            output.push_back(out);
        }// for times
    } catch (const std::exception& e) {
        std::cerr << "LERP CUDA failed: " << e.what() << std::endl;
        output.clear();
        return;
    }
}// lerp_cuda

void bicubic_cuda(const FrameSequence& input, int nFrames, int block_matching_block_size, int block_matching_search_window_size, FrameSequence& output, framegen::block_matching::BMVariant variant, int cuda_block_x, int cuda_block_y, BicubicVariant bicubic_variant) {
    
    if (input.size() != 4) {
        std::cerr << "Numero di Frames in input diverso da 4!" << std::endl;
        return;
    }
    if (nFrames < 1) {
        std::cerr << "Numero di Frames intermedi richiesti minore di 1!" << std::endl;
        return;
    }

    const int W = input[0].width;
    const int H = input[0].height;
    if (W <= 0 || H <= 0) {
        std::cerr << "Frames in input con dimensioni non valide!" << std::endl;
        return;
    }
    for (const Frame& f : input) {
        if (f.width != W || f.height != H) {
            std::cerr << "Frames in input di dimensione diversa!" << std::endl;
            return;
        }
        if (f.data.size() != (size_t)W * H) {
            std::cerr << "Frames in input con dati non validi!" << std::endl;
            return;
        }
    }

    if (!framegen::block_matching::bm_params_valid(W, H, block_matching_block_size,
            block_matching_search_window_size, cuda_block_x * cuda_block_y)) {
        return;   // l'helper ha già stampato il motivo
    }

    if (!framegen::block_matching::bm_variant_supported(variant, block_matching_block_size, block_matching_search_window_size)) {
        std::cerr << "Variante non supportata per shared memory troppo grande" << std::endl;
        return;
    }

    if (!bicubic_variant_supported(bicubic_variant, block_matching_block_size)) {
        std::cerr << "block_size troppo grande per la variante SharedTile (max 75)!" << std::endl;
        return;
    }

    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        std::cerr << "Nessun dispositivo CUDA disponibile!" << std::endl;
        return;
    }
    
    int n_pixels = W * H;
    int grid_w = (W + block_matching_block_size - 1) / block_matching_block_size;
    int grid_h = (H + block_matching_block_size - 1) / block_matching_block_size;
    int n_blocks = grid_w * grid_h;

    try {
        // Alloco in global memory i 4 frames in input, il frame di output e i 3 vettori di movimento da calcolare con il block_matching
        DeviceBuffer<Pixel> d_F0(n_pixels); 
        DeviceBuffer<Pixel> d_F1(n_pixels);
        DeviceBuffer<Pixel> d_F2(n_pixels);
        DeviceBuffer<Pixel> d_F3(n_pixels);
        DeviceBuffer<Pixel> d_out(n_pixels);
        DeviceBuffer<int2> d_d0(n_blocks);
        DeviceBuffer<int2> d_d2(n_blocks);
        DeviceBuffer<int2> d_d3(n_blocks);
        // Copio in global memory i frames
        d_F0.upload(input.at(0).data.data());
        d_F1.upload(input.at(1).data.data());
        d_F2.upload(input.at(2).data.data());
        d_F3.upload(input.at(3).data.data());

        // invoco il block_matching per calcolare i vettori spostamento di F1 verso F0, F2, F3.
        framegen::block_matching::BM_cuda_device(d_F1.get(), d_F0.get(), d_d0.get(), W, H, block_matching_block_size, block_matching_search_window_size, variant, cuda_block_x * cuda_block_y);
        framegen::block_matching::BM_cuda_device(d_F1.get(), d_F2.get(), d_d2.get(), W, H, block_matching_block_size, block_matching_search_window_size, variant, cuda_block_x * cuda_block_y);
        framegen::block_matching::BM_cuda_device(d_F1.get(), d_F3.get(), d_d3.get(), W, H, block_matching_block_size, block_matching_search_window_size, variant, cuda_block_x * cuda_block_y);
        
        int cuda_grid_x = (W + cuda_block_x - 1) / cuda_block_x;
        int cuda_grid_y = (H + cuda_block_y - 1) / cuda_block_y;
        dim3 cuda_block(cuda_block_x, cuda_block_y);
        dim3 cuda_grid(cuda_grid_x, cuda_grid_y);

        DeviceBuffer<BlockParams> d_params(bicubic_variant == BicubicVariant::Precomputed ? n_blocks : 1);
        const int threads_1d = cuda_block_x * cuda_block_y;     // per SharedTile e per il BM
        dim3 grid_bm(grid_w, grid_h);                           // un blocco CUDA per blocco di matching
        const size_t tile_bytes = 2 * (size_t)(block_matching_block_size + 3) * (block_matching_block_size + 3) * sizeof(Pixel);
        const int params_threads = 128;
        const int params_grid = ceil_div(n_blocks, params_threads);

        // calcolo ora sull'host i pesi di Catmull-Rom per ogni valore di t
        auto times = framegen::utils::make_times(nFrames);
        for (float t : times) {
            float w0, w2, w3;
            framegen::utils::temporalWeights(t, w0, w2, w3);

            switch (bicubic_variant) {
                case BicubicVariant::Baseline:
                    bicubic_kernel<<<cuda_grid, cuda_block>>>(d_F1.get(), d_F2.get(), d_d0.get(), d_d2.get(), d_d3.get(),
                                                            d_out.get(), t, w0, w2, w3, W, H,
                                                            block_matching_block_size, grid_w);
                    break;
                case BicubicVariant::Precomputed:
                    bicubic_params_kernel<<<params_grid, params_threads>>>(d_d0.get(), d_d2.get(), d_d3.get(),
                                                                        d_params.get(), n_blocks, w0, w2, w3);
                    bicubic_kernel_pre<<<cuda_grid, cuda_block>>>(d_F1.get(), d_F2.get(), d_params.get(),
                                                                d_out.get(), t, W, H,
                                                                block_matching_block_size, grid_w);
                    break;
                case BicubicVariant::SharedTile:
                    bicubic_kernel_tile<<<grid_bm, threads_1d, tile_bytes>>>(d_F1.get(), d_F2.get(),
                                                                            d_d0.get(), d_d2.get(), d_d3.get(),
                                                                            d_out.get(), t, w0, w2, w3, W, H,
                                                                            block_matching_block_size);
                    break;
            }
            CUDA_CHECK(cudaGetLastError());

            Frame out; out.width = W; out.height = H; out.data.resize(n_pixels);
            d_out.download(out.data.data());
            output.push_back(std::move(out));
        }// for times
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        output.clear();
        return;
    }
}// bicubic_cuda

}// namespace