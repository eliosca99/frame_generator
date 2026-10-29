#include "../../include/parallel/interpolation_cuda.hpp" 
#include "../../include/utils/utils.hpp"
#include "../../include/utils/bicubic_math.hpp"
#include "../../include/frame.hpp"
#include "../../include/parallel/cuda_utils.cuh"
#include "../../include/parallel/block_matching_cuda.hpp"

#include <iostream>

namespace framegen::interpolation {
using framegen::utils::int2;

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

__global__ void bicubic_kernel(const Pixel* d_F1, const Pixel* d_F2, int2* d_d0, int2* d_d2, int2* d_d3, Pixel* d_out, int t, float w0, float w2, float w3, int W, int H, int block_matching_block_size, int grid_w) {
    // un thread per pixel di output.
    // ricavo prima il blocco di matching in cui il thread si trova
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int block = ((y / block_matching_block_size) * grid_w) * (x / block_matching_block_size);

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

    if(cudaBlockSize <= 0 || cudaBlockSize > 1024) {
        std::cerr << "cudaBlockSize non valido!" << std::endl;
        return;
    }
    
    size_t n = F1.height * F1.width;
    int gridSize = (n + cudaBlockSize - 1) / cudaBlockSize;
    DeviceBuffer<Pixel> d_F1(n);
    DeviceBuffer<Pixel> d_F2(n);
    DeviceBuffer<Pixel> d_out(n);
    d_F1.upload(F1.data.data());
    d_F2.upload(F2.data.data());
    d_out.upload(F1.data.data());


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

        d_out.download(out.data.data());
        output.push_back(out);
    }// for times
}// lerp_cuda

void bicubic_cuda(const FrameSequence& input, int nFrames, int block_matching_block_size, int block_matching_search_window_size, FrameSequence& output, int cuda_block_x, int cuda_block_y) {
    
    int W = input.at(0).width; int H = input.at(0).height;
    int n_pixels = W * H;
    int grid_w = (W + block_matching_block_size - 1) / block_matching_block_size;
    int grid_h = (H + block_matching_block_size - 1) / block_matching_block_size;
    int n_blocks = grid_w * grid_h;

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
    framegen::block_matching::BM_cuda_device(d_F1.get(), d_F0.get(), d_d0.get(), W, H, block_matching_block_size, block_matching_search_window_size, cuda_block_x * cuda_block_y);
    framegen::block_matching::BM_cuda_device(d_F1.get(), d_F2.get(), d_d2.get(), W, H, block_matching_block_size, block_matching_search_window_size, cuda_block_x * cuda_block_y);
    framegen::block_matching::BM_cuda_device(d_F1.get(), d_F3.get(), d_d3.get(), W, H, block_matching_block_size, block_matching_search_window_size, cuda_block_x * cuda_block_y);
    
    int cuda_grid_x = (W + cuda_block_x - 1) / cuda_block_x;
    int cuda_grid_y = (H + cuda_block_y - 1) / cuda_block_y;
    dim3 cuda_block(cuda_block_x, cuda_block_y);
    dim3 cuda_grid(cuda_grid_x, cuda_grid_y);

    // calcolo ora sull'host i pesi di Catmull-Rom per ogni valore di t
    auto times = framegen::utils::make_times(nFrames);
    for(int t : times) {
        // calcolo i pesi temporali, che dipendono solo dal tempo t
        float w0, w2, w3;
        framegen::utils::temporalWeights(t, w0, w2, w3);
        
        bicubic_kernel<<<cuda_grid, cuda_block>>>(d_F1.get(), d_F2.get(), d_d0.get(), d_d2.get(), d_d3.get(), d_out.get(), t, w0, w2, w3, W, H, block_matching_block_size, grid_w);
        CUDA_CHECK(cudaGetLastError());
        Frame out; out.width = W; out.height = H; out.data.resize(n_pixels);
        d_out.download(out.data.data());
        output.push_back(std::move(out));
    }// for times

}// bicubic_cuda

}// namespace