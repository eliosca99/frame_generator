#include "../../include/parallel/interpolation_cuda.hpp" 
#include "../../include/utils/utils.hpp"
#include "../../include/frame.hpp"
#include "../../include/parallel/cuda_utils.cuh"

#include <iostream>

namespace framegen::interpolation {

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

}// namespace

