#include "../../include/parallel/block_matching_cuda.hpp"
#include <iostream>
#include <climits>

namespace framegen::block_matching {

__global__ void BM_kernel(const Pixel* d_F1, const Pixel* d_F2, int2* d_out, int W, int H, int block_size, int search_window_size) {
    extern __shared__ Pixel tile[];
    __shared__ unsigned long long best_sad;
    
    int grid_c = blockIdx.x;
    int grid_r = blockIdx.y;
    int x0 = grid_c * block_size;
    int y0 = grid_r * block_size;
    int block_w = min(block_size, W - x0);
    int block_h = min(block_size, H - y0);
    int dxMin = max(-search_window_size, -block_size * grid_c);
    int dxMax = min(search_window_size, W - (grid_c * block_size + block_w));
    int dyMin = max(-search_window_size, -block_size * grid_r);
    int dyMax = min(search_window_size, H - (grid_r * block_size + block_h));

    if(threadIdx.x == 0) best_sad = ULLONG_MAX;

    // popolo la shared memory, che per ogni blocco cuda conterrà il blocco corrispondente dell'immagine
    for(int i = threadIdx.x; i < block_w * block_h; i += blockDim.x) {
        int r = i / block_w;
        int c = i % block_w;
        tile[r * block_w + c] = d_F1[(x0 + c) + ((y0 + r) * W)]; 
    }
    __syncthreads();

    // ogni thread calcola il SAD per uno o piu spostamenti (dx, dy) all'interno della finestra di ricerca
    unsigned long long local_best_sad = ULLONG_MAX;
    int nx = dxMax - dxMin + 1;
    int total = nx * (dyMax - dyMin + 1);
    for (int k = threadIdx.x; k < total; k += blockDim.x) {
        const int dx = dxMin + k % nx;
        const int dy = dyMin + k / nx;

        unsigned int sad = 0;
        for (int i = 0; i < block_h; i++) {
            const Pixel* rowA = tile + i * block_w;
            const Pixel* rowB = d_F2 + (y0 + dy + i) * W + (x0 + dx);
            for (int j = 0; j < block_w; j++) {
                sad += abs(rowA[j].r - rowB[j].r)
                     + abs(rowA[j].g - rowB[j].g)
                     + abs(rowA[j].b - rowB[j].b);
            }
        }

        const unsigned int priority = (dx == 0 && dy == 0) ? 0u : (unsigned int)(k + 1);
        const unsigned long long key = ((unsigned long long)sad << 32) | priority;
        if (key < local_best_sad) local_best_sad = key;
    }

    atomicMin(&best_sad, local_best_sad);
    __syncthreads();

    if (threadIdx.x == 0) {
        const unsigned int priority = (unsigned int)(best_sad & 0xFFFFFFFFull);
        int bdx = 0, bdy = 0;
        if (priority != 0) {
            const int k = (int)priority - 1;
            bdx = dxMin + k % nx;
            bdy = dyMin + k / nx;
        }
        int2 v;
        v.x = bdx;
        v.y = bdy;
        d_out[grid_r * gridDim.x + grid_c] = v;
    }

}// BM_kernel

std::optional<std::vector<int2>> BM_cuda(const Frame& frame1, const Frame& frame2, int block_size, int search_window_size, int cuda_block_size) {
    // check validità dei parametri
    if (frame1.width != frame2.width || frame1.height != frame2.height ||
        frame1.data.size() != (size_t)frame1.width * frame1.height ||
        frame2.data.size() != frame1.data.size() ||
        block_size <= 0 || search_window_size < 0 ||
        block_size > min(frame1.width, frame1.height)) {
        std::cerr << "Dati in input non validi!" << std::endl;
        return std::nullopt;
    }

    int grid_w = (frame1.width + block_size - 1) / block_size;
    int grid_h = (frame1.height + block_size - 1) / block_size;
    int n_pixel = frame1.width * frame1.height;

    dim3 cuda_grid = dim3(grid_w, grid_h);
    int sm_bytes = block_size * block_size * sizeof(Pixel);

    DeviceBuffer<Pixel> d_F1(n_pixel);
    DeviceBuffer<Pixel> d_F2(n_pixel);
    DeviceBuffer<int2> d_out(grid_w * grid_h);
    d_F1.upload(frame1.data.data());
    d_F2.upload(frame2.data.data());

    BM_kernel<<<cuda_grid, cuda_block_size, sm_bytes>>>(d_F1.get(), d_F2.get(), d_out.get(), frame1.width, frame1.height, block_size, search_window_size);

    CUDA_CHECK(cudaGetLastError());

    std::vector<int2> out;
    d_out.download(out.data());
    return out;

}// BM_cuda

void BM_cuda_device(const Pixel* d_A, const Pixel* d_B, int2* d_out, int width, int height, int block_size, int search_window_size, int cuda_block_size) {
    int grid_w = (width + block_size - 1) / block_size;
    int grid_h = (height + block_size - 1) / block_size;

    dim3 cuda_grid = dim3(grid_w, grid_h);
    int sm_bytes = block_size * block_size * sizeof(Pixel);

    BM_kernel<<<cuda_grid, cuda_block_size, sm_bytes>>>(d_A, d_B, d_out, width, height, block_size, search_window_size);

    CUDA_CHECK(cudaGetLastError());
}

}// namespace