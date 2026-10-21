#include "../../include/parallel/block_matching_cuda.hpp"
#include <iostream>

namespace framegen::block_matching {

__global__ void BM_kernel(Pixel* d_F1, Pixel* d_F2, int2* d_out, int W, int H, int block_size, int search_window_size, int cuda_block_size) {
    extern __shared__ Pixel tile[];
    __shared__ unsigned long long best_sad;
    
    int grid_c = blockIdx.x;
    int grid_r = blockIdx.y;
    int x0 = grid_c * block_size;
    int y0 = grid_r * block_size;
    int block_w = min(block_size, W - x0);
    int block_h = min(block_size, H - y0);
    int dxMin = std::max(-search_window_size, -block_size * grid_c);
    int dxMax = std::min(search_window_size, W - (grid_c * block_size + block_w));
    int dyMin = std::max(-search_window_size, -block_size * grid_r);
    int dyMax = std::min(search_window_size, H - (grid_r * block_size + block_h));

    if(threadIdx.x == 0) best_sad = LLONG_MAX;
    __syncthreads();

    // popolo la shared memory, che per ogni blocco cuda conterrà il blocco corrispondente dell'immagine
    for(int i = threadIdx.x; i < block_w * block_h; i += blockDim.x) {
        int r = i / block_w;
        int c = i % block_w;
        tile[r * block_w + c] = d_F1[(x0 + c) + ((y0 + r) * W)]; 
    }
    __syncthreads();



}// BM_kernel

std::optional<std::vector<int2>> BM_cuda(const Frame& frame1, const Frame& frame2, int block_size, int search_window_size, int cuda_block_size) {
    // check validità dei parametri
    if (frame1.width != frame2.width || frame1.height != frame2.height ||
        frame1.data.size() != (size_t)frame1.width * frame1.height ||
        frame2.data.size() != frame1.data.size() ||
        block_size <= 0 || search_window_size < 0 ||
        block_size > std::min(frame1.width, frame1.height)) {
        std::cerr << "Dati in input non validi!" << std::endl;
        return std::nullopt;
    }

    int grid_w = (frame1.width + block_size - 1) / block_size;
    int grid_h = (frame1.height + block_size - 1) / block_size;
    int n_pixel = frame1.width * frame1.height;

    dim3 cuda_grid = (grid_w, grid_h);
    int sm_bytes = cuda_block_size * cuda_block_size * sizeof(Pixel);

    DeviceBuffer<Pixel> d_F1(n_pixel);
    DeviceBuffer<Pixel> d_F2(n_pixel);
    DeviceBuffer<int2> d_out(grid_w * grid_h);
    d_F1.upload(frame1.data.data());
    d_F2.upload(frame2.data.data());

    BM_kernel<<<cuda_grid, cuda_block_size, sm_bytes>>>(d_F1.get(), d_F2.get(), d_out.get(), frame1.width, frame1.height, block_size, search_window_size, cuda_block_size);

    std::vector<int2> out;
    d_out.download(out.data());
    return out;

}// BM_cuda


}// namespace