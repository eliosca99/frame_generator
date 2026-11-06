#ifndef INTERPOLATION_CUDA_HPP
#define INTERPOLATION_CUDA_HPP

#include "../frame.hpp"
#include "../../include/parallel/block_matching_cuda.hpp"

namespace framegen::interpolation {

// Baseline: un thread per pixel, parametri ricalcolati da ogni thread
// Precomputed: un kernel calcola i parametri per blocco, poi un kernel per pixel li legge
// SharedTile: un blocco CUDA per blocco di matching, sorgente in shared memory
enum class BicubicVariant { Baseline, Precomputed, SharedTile };

void lerp_cuda(const FrameSequence& input, int nFrames, FrameSequence& output, int cudaBlockSize = 256);

void bicubic_cuda(const FrameSequence& input, int nFrames,
                  int block_matching_block_size, int block_matching_search_window_size,
                  FrameSequence& output,
                  framegen::block_matching::BMVariant variant,
                  int cuda_block_x = 32, int cuda_block_y = 8,
                  BicubicVariant bicubic_variant = BicubicVariant::Baseline);

// true se la variante è utilizzabile con quel block_size (limite di shared memory)
bool bicubic_variant_supported(BicubicVariant v, int block_matching_block_size);

}

#endif