/*  Il block matching è un algoritmo che calcola i vettori spostamento (dx, dy) dei pixel dal Frame A al Frame B.
    Il Frame A viene suddiviso in blocchi NxN, si fa scorrere ogni blocco sul Frame B entro uno spostamento massimo
    e si salva il vettore (dx, dy) (in pixel) che portano secondo una certa metrica (SAD) alla massima similarità.
    Si ottiene quindi una matrice di int2, i cui elementi rappresentano i vettori spostamento di ogni blocco, 
    e ogni pixel del blocco condivide lo stesso vettore spostamento. 
*/

#include "../../include/sequential/block_matching.hpp"
#include <iostream>
#include <vector>

namespace framegen::block_matching {

std::optional<std::vector<int2>> block_matching(const Frame& frame1, const Frame& frame2, int block_size, int search_window_size) {
    // check validità dei parametri
    if (frame1.data.size() != frame2.data.size() || block_size > std::min(frame1.height, frame2.width)) {
        std::cerr << "Dati in input non validi!" << std::endl;
        return std::nullopt;
    }

    // calcolo la dimensione della griglia
    int grid_height = (frame1.height + block_size - 1) / block_size;
    int grid_width = (frame1.width + block_size -1) / block_size;
    std::vector<int2> mv_matrix;
    mv_matrix.reserve(grid_height * grid_width);
    
    // inizializzo il best sad, cioè la differenza tra i due blocchi tra i due frame. via via che scorro,
    // se trovo una corrispondenza migliore aggiorno questo valore e mi salvo gli indici del blocco
    
    // ciclo sui blocchi della griglia
    for (int grid_r = 0; grid_r < grid_height; grid_r++) {
        for (int grid_c = 0; grid_c < grid_width; grid_c++) {
            std::cout << grid_r << " " << grid_c << "\n" << std::endl;
            int best_sad = INT32_MAX;
            int2 best_vector = {0, 0};
            // sono in un blocco. ciclo per ogni elemento della finestra
            // calcolo gli spostamenti minimi e massimi (per gestire i bordi)
            int dxMin = std::max(-search_window_size, -block_size * grid_c);
            int dxMax = std::min(search_window_size, frame2.width - (block_size * (grid_c+1)));
            int dyMin = std::max(-search_window_size, -block_size * grid_r);
            int dyMax = std::min(search_window_size, frame2.height - (block_size * (grid_r+1)));
            for (int dy = dyMin; dy <= dyMax; dy++) {
                for (int dx = dxMin; dx <= dxMax; dx++) {
                    int cur_sad = 0;
                    // ora devo confrontare ogni pixel nel blocco nel frame1 con il corrispondente pixel nel frame2
                    for (int i = 0; i < block_size; i++) {
                        const Pixel* rowA = frame1.data.data()
                            + (grid_r * block_size + i) * frame1.width
                            + grid_c * block_size;

                        const Pixel* rowB = frame2.data.data()
                            + (grid_r * block_size + dy + i) * frame2.width
                            + (grid_c * block_size + dx);

                        for (int j = 0; j < block_size; j++) {
                            cur_sad += std::abs(rowA[j].r - rowB[j].r)
                                    + std::abs(rowA[j].g - rowB[j].g)
                                    + std::abs(rowA[j].b - rowB[j].b);
                        }

                        // early termination
                        if (cur_sad >= best_sad) break;
                    }   

                    if (cur_sad < best_sad) {
                        best_sad = cur_sad;
                        best_vector = {dx, dy};
                    }
                }
            }
            mv_matrix[grid_r * grid_width + grid_c] = best_vector;
        }
    }
}

}// namespace framegen::block_matching