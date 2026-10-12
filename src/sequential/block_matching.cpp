/*  Il block matching è un algoritmo che calcola i vettori spostamento (dx, dy) dei pixel dal Frame A al Frame B.
    Il Frame A viene suddiviso in blocchi NxN, si fa scorrere ogni blocco sul Frame B entro uno spostamento massimo
    e si salva il vettore (dx, dy) (in pixel) che portano secondo una certa metrica (SAD) alla massima similarità.
    Si ottiene quindi una matrice di int2, i cui elementi rappresentano i vettori spostamento di ogni blocco, 
    e ogni pixel del blocco condivide lo stesso vettore spostamento. 
*/

#include "../../include/sequential/block_matching.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstdint>

namespace framegen::block_matching {

std::optional<std::vector<int2>> block_matching(const Frame& frame1, const Frame& frame2, int block_size, int search_window_size) {
    // check validità dei parametri
    if (frame1.width != frame2.width || frame1.height != frame2.height ||
        frame1.data.size() != (size_t)frame1.width * frame1.height ||
        frame2.data.size() != frame1.data.size() ||
        block_size <= 0 || search_window_size < 0 ||
        block_size > std::min(frame1.width, frame1.height)) {
        std::cerr << "Dati in input non validi!" << std::endl;
        return std::nullopt;
    }

    // calcolo la dimensione della griglia
    // (arrotondamento per eccesso: nessun pixel resta fuori, l'ultimo blocco di riga/colonna può essere parziale)
    int grid_height = (frame1.height + block_size - 1) / block_size;
    int grid_width = (frame1.width + block_size - 1) / block_size;

    std::vector<int2> mv_matrix(grid_height * grid_width);

    // ciclo sui blocchi della griglia
    for (int grid_r = 0; grid_r < grid_height; grid_r++) {
        for (int grid_c = 0; grid_c < grid_width; grid_c++) {

            // dimensione effettiva del blocco (ridotta sui bordi destro e inferiore)
            int block_h = std::min(block_size, frame1.height - grid_r * block_size);
            int block_w = std::min(block_size, frame1.width - grid_c * block_size);

            // spostamenti minimi e massimi (per gestire i bordi)
            // per un blocco parziale dxMax/dyMax valgono 0: è già attaccato al bordo destro/inferiore
            int dxMin = std::max(-search_window_size, -block_size * grid_c);
            int dxMax = std::min(search_window_size, frame2.width - (grid_c * block_size + block_w));
            int dyMin = std::max(-search_window_size, -block_size * grid_r);
            int dyMax = std::min(search_window_size, frame2.height - (grid_r * block_size + block_h));

            // lambda che calcola il SAD del blocco corrente per il candidato (dx, dy).
            // "limit" serve per l'early termination: se la somma parziale raggiunge
            // il miglior SAD trovato finora, il candidato non può vincere e mi fermo.
            auto sad = [&](int dx, int dy, int limit) {
                int cur = 0;
                for (int i = 0; i < block_h; i++) {
                    const Pixel* rowA = frame1.data.data()
                        + (grid_r * block_size + i) * frame1.width
                        + grid_c * block_size;

                    const Pixel* rowB = frame2.data.data()
                        + (grid_r * block_size + dy + i) * frame2.width
                        + (grid_c * block_size + dx);

                    for (int j = 0; j < block_w; j++) {
                        cur += std::abs(rowA[j].r - rowB[j].r)
                             + std::abs(rowA[j].g - rowB[j].g)
                             + std::abs(rowA[j].b - rowB[j].b);
                    }

                    // early termination
                    if (cur >= limit) break;
                }
                return cur;
            };

            // parto dal vettore nullo come candidato di default: a parità di SAD
            // (zone piatte) vince (0,0) invece del primo vettore scandito.
            // Lo calcolo per intero (limit = INT32_MAX) perché mi serve il valore completo.
            int best_sad = sad(0, 0, INT32_MAX);
            int2 best_vector = {0, 0};

            // scandisco tutti gli altri candidati, saltando (0,0) già valutato.
            // Accetto solo se strettamente migliore.
            for (int dy = dyMin; dy <= dyMax; dy++) {
                for (int dx = dxMin; dx <= dxMax; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    int s = sad(dx, dy, best_sad);
                    if (s < best_sad) {
                        best_sad = s;
                        best_vector = {dx, dy};
                    }
                }
            }

            mv_matrix[grid_r * grid_width + grid_c] = best_vector;
        }
    }
    return mv_matrix;
}

}// namespace framegen::block_matching