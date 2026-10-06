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

    // il block matching suddivide il frame1 in blocchi, poi scorre il blocco all'interno della finestra nel frame2
    // e confronta i pixel. in base alla corrispondenza migliore (seguendo una metrica) definisce il vettore
    // di movimento di quel blocco

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
            int best_sad = INT32_MAX;
            int2 best_vector;
            // sono in un blocco. ciclo per ogni elemento della finestra
            for (int dx = -search_window_size; dx < search_window_size; dx++) {
                for (int dy = -search_window_size; dy < search_window_size; dy++) {
                    int cur_sad = 0;
                    // ora devo confrontare ogni pixel nel blocco nel frame1 con il corrispondente pixel nel frame2
                    for (int i = 0; i < frame1.height; i++) {
                        for (int j = 0; j < frame1.width; j++) {
                            
                        }
                    }

                }
            }
        }
    }
}


}// namespace framegen::block_matching