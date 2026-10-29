#include "../include/sequential/interpolation.hpp"
#include "../include/utils/utils.hpp"
#include "../include/utils/bicubic_math.hpp"
#include "../include/sequential/block_matching.hpp"
#include <iostream>
#include <stdint.h>
#include <cmath>
#include <algorithm>

namespace framegen::interpolation {

    void lerp_sequential(const FrameSequence& input, int nFrames, FrameSequence& output) {
        // implementazione sequenziale dell'interpolazione con algoritmo lerp. Con il parametro nFrames, si indicano
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
        auto times = framegen::utils::make_times(nFrames);
        for (float t : times) {
            Frame out;
            out.height = F1.height;
            out.width = F1.width;
            int n = F1.height * F1.width;
            out.data.resize(static_cast<size_t>(n));

            // per ogni pixel del frame devo applicare a ogni canale la formula dell'algoritmo LERP:
            // P(out) = (1 - t) * P1 + t * P2.
            // invece di usare il float t, evito il casting dei pixel a float e poi di nuovo a uint8_t 
            // mappando t tra 0 e 256, che mi permette di fare alla fine uno shift di 8 bit e tornare al valore corretto
            uint16_t T = (uint16_t)(t * 256 + 0.5);
            if(T > 256) T = 256;
            uint16_t T_inv = 256 - T;
            
            // i pixel nel frame sono organizzati come Pixel RGBA, quindi basta iterare su ogni Pixel
            for (int i = 0; i < n; i++) {
                const Pixel& p1 = F1.data[static_cast<size_t>(i)];
                const Pixel& p2 = F2.data[static_cast<size_t>(i)];
                Pixel out_pixel;
                out_pixel.r = static_cast<uint8_t>(((uint32_t)p1.r * T_inv + (uint32_t)p2.r * T + 128) >> 8);
                out_pixel.g = static_cast<uint8_t>(((uint32_t)p1.g * T_inv + (uint32_t)p2.g * T + 128) >> 8);
                out_pixel.b = static_cast<uint8_t>(((uint32_t)p1.b * T_inv + (uint32_t)p2.b * T + 128) >> 8);
                out_pixel.a = static_cast<uint8_t>(((uint32_t)p1.a * T_inv + (uint32_t)p2.a * T + 128) >> 8);
                out.data[static_cast<size_t>(i)] = out_pixel;
            }

            // calcolato il valore di ogni canale di ogni pixel, basta inserire il frame nel Framesequence di output
            output.push_back(out);
        }
    }// lerp_sequential

    void bicubic_sequential(const FrameSequence& input, int nFrames, int block_matching_block_size, int block_matching_search_window_size, FrameSequence& output) {
        // implementazione sequenziale dell'algoritmo bicubic. Anche in questo caso, in base al numero
        // nFrames verranno generati nFrames intermedi distribuiti uniformemente tra 0.0 e 1.0.
        // L'algoritmo bicubic ha bisogno in input di 4 frames, in instanti di tempo t(-1), t(0), t(1), t(2)
        // e stima il o i frames intermedi tra t(0) e t(1). Tale algoritmo cattura anche l'accelerazione del moto.
        // Si basa sulla formulazione dei pesi di Catmull-Rom. 
        // NOTA: i frames intermedi vengono calcolati tra F1 e F2.

        // check di errori nei parametri
        if(input.size() != 4) {
            std::cerr << "Numero di Frames in input diverso da 4!" << std::endl;
            return;
        }
        if(nFrames < 1) {
            std::cerr << "Numero di Frames intermedi richiesti minore di 1!" << std::endl;
            return;
        }

        const Frame& F0 = input.at(0);
        const Frame& F1 = input.at(1);
        const Frame& F2 = input.at(2);
        const Frame& F3 = input.at(3);
        
        if(F0.height <= 0 || F0.width <= 0 ||
           F1.height <= 0 || F1.width <= 0 ||
           F2.height <= 0 || F2.width <= 0 ||
           F3.height <= 0 || F3.width <= 0) {
            std::cerr << "Frames in input con dimensioni non valide!" << std::endl;
            return;
        }
        if(F0.height != F1.height || F0.height != F2.height || F0.height != F3.height ||
           F0.width != F1.width || F0.width != F2.width || F0.width != F3.width) {
            std::cerr << "Frames in input di dimensione diversa!" << std::endl;
            return;
        }
        size_t expected_size = static_cast<size_t>(F1.height) * static_cast<size_t>(F1.width);
        if(F0.data.size() != expected_size || F1.data.size() != expected_size ||
           F2.data.size() != expected_size || F3.data.size() != expected_size) {
            std::cerr << "Frames in input con dati non validi!" << std::endl;
            return;
        }

        // applico il block matching da F1 a F0, da F1 a F2, da F1 a F3, che restituisce
        // le matrici di movimento dal Frame F1 agli altri, d0, d2, d3.
        using framegen::utils::int2;
        using framegen::block_matching::block_matching;
        std::optional<std::vector<int2>> d0 = block_matching(F1, F0, block_matching_block_size, block_matching_search_window_size);
        std::optional<std::vector<int2>> d2 = block_matching(F1, F2, block_matching_block_size, block_matching_search_window_size);
        std::optional<std::vector<int2>> d3 = block_matching(F1, F3, block_matching_block_size, block_matching_search_window_size);
        if (!d0 || !d2 || !d3) {
            std::cerr << "Impossibile calcolare i vettori di movimento!" << std::endl;
            return;
        }

        auto times = framegen::utils::make_times(nFrames);
        for (float t : times) {
            // applico la formula dei pesi di Catmull-Rom per la cubica temporale. Dato che si parte da F1,
            // i pesi saranno 3 come le matrici di movimento, ovviamente perche il movimento da F1 a F1 è nullo
            // i pesi temporali sono calcolati solo in base a t, quindi sono comuni a tutti i blocchi.
            // Indicano quanto peso dare allo spostamento da F1 verso un altro frame.
            float w0, w2, w3;
            framegen::utils::temporalWeights(t, w0, w2, w3);

            Frame out;
            out.width = F1.width;
            out.height = F1.height;
            out.data.resize(expected_size);
            output.push_back(std::move(out));

            // dimensioni della griglia di blocchi
            int grid_w = (F1.width + block_matching_block_size - 1) / block_matching_block_size;
            int grid_h = (F1.height + block_matching_block_size - 1) / block_matching_block_size;

            // ora ciclo sui blocchi
            for (int by = 0; by < grid_h; by ++) {
                for (int bx = 0; bx < grid_w; bx ++) {
                    /*  per ogni blocco, devo calcolare i seguenti parametri:
                        - D(b)              Spostamento del blocco b da F1 a F2
                        - off1[b]	        Di quanto spostarsi dal pixel di output per leggere in F1
                        - off2[b]	        Di quanto spostarsi dal pixel di output per leggere in F2
                        - ix1, iy1	        Parte intera di off1 (x e y): spostamento in pixel interi per F1
                        - ix2, iy2	        Parte intera di off2: spostamento in pixel interi per F2
                        - fx1, fy1	        Parte frazionaria di off1: dove cade il punto dentro il pixel in F1
                        - fx2, fy2	        Parte frazionaria di off2: lo stesso per F2
                        - wFx0[4], wFy0[4]	Pesi cubici spaziali (lungo x e lungo y) per campionare F1, calcolati da fx0, fy0
                        - wFx1[4], wFy1[4]	Pesi cubici spaziali per campionare F2, calcolati da fx1, fy1
                    */
                    int block = by * grid_w + bx;
                    int2 d0_b = d0.value().at(block);
                    int2 d2_b = d2.value().at(block);
                    int2 d3_b = d3.value().at(block);

                    float Dx = w0*d0_b.x + w2*d2_b.x + w3*d3_b.x;
                    float Dy = w0*d0_b.y + w2*d2_b.y + w3*d3_b.y;

                    int ixF1, iyF1, ixF2, iyF2;
                    float wxF1[4], wyF1[4], wxF2[4], wyF2[4];

                    framegen::utils::spatialWeights(     -Dx, ixF1, wxF1);
                    framegen::utils::spatialWeights(     -Dy, iyF1, wyF1);
                    framegen::utils::spatialWeights(d2_b.x - Dx, ixF2, wxF2);
                    framegen::utils::spatialWeights(d2_b.y - Dy, iyF2, wyF2);

                    // calcolati tutti i parametri relativi al blocco, ciclo sui pixel del blocco
                    int xStart = bx * block_matching_block_size;
                    int yStart = by * block_matching_block_size;
                    int xEnd = std::min(xStart + block_matching_block_size, F1.width);
                    int yEnd = std::min(yStart + block_matching_block_size, F1.height);
                    for (int y = yStart; y < yEnd; y++) {
                        for (int x = xStart; x < xEnd; x++) {
                            framegen::utils::Color c1 = framegen::utils::sample(F1.data.data(), F1.width, F1.height, x + ixF1, y + iyF1, wxF1, wyF1);
                            framegen::utils::Color c2 = framegen::utils::sample(F2.data.data(), F2.width, F2.height, x + ixF2, y + iyF2, wxF2, wyF2);
                            // campionati i due pixel, eseguo il blending dei colori
                            float r = ((1.f - t) * c1.r) + (t * c2.r);
                            float g = ((1.f - t) * c1.g) + (t * c2.g);
                            float b = ((1.f - t) * c1.b) + (t * c2.b);
                            float a = ((1.f - t) * c1.a) + (t * c2.a);
                            
                            Pixel o;
                            o.r = (uint8_t)(std::min(std::max(r, 0.f), 255.f) + 0.5);
                            o.g = (uint8_t)(std::min(std::max(g, 0.f), 255.f) + 0.5);
                            o.b = (uint8_t)(std::min(std::max(b, 0.f), 255.f) + 0.5);
                            o.a = (uint8_t)(std::min(std::max(a, 0.f), 255.f) + 0.5);
                            output.back().data[y * F1.width + x] = o;
                        }
                    }
                }// for bx
            }// for by
        }// for t in times
    }// bicubic_sequential
}// namespace framegen::interpolation