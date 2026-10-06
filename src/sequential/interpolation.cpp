#include "../include/sequential/interpolation.hpp"
#include "../include/utils/utils.hpp"
#include <iostream>
#include <stdint.h>

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

    void bicubic_sequential(const FrameSequence& input, int nFrames, FrameSequence& output) {
        // implementazione sequenziale dell'algoritmo bicubic. Anche in questo caso, in base al numero
        // nFrames verranno generati nFrames intermedi distribuiti uniformemente tra 0.0 e 1.0.
        // L'algoritmo bicubic ha bisogno in input di 4 frames, in instanti di tempo t(-1), t(0), t(1), t(2)
        // e stima il o i frames intermedi tra t(0) e t(1). Tale algoritmo cattura anche l'accelerazione del moto.
        // Si basa sulla formulazione di Catmull-Rom


    }

}// namespace framegen::interpolation