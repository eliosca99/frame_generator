#include "../include/sequential/processing.hpp"
#include "../include/utils/utils.hpp"
#include <iostream>
#include <stdint.h>

namespace framegen::processing {

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
        if(F1.height != F2.height || F1.width != F2.width || F1.format != F2.format) {
            std::cerr << "Frames in input di dimensione o formato diversi!" << std::endl;
            return;
        }
        auto times = framegen::utils::make_times(nFrames);
        for (float t : times) {
            Frame out;
            out.format = F1.format;
            out.height = F1.height;
            out.width = F1.width;
            int n = F1.height * F1.width * (int)F1.format;
            out.data.resize(n);

            // per ogni pixel del frame devo applicare a ogni canale la formula dell'algoritmo LERP:
            // P(out) = (1 - t) * P1 + t * P2.
            // invece di usare il float t, evito il casting dei pixel a float e poi di nuovo a uint8_t 
            // mappando t tra 0 e 256, che mi permette di fare alla fine uno shift di 8 bit e tornare al valore corretto
            uint16_t T = (uint16_t)(t * 256 + 0.5);
            if(T > 256) T = 256;
            uint16_t T_inv = 256 - T;
            
            // i pixel nel frame non sono suddivisi, quindi basta fare un for su ogni uint8_t del frame
            for (int i = 0; i < n; i++) {
                out.data[i] = (uint8_t)(((uint32_t)F1.data[i] * T_inv + (uint32_t)F2.data[i] * T + 128) >> 8);
            }

            // calcolato il valore di ogni canale di ogni pixel, basta inserire il frame nel Framesequence di output
            output.push_back(out);
        }
    }// lerp_sequential

}// namespace framegen::utils