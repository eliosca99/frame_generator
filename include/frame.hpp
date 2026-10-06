#ifndef FRAME_HPP
#define FRAME_HPP

// definizione della struct Frame e del tipo FrameSequence, che rappresenta una sequenza di frame.
// La struct Frame contiene un vettore di Pixel (data) che rappresenta i dati dell'immagine
// in formato RGBA (8 bit per canale) e le dimensioni dell'immagine (width e height).
// i frame caricati verranno memorizzati in tale struct

#include <cstdint>
#include <vector>

namespace framegen { // definisco un namespace per evitare conflitti di nomi

struct Pixel {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
};

struct Frame {
    std::vector<Pixel> data;
    int width;
    int height;
};

using FrameSequence = std::vector<Frame>; // per usare una sequenza di frame, simile al typedef

}//namespace framegen
#endif