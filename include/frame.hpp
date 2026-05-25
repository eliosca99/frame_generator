#ifndef FRAME_HPP
#define FRAME_HPP

// definizione della struct Frame e del tipo FrameSequence, che rappresenta una sequenza di frame.
// La struct Frame contiene un vettore di byte (data) che rappresenta i dati dell'immagine,
// le dimensioni dell'immagine (width e height) e il formato dei pixel (format).
// i frame caricati verranno memorizzati in tale struct

#include <cstdint>
#include <vector>

namespace framegen { // definisco un namespace per evitare conflitti di nomi

enum class PixelFormat: uint8_t {
    GRAY = 1,
    RGB = 3,
    RGBA = 4
};

struct Frame {
    std::vector<uint8_t> data;
    int width;
    int height;
    PixelFormat format;
};

using FrameSequence = std::vector<Frame>; // per usare una sequenza di frame, simile al typedef

}//namespace framegen
#endif