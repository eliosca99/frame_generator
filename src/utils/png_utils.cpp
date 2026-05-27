#include "../../include/utils/png_utils.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <png.h>

// implementazione delle funzioni di utilità

namespace framegen::utils {

namespace fs = std::filesystem;

std::optional<Frame> load_png(const fs::path& path) {
    // per caricare un frame da un file PNG, viene utilizzata la libreria libpng. 
    // la funzione restituisce un oggetto Frame se il caricamento è riuscito, altrimenti restituisce std::nullopt.
    
    if(!fs::exists(path)) {
        std::cerr << "File non trovato" << path << std::endl;
        return std::nullopt;
    }

    FILE* fp = fopen(path.string().c_str(), "rb");
    if(!fp) {
        std::cerr << "Impossibile aprire il file" << path << std::endl;
        return std::nullopt;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        fclose(fp);
        return std::nullopt;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, nullptr, nullptr);
        fclose(fp);
        return std::nullopt;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        return std::nullopt;
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    Frame frame;
    frame.width = png_get_image_width(png, info);
    frame.height = png_get_image_height(png, info);

    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth  = png_get_bit_depth(png, info);

    if (bit_depth == 16)
        png_set_strip_16(png);

    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);

    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);

    if (png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);

    png_read_update_info(png, info);
    frame.format = static_cast<PixelFormat>(png_get_channels(png, info));

    size_t row_bytes = png_get_rowbytes(png, info);
    frame.data.resize(frame.height * row_bytes);

    std::vector<png_bytep> row_pointers(frame.height);
    for (uint32_t y = 0; y < frame.height; ++y) {
        row_pointers[y] = frame.data.data() + (y * row_bytes);
    }

    png_read_image(png, row_pointers.data());

    png_destroy_read_struct(&png, &info, nullptr);
    fclose(fp);

    return frame;
}

std::optional<FrameSequence> load_png_sequence(const fs::path& dir) {
    if(!fs::exists(dir) || !fs::is_directory(dir)) {
        std::cerr << "Directory non trovata" << dir << std::endl;
        return std::nullopt;
    }

    FrameSequence sequence;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            if (auto frame = load_png(entry.path())) {
                sequence.push_back(frame.value());
            } else {
                std::cerr << "Impossibile caricare il file" << entry.path() << std::endl;
            }
        }
    }

    return sequence;
}

bool save_png(const Frame& frame, const fs::path& path) {
    if (frame.width <= 0 || frame.height <= 0 || frame.data.empty()) {
        std::cerr << "Frame non valido" << std::endl;
        return false;
    }

    if (path.empty()) {
        std::cerr << "Percorso non valido" << std::endl;
        return false;
    }

    fs::path parent = path.parent_path();
    if (!parent.empty() && !fs::exists(parent)) {
        if (!fs::create_directories(parent)) {
            std::cerr << "Impossibile creare la directory" << parent << std::endl;
            return false;
        }
    }

    FILE* fp = fopen(path.string().c_str(), "wb");
    if (!fp) {
        std::cerr << "Impossibile aprire il file" << path << std::endl;
        return false;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        fclose(fp);
        return false;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, nullptr);
        fclose(fp);
        return false;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return false;
    }

    int color_type = PNG_COLOR_TYPE_RGB;
    int channels = 3;
    switch (frame.format) {
        case PixelFormat::GRAY:
            color_type = PNG_COLOR_TYPE_GRAY;
            channels = 1;
            break;
        case PixelFormat::RGB:
            color_type = PNG_COLOR_TYPE_RGB;
            channels = 3;
            break;
        case PixelFormat::RGBA:
            color_type = PNG_COLOR_TYPE_RGBA;
            channels = 4;
            break;
        default:
            std::cerr << "Formato pixel non supportato" << std::endl;
            png_destroy_write_struct(&png, &info);
            fclose(fp);
            return false;
    }

    size_t row_bytes = static_cast<size_t>(frame.width) * static_cast<size_t>(channels);
    size_t expected_size = row_bytes * static_cast<size_t>(frame.height);
    if (frame.data.size() < expected_size) {
        std::cerr << "Dimensione dati non valida" << std::endl;
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return false;
    }

    png_init_io(png, fp);
    png_set_IHDR(
        png,
        info,
        frame.width,
        frame.height,
        8,
        color_type,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png, info);

    std::vector<png_bytep> row_pointers(static_cast<size_t>(frame.height));
    for (int y = 0; y < frame.height; ++y) {
        row_pointers[static_cast<size_t>(y)] = const_cast<png_bytep>(
            frame.data.data() + (static_cast<size_t>(y) * row_bytes)
        );
    }

    png_write_image(png, row_pointers.data());
    png_write_end(png, nullptr);

    png_destroy_write_struct(&png, &info);
    fclose(fp);

    return true;
}

bool save_png_sequence(const FrameSequence& sequence, const fs::path& dir, const std::string& prefix) {
    if (sequence.empty()) {
        std::cerr << "Sequenza vuota" << std::endl;
        return false;
    }

    if (dir.empty()) {
        std::cerr << "Directory non valida" << std::endl;
        return false;
    }

    if (fs::exists(dir)) {
        if (!fs::is_directory(dir)) {
            std::cerr << "Directory non valida" << dir << std::endl;
            return false;
        }
    } else if (!fs::create_directories(dir)) {
        std::cerr << "Impossibile creare la directory" << dir << std::endl;
        return false;
    }

    size_t pad = std::to_string(sequence.size()).size();
    if (pad < 4) {
        pad = 4;
    }

    for (size_t i = 0; i < sequence.size(); ++i) {
        std::ostringstream name;
        name << prefix << "_" << std::setw(static_cast<int>(pad)) << std::setfill('0') << i << ".png";
        fs::path out_path = dir / name.str();
        if (!save_png(sequence[i], out_path)) {
            std::cerr << "Impossibile salvare il file" << out_path << std::endl;
            return false;
        }
    }

    return true;
}

}//namespace framegen::utils