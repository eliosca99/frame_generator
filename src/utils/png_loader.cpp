#include "../../include/utils/png_loader.hpp"
#include <iostream>
#include <png.h>

namespace framegen::utils {

namespace fs = std::filesystem;

std::optional<Frame> load_png(const fs::path& path) {
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

}//namespace framegen::utils