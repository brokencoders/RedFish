#pragma once 
#include <iostream>
#include <chrono>
#include <thread>
#include "Tensor.h"

#include "gnuplot-iostream.h"

namespace RedFish
{
    static Gnuplot gp;

    inline void plot_clear()
    {
        gp << "clear" << std::endl;
    }

    inline void plot_RGBimage(const Tensor& batch, size_t index = 0, size_t size_data = 1, size_t plot_height = 0, size_t plot_width = 0)
    {
        size_t channels = 3;
        std::vector<std::vector<std::tuple<uint8_t, uint8_t, uint8_t>>> F;

        size_t width  = batch.getShape()[3];
        size_t height = batch.getShape()[2];

        for (size_t i = 0; i < width; i++)
        {
            std::vector<std::tuple<uint8_t, uint8_t, uint8_t>> F_;
            for (size_t j = 0; j < height; j++)
                F_.insert(F_.begin(), { batch(index, 0, j, i), batch(index, 1, j, i), batch(index, 2, j, i)});
            F.push_back(F_);
        }

        gp << "width = " << ((plot_width == 0) ? width : plot_width) << std::endl;
        gp << "height = " << ((plot_height == 0) ? height : plot_height) << std::endl;
        gp << "set xrange [0:width-1]" << std::endl;
        gp << "set yrange [0:height-1]" << std::endl;

        std::string s = "";
        for (size_t i = 0; i < size_data * channels; i++)
            s += "\%uchar"; 
        
        gp << "plot '-'  binary array=" << width << "x" << height << " format='" << s << "' with rgbimage title 'RGB Image'\n";
        gp.sendBinary(F);
    }

    inline void plot_RGBAimage(const Tensor& batch,  size_t index = 0, size_t size_data = 1, size_t plot_height = 0, size_t plot_width = 0)
    {
        size_t channels = 4;
        std::vector<std::vector<std::tuple<uint8_t, uint8_t, uint8_t, uint8_t>>> F;

        size_t width  = batch.getShape()[3];
        size_t height = batch.getShape()[2];

        for (size_t i = 0; i < width; i++)
        {
            std::vector<std::tuple<uint8_t, uint8_t, uint8_t, uint8_t>> F_;
            for (size_t j = 0; j < height; j++)
                F_.insert(F_.begin(), { batch(index, 0, j, i), batch(index, 1, j, i), batch(index, 2, j, i), batch(index, 3, j, i)});
            F.push_back(F_);
        }

        gp << "width = " << ((plot_width == 0) ? width : plot_width) << std::endl;
        gp << "height = " << ((plot_height == 0) ? height : plot_height) << std::endl;
        gp << "set xrange [0:width-1]" << std::endl;
        gp << "set yrange [0:height-1]" << std::endl;

        std::string s = "";
        for (size_t i = 0; i < size_data * channels; i++)
            s += "\%uchar"; 
        
        gp << "plot '-'  binary array=" << width << "x" << height << " format='" << s << "' with rgbimage title 'RGBA Image'\n";
        gp.sendBinary(F);
    }

    inline void plot_image(const Tensor& batch, size_t channels = 3, size_t index = 0, size_t size_data = 1, size_t plot_height = 0, size_t plot_width = 0)
    {
        if(channels == 3)
            plot_RGBimage(batch, index, size_data, plot_height, plot_width);
        else if(channels == 4)
            plot_RGBAimage(batch, index, size_data, plot_height, plot_width);
        else 
            throw std::runtime_error("Only 3 and 4 channel support");
    }

}