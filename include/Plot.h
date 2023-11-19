#pragma once 
#include <iostream>
#include <chrono>
#include <thread>
#include "Tensor.h"

#include "../lib/gnuplot/gnuplot-iostream.h"

namespace RedFish
{
    // Specify image size and format
    void plot_image(const Tensor& batch, size_t index)
    {
        Gnuplot gp("gnuplot -persist");
        std::vector<std::vector<std::tuple<uint8_t, uint8_t, uint8_t>>> F;

        for (size_t i = 0; i < 32; i++)
        {
            std::vector<std::tuple<uint8_t, uint8_t, uint8_t>> F_;
            for (size_t j = 0; j < 32; j++)
                F_.insert(F_.begin(),{ batch(index, 0, j, i), batch(index, 1, j, i), batch(index, 2, j, i)});
            F.push_back(F_);
        }

        gp << "set cbrange [0:255]\n";
        gp << "plot '-'  binary array=32x32 format='\%uchar\%uchar\%uchar' with rgbimage title 'RGB Image'\n";
        gp.sendBinary(F);
    }
}