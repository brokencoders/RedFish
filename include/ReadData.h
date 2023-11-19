#pragma once 
#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <ctime>
#include "Tensor.h"
#include "Model.h"
#include "swap_endian.h"

using namespace RedFish;

char grayscale[] = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ";

std::tuple<Tensor, Tensor> readDataset(const std::string& path_labels, const std::string& path_img)
{   
    std::ifstream file_labels(path_labels, std::ios::binary);
    int32_t magic_number, size, w, h;

    if (!file_labels.is_open())
        throw std::runtime_error("Error opening file_labels: " + path_labels + "\n");


    file_labels.read((char*)(&magic_number), sizeof(int));
    file_labels.read((char*)(&size), sizeof(int));
    swap_endian(magic_number, size);

    std::vector<char> labels(size);
    file_labels.read(labels.data(), size);

    std::cout << magic_number << " " << size << "\n";

    file_labels.close();

    Tensor out({(size_t)size});

    for (size_t i = 0, sz = size; i < sz; i++)
        out(i) = labels[i];

    out.reshape({(size_t)size,(size_t)1});

    /* Images */

    std::ifstream file_img(path_img, std::ios::binary);

    if (!file_img.is_open())
        throw std::runtime_error("Error opening file_labels: " + path_img + "\n");

    file_img.read((char*)(&magic_number), sizeof(int));
    file_img.read((char*)(&size), sizeof(int));
    file_img.read((char*)(&w), sizeof(int));
    file_img.read((char*)(&h), sizeof(int));
    swap_endian(magic_number, size, w, h);

    std::vector<unsigned char> imgs(size*w*h);
    file_img.read((char*)imgs.data(), size*w*h);

    std::cout << magic_number << " " << size << " " << w << " " << h << "\n";
    
    Tensor in({(size_t)size, (size_t)w*h});
    for (size_t i = 0, sz = size; i < sz; i++)
        for (size_t j = 0; j < w*h; j++)
            in(i,j) = imgs[i*w*h + j] / 255.;

    return {in, out};
}

void print_numbers(const Tensor& n, size_t count = 1)
{
    for (size_t r = 0; r < count; r++)
    {
        for (size_t c = 0; c < n.colSize(); c++)
        {
            std::cout << grayscale[(size_t)(std::round(n(r,c) * 69))] << grayscale[(size_t)(std::round(n(r,c) * 69))];
            if (c % (size_t)std::sqrt(n.colSize()) == std::sqrt(n.colSize())-1)
                std::cout << "\n";
        }
        std::cout << "\n";
    }
}
