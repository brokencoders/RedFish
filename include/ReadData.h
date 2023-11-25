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

// MINST Dataset http://yann.lecun.com/exdb/mnist/

char grayscale[] = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ";

std::tuple<Tensor, Tensor> readMINSTDataset(const std::string& path_labels, const std::string& path_img)
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

void print_MINST_numbers(const Tensor& n, size_t count = 1)
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

// CIFRA10 Dataset https://www.cs.toronto.edu/~kriz/cifar.html

/*
For each file 10000 images 
- 1 byte for the label 
- 1024 bytes for red channel 
- 1024 bytes for green channel
- 1024 bytes for blue channel
*/
std::tuple<Tensor, Tensor> readCIFRA10Dataset(const std::string& batch_path)
{
    std::ifstream file_labels(batch_path, std::ios::binary);

    if (!file_labels.is_open())
        throw std::runtime_error("Error opening file_labels: " + batch_path + "\n");

    size_t images_n = 10000; 
    size_t chanels = 3;
    size_t width = 32;
    size_t height = 32;

    Tensor images({images_n, chanels, height, width});
    Tensor labels({images_n});

    for (size_t i = 0; i < images_n; i++)
    {
        size_t label = 0;
        file_labels.read((char*)(&label), 1);
        labels(i) = label;
        for (size_t j = 0; j < chanels; j++)
        {
            for (size_t h = 0; h < height; h++)
            {
                for (size_t w = 0; w < width; w++)
                {            
                    size_t pixel = 0;
                    file_labels.read((char*)(&pixel), 1);
                    images(i, j, h, w) = pixel / 255.;
                }
            }   
        }
    }
    
    file_labels.close();

    return {images, labels};
}