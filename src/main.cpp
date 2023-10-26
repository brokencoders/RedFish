#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <ctime>

#include "gnuplot-iostream.h"
#include "Tensor.h"
#include "LinearLayer.h"
#include "Model.h"
#include "swap_endian.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace RedFish;

char grayscale[] = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ";

std::tuple<Tensor, Tensor> readDataset(const std::string& path_labels, const std::string& path_img)
{
    /* Labels */
    
    std::ifstream file_labels(path_labels, std::ios::binary);
    int32_t magic_number;
    size_t size, w, h;

    if (!file_labels.is_open())
        throw std::runtime_error("Error opening file_labels: " + path_labels + "\n");


    file_labels.read((char*)(&magic_number), sizeof(int));
    file_labels.read((char*)(&size), sizeof(int));
    swap_endian(magic_number, size);

    std::vector<char> labels(size);
    file_labels.read(labels.data(), size);

    std::cout << magic_number << " " << size << "\n";

    file_labels.close();

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
    
    Tensor in({size, w*h});
    for (size_t i = 0; i < size; i++)
        for (size_t j = 0; j < w*h; j++)
            in(i,j) = imgs[i*w*h + j] / 255.;

    Tensor out(size);

    for (size_t i = 0; i < size; i++)
        out(i) = labels[i];

    return {in, out};
}

void print_number(const Tensor& n)
{
    for (size_t r = 0; r < n.rows(); r++)
    {
        for (size_t c = 0; c < n.cols(); c++)
            std::cout << grayscale[(size_t)(n[r][c] * 70)];
        std::cout << "\n";
    }
}

int main(int, char**)
{
        std::srand(std::time(nullptr));
    using namespace RedFish;
    /* int width, height, channels;
    unsigned char *img = stbi_load("sky.jpg", &width, &height, &channels, 0);
	
    Gnuplot gp("gnuplot -persist");

    std::ofstream file("tst.rgb", std::ios::binary);
    file.write((char*)img, width*height*channels);

	gp << "plot 'tst.rgb' binary format='%uchar' array=(" << width << "," << height << ") with rgbimage notitle" << std::endl;

    return 0; */

    RedFish::Model model(784, {{784, RedFish::Activation::ReLU}, {10, RedFish::Activation::Softmax}}, RedFish::CrossEntropyLoss::get(), RedFish::Adam::get());
    //RedFish::LinearLayer ll1(784, 784, RedFish::Activation::ReLU);
    //RedFish::LinearLayer ll2(784, 10, RedFish::Activation::Softmax);
    //RedFish::Model model("model",  RedFish::CrossEntropyLoss::get());

    auto [input, output] = readDataset("../dataset/train_labels", "../dataset/train_images");

    model.train(input, output, 1000, .02, 20);
    
    //model.save("model");
    auto [input_test, output_test] = readDataset("../dataset/test_labels", "../dataset/test_images");

    double accuracy = model.test(input_test, output_test, [](const Tensor& m1, const Tensor& m2) {
        double max = 0; 
        int max_index = 0;
        int result_index = m2(0);
        for (size_t i = 0; i < m1.getSize(); i++)
        {
            if (m1(i) > max) max = m1(i), max_index = i; 
        }
        return result_index == max_index;
    });

    std::cout << "Accuracy: " << accuracy * 100 << " %\n";
}

/*     Gnuplot gp;
    std::vector<std::pair<double, double>> pts_A_xy;
    std::vector<std::pair<double, double>> pts_B_xy;

    for (double i = -10; i < 10; i+=0.1)
    {
        pts_A_xy.push_back({i, RedFish::ActivationFn::Gaussian::fn(i)});
        pts_B_xy.push_back({i, RedFish::ActivationFn::Gaussian::bn(i)});
    }

	gp << "set xrange [-10:10]\nset yrange [-1:10]\n";
	gp << "plot" << gp.file1d(pts_A_xy) << "with lines title 'fn',";
    gp << gp.file1d(pts_B_xy) << "with lines title 'bn'" << std::endl;
 */