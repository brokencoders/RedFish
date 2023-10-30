#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <ctime>

/* #include "gnuplot-iostream.h" */
#include "Tensor.h"
#include "LinearLayer.h"
#include "ActivationLayer.h"
#include "Loss.h"
#include "swap_endian.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace RedFish;

char grayscale[] = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ";

std::tuple<Tensor, Tensor> readDataset(const std::string& path_labels, const std::string& path_img)
{
    /* Labels */
    
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

    //RedFish::Model model(784, {{784, RedFish::Activation::ReLU}, {10, RedFish::Activation::Softmax}}, RedFish::CrossEntropyLoss::get(), RedFish::Adam::get());
    Adam opt(.01);
    LinearLayer ll1(784, 784, &opt);
    Activation::ReLU act1;
    LinearLayer ll2(784, 10,  &opt);
    Activation::Softmax act2;
    CrossEntropyLoss loss;

    LinearLayer test(1, 1, &opt);
    SquareLoss sqloss;

    auto [input, output] = readDataset("../dataset/train_labels", "../dataset/train_images");
    //print_numbers(input,5);

    //model.train(input, output, 1000, .02, 20);
    const size_t epochs = 1000, mini_batch_size = 20;
    Tensor mini_batch_in( {mini_batch_size,  input.colSize()});
    Tensor mini_batch_out({mini_batch_size, output.colSize()});
    for (size_t i = 0; i < epochs; i++)
    {
        for (size_t j = 0; j < mini_batch_size; j++)
        {
            size_t n = rand() % input.rowSize();
            for (size_t i = 0; i < input.colSize(); i++)
                mini_batch_in(j,i)  = input(n,i);

            for (size_t i = 0; i < output.colSize(); i++)
                mini_batch_out(j,i) = output(n,i);

            /* mini_batch_in(j) = ((double)rand() / RAND_MAX - .5) * 200.;
            mini_batch_out(j) = 3*mini_batch_in(j) + 5; */
        }

        auto& f0 = mini_batch_in;
        auto  f1 = ll1.farward(f0);
        auto  f2 = act1.farward(f1);
        auto  f3 = ll2.farward(f2);
        auto  f4 = act2.farward(f3);
        auto  f5 = loss.farward(f4, mini_batch_out);
        auto  b5 = loss.backward(f4, mini_batch_out);
        auto  b4 = act2.backward(f3, b5);
        auto  b3 = ll2.backward(f2, b4);
        auto  b2 = act1.backward(f1, b3);
        auto  b1 = ll1.backward(f0, b2);

        //std::cout << input << f0 << f1 << f2 << f3 << f4 << b5 << b4 << b3 << b2 << b1;

        /* auto f1 = test.farward(f0);
        auto f2 = sqloss.farward(f1, mini_batch_out);
        auto b2 = sqloss.backward(f1, mini_batch_out);
        auto b1 = test.backward(f0, b2); */


        std::cout << "Epoch " << i << " - loss: " << f5 << "\n";

    }
    
    
    //model.save("model");
    auto [input_test, output_test] = readDataset("../dataset/test_labels", "../dataset/test_images");

    /* double accuracy = model.test(input_test, output_test, [](const Tensor& m1, const Tensor& m2) {
        double max = 0; 
        int max_index = 0;
        int result_index = m2((size_t)0);
        for (size_t i = 0; i < m1.getSize(); i++)
        {
            if (m1(i) > max) max = m1(i), max_index = i; 
        }
        return result_index == max_index;
    }); */

    auto& f0 = input_test;
    auto  f1 = ll1.farward(f0);
    auto  f2 = act1.farward(f1);
    auto  f3 = ll2.farward(f2);
    auto  preds = act2.farward(f3);

    double accuracy = 0;
    for (size_t i = 0; i < preds.rowSize(); i++)
    {
        double max = 0; 
        int max_index = 0;
        int result_index = output_test((size_t)i);
        for (size_t j = 0; j < preds.colSize(); j++)
        {
            auto it = preds(i, j);
            if (it > max) max = it, max_index = j; 
        }
        accuracy += result_index == max_index;
    }
    accuracy /= input_test.rowSize();

    std::cout << preds << output_test << "Accuracy: " << accuracy * 100 << " %\n";

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