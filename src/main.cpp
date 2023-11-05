#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <ctime>

/* #include "gnuplot-iostream.h" */
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

    Model model({{Layer::Descriptor::LINEAR, {(size_t)784, (size_t)784}},
                 {Layer::Descriptor::RELU},
                 {Layer::Descriptor::LINEAR, {(size_t)784, (size_t)10}},
                 {Layer::Descriptor::SOFTMAX}},
                 CROSS_ENTROPY_LOSS,
                 ADAM_OPTIMIZER);

    auto [input, output] = readDataset("../dataset/train_labels", "../dataset/train_images");

    model.train(input, output, 500, .02, 400);
    
    //model.save("model");
    auto [input_test, output_test] = readDataset("../dataset/test_labels", "../dataset/test_images");

    double accuracy = model.test(input_test, output_test, [](const Tensor& m1, const Tensor& m2) {
        double max = 0; 
        int max_index = 0;
        int result_index = m2((size_t)0);
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
        pts_A_xy.push_back({i, ActivationFn::Gaussian::fn(i)});
        pts_B_xy.push_back({i, ActivationFn::Gaussian::bn(i)});
    }

	gp << "set xrange [-10:10]\nset yrange [-1:10]\n";
	gp << "plot" << gp.file1d(pts_A_xy) << "with lines title 'fn',";
    gp << gp.file1d(pts_B_xy) << "with lines title 'bn'" << std::endl;
 */