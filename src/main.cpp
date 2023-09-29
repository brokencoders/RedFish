#include <iostream>
#include <map>
#include <vector>
#include <cmath>

#include <fstream>
#include <sstream>
#include <vector>
#include <utility>
#include <unordered_map>
#include <cstdarg>

#include "gnuplot-iostream.h"

#include "Math.h"
#include "LinearRegression.h"
#include "LinearLayer.h"
#include "Model.h"
#include "swap_endian.h"

using namespace Algebra;

char grayscale[] = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ";

std::tuple<Matrix, Matrix> readDataset(const std::string& path_labels, const std::string& path_img)
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
    
    Matrix in(size, w*h);
    for (size_t i = 0; i < size; i++)
        for (size_t j = 0; j < w*h; j++)
            in(i,j) = imgs[i*w*h + j] / 255.;

    Matrix out(size, 10);
    zero(out);

    for (size_t i = 0; i < size; i++)
        out(i,labels[i]) = 1.;

    return {in, out};
}

void print_number(const Matrix& n)
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
    using namespace RedFish;
    RedFish::Model model(784, {{784, RedFish::Activation::ReLU}, {10, RedFish::Activation::Softmax}});
    // RedFish::Model model("model");

    auto [input, output] = readDataset("../dataset/train_labels", "../dataset/train_images");

    model.train(input, output, 100, .01, 100);
    // model.save("model");
    auto [input_test, output_test] = readDataset("../dataset/test_labels", "../dataset/test_images");

    double accuracy = model.test(input_test, output_test, [](const Algebra::Matrix& m1, const Algebra::Matrix& m2) {
        double max = 0; 
        int max_index = 0;
        int result_index = 0;
        for (size_t i = 0; i < m1.getSize(); i++)
        {
            if (m2(i) == 1) result_index = i;
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