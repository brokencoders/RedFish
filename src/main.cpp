/*
Housing Dataset Linear Regression
This data set has a number of features, including:
-   The average income in the area of the house
-   The average number of total rooms in the area
-   The price that the house sold for
-   The address of the house
*/


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
#include "swap_endian.h"


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

    std::vector<char> imgs(size*w*h);
    file_img.read(imgs.data(), size*w*h);

    std::cout << magic_number << " " << size << " " << w << " " << h << "\n";

    for (size_t i = 0; i < 2; i++) {
        std::cout << "\n" << (int)labels[i] << "\n";
        for (size_t r = 0; r < h; r++)
        {
            for (size_t c = 0; c < w; c++)
            {
                std::cout << grayscale[(size_t)((unsigned char)imgs[r*w + c + w*h*i]) * 68 / 255];
            }
            std::cout << "\n";
        }
    }
    
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


class DigitsModel
{
private:
    RedFish::LinearLayer<784, 784, RedFish::ActivationFn::ReLU>  hidden;
    RedFish::LinearLayer<784, 10,  RedFish::ActivationFn::Softmax> last;

public:

    void train(const Matrix& in, const Matrix& out, uint epochs = 100, double learning_rate = .01);
    int estimate();
};

void DigitsModel::train(const Matrix& in, const Matrix& out, uint epochs, double learning_rate)
{
    double lloss = 100.;
    for (int i = 0; i < epochs; i++)
    {
        auto m1 = hidden.farward(in);
        auto m2 = last.farward(m1);
        double loss = (m2 - out).normSquare() / out.rows();
        std::cout << "Epoch " << i << " - loss: " << loss << "\n";
        lloss = loss;
        auto dm2dm1 = last.backward(m1, m2 - out, learning_rate);
        hidden.backward(in, dm2dm1, learning_rate);
    }
}



int main(int, char**)
{
    auto [input, output] = readDataset("../dataset/train_labels", "../dataset/train_images");

    DigitsModel dm;
    dm.train(input.subMatrix(0,0, 10), output.subMatrix(0,0, 10), 1000, .001);

    exit(0);

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
    // Get Data 

    /* Parser::CSV::File housing_dataset("../dataset/line.csv");
    std::cout << housing_dataset;
    
    RedFish::LinearRegression lr = RedFish::LinearRegression();

    lr.splitTrainTestBatch(housing_dataset, 80);

    lr.train(1000);
    lr.ne.print(); */
    // tr.prediction(x_test)

    double learning_rate = 0.0001;
    int epochs = 100000;
    RedFish::LinearLayer<2, 2, RedFish::ActivationFn::Softmax> LL1;
    /* RedFish::LinearLayer<10, 1, RedFish::ReLU> LL2; */
    RedFish::Neuron<1> ne;
    Matrix in(20, 2), out(20, 2);
    for (size_t i = 0; i < in.rows(); i++)
        in[i][0] = i,
        in[i][1] = 0;
    for (size_t i = 0; i < out.rows(); i++)
        out[i][0] = i+1,
        out[i][1] = i+5;
    
    in.print();
    out.print();

    double loss;

    for (int i = 0; i < epochs; i++)
    {
        auto m1 = LL1.farward(in);
        /* auto m2 = LL2.farward(m1); */
        loss = (m1 - out).normSquare();
        /* auto dm2dm1 = LL2.backward(m1, m2 - out, learning_rate); */
        auto dm1din = LL1.backward(in, m1 - out, learning_rate);

    }
    std::cout << "Loss: " << loss << "\n";

    LL1.farward(Matrix({-20, 0}).T()).print();

    return 0;
}
