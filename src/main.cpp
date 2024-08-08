#include "RedFish.h"
#include "Test.h"

#include <iostream>
#include <thread>
#include <memory>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;
using namespace RedFish;

std::string dataset_folder = "../dataset/";
std::string model_folder = "../models/";

bool correctly_classified(const Tensor& pred, const Tensor& gt)
{
    double max = 0; 
    int max_index = 0;
    int result_index = gt((size_t)0);
    for (size_t i = 0; i < pred.getSize(); i++)
    {
        if (pred(i) > max) max = pred(i), max_index = i; 
    }
    return result_index == max_index;
}

bool correctly_classified_MNIST(const Tensor& pred, const Tensor& gt)
{
    double max = 0;
    int max_index = 0;
    int result_index = gt((size_t)0);
    for (size_t i = 0; i < pred.getSize(); i++)
    {
        if (pred(i) > max) max = pred(i), max_index = i; 
    }
    return result_index == max_index;
}

std::string characters = "0123456789+-*/";

namespace RedFish {void print_ttime();void print_ctime();}

int main(int, char**)
{
    RecurrentLayer<Activation::ReLU> rnn(10, 10, make_optimizer<ADAM_OPT>());

    auto [train_images, train_labels] = readMNISTDataset(dataset_folder + "MNIST/train_labels", dataset_folder + "MNIST/train_images");
    auto [test_images,  test_labels]  = readMNISTDataset(dataset_folder + "MNIST/test_labels",  dataset_folder + "MNIST/test_images");

    Model model({{LAYER::LINEAR, {(size_t)784, (size_t)128}},
                 {LAYER::RELU},
                 {LAYER::LINEAR, {(size_t)128, (size_t)10}},
                 {LAYER::SOFTMAX}},
                 CROSS_ENTROPY_LOSS,
                 ADAM_OPT);

    model.train(train_images, train_labels, 50, .1, 5000);
    auto accuracy = model.test(test_images, test_labels, correctly_classified_MNIST);
    std::cout << "Accuracy: " << accuracy * 100 << " %\n";

    print_ttime();

    return 0;
}