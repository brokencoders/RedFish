#include "RedFish.h"

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

void test_grad()
{
    Adam opt;
    SquareLoss sl;
    MaxPool2dLayer layer(3, 2, 0, 3);

    Tensor a({3, 10, 9,18});
    a.randNormal();
    Tensor out = layer.forward(a);
    Tensor gt  = Tensor::empty_like(out);
    gt.randNormal();
    Tensor& param = a;
    Tensor grad = layer.backward(a, sl.backward(out, gt));
    Tensor grad_es = Tensor::empty_like(param);
    double loss = sl.forward(out, gt);
    double delta = 1e-6;

    for (size_t i = 0; i < grad_es.getSize(); i++)
    {
        param(i) += delta;
        grad_es(i) = (sl.forward(layer.forward(a), gt) - loss) / delta;
        param(i) -= delta;
    }

    std::cout << grad << grad_es << (grad - grad_es).squareSum() / grad.getSize();
}

void test_learning()
{
    Adam opt;
    opt.setLearningRate(.1);
    SquareLoss sl;
    Conv3dLayer cl(3, 10, {7,5,3}, &opt);
    Conv3dLayer cl_learner(3, 10, {7,5,3}, &opt);

    for (size_t i = 0; i < 200; i++)
    {
        Tensor X({10, 3, 64, 64, 6});
        X.randNormal();
        Tensor gt = cl.forward(X);
        Tensor pred = cl_learner.forward(X);
        auto loss = sl.forward(pred, gt);
        cl_learner.backward(X, sl.backward(pred, gt));

        std::cout << "loss: " << loss << "\t"
                  << "diff: " << (cl.kernels - cl_learner.kernels).squareSum() + (cl.bias - cl_learner.bias).squareSum() << std::endl;
    }
}

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