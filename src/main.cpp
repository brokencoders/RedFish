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

std::string characters = "0123456789+-*/";

namespace RedFish {void print_ttime();void print_ctime();}

int main(int, char**)
{
    {
        test_grad<Conv1dLayer>({2,3,200}, 3, 10, 9);
        test_grad<Conv2dLayer>({2,3,20,20}, 3, 10, 5);
        test_grad<Conv3dLayer>({2,3,10,8,7}, 3, 6, 5);

        test_learning<Conv1dLayer>({2,3,200}, 30, 3, 10, 9);
        test_learning<Conv2dLayer>({2,3,20,20}, 30, 3, 10, 5);
        test_learning<Conv3dLayer, Adam>({2,3,25,25,10}, 3, 6, 5);
    }
    {
        auto [test_images, test_labels] = readCIFRA10Dataset(dataset_folder + "CIFAR10/data_batch_5.bin");
        Tensor train_images, train_labels;

        {
            auto [train_images1, train_labels1] = readCIFRA10Dataset(dataset_folder + "CIFAR10/data_batch_1.bin");
            auto [train_images2, train_labels2] = readCIFRA10Dataset(dataset_folder + "CIFAR10/data_batch_2.bin");
            auto [train_images3, train_labels3] = readCIFRA10Dataset(dataset_folder + "CIFAR10/data_batch_3.bin");
            auto [train_images4, train_labels4] = readCIFRA10Dataset(dataset_folder + "CIFAR10/data_batch_4.bin");
            
            train_images = Tensor::stack({train_images1,train_images2,train_images3,train_images4},3);
            train_labels = Tensor::stack({train_labels1,train_labels2,train_labels3,train_labels4},1);
        }

        Model model({
                    new Conv2dLayer(3, 6, 5),
                    new Activation::ReLU,
                    new MaxPool2dLayer(2),
                    new DropoutLayer(.25),
                    new Conv2dLayer(6, 16, 5),
                    new Activation::ReLU,
                    new MaxPool2dLayer(2),
                    new FlattenLayer(2),
                    new LinearLayer(16*5*5, 120),
                    new Activation::ReLU,
                    new DropoutLayer(.25),
                    new LinearLayer(120, 84),
                    new Activation::ReLU,
                    new DropoutLayer(.25),
                    new LinearLayer(84, 10),
                    new Activation::Softmax
                    },
                    new CrossEntropyLoss,
                    new Adam);

        for (size_t i = 0; i < 10; i++)
        {
            model.train(train_images, train_labels, 10, .005, 500, i == 0);
            auto accuracy = model.test(test_images, test_labels, correctly_classified);
            std::cout << "Accuracy: " << accuracy * 100 << " %\n";
        }
    }

    {
        auto [train_images, train_labels] = readMNISTDataset(dataset_folder + "MNIST/train_labels", dataset_folder + "MNIST/train_images");
        auto [test_images,  test_labels]  = readMNISTDataset(dataset_folder + "MNIST/test_labels",  dataset_folder + "MNIST/test_images");
        
        Model model({new Conv2dLayer(1,32,5),
                    new Activation::ReLU,
                    new MaxPool2dLayer(2),
                    new DropoutLayer(.25),
                    new FlattenLayer(2),
                    new LinearLayer(12*12*32, 256),
                    new Activation::ReLU,
                    new LinearLayer(256, 128),
                    new Activation::ReLU,
                    new DropoutLayer(.25),
                    new LinearLayer(128, 10),
                    new Activation::Softmax,
                    },
                    new CrossEntropyLoss,
                    new Adam);

        for (size_t i = 0; i < 10; i++)
        {
            model.train(train_images.asShape({60000,1,28,28}), train_labels, 1, .02, 500);
            auto accuracy = model.test(test_images.asShape({10000,1,28,28}), test_labels, correctly_classified);
            std::cout << "Accuracy: " << accuracy * 100 << " %\n";
        }
    }

    return 0;
}