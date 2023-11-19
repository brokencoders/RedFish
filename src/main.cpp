#include <iostream>
#include "RedFish.h"

using namespace RedFish;

std::string dataset_folder = "../RedFish/dataset/";
std::string model_folder = "../RedFish/models/";

int main(int, char**)
{
    std::srand(std::time(nullptr));

    Model model({{Layer::Descriptor::LINEAR, {(size_t)784, (size_t)784}},
                 {Layer::Descriptor::RELU},
                 {Layer::Descriptor::LINEAR, {(size_t)784, (size_t)10}},
                 {Layer::Descriptor::SOFTMAX}},
                 CROSS_ENTROPY_LOSS,
                 ADAM_OPTIMIZER);

    auto [input, output] = readDataset(dataset_folder + "train_labels", dataset_folder + "/train_images");

    model.train(input, output, 500, .02, 50);
    
    // model.save(model_folder + "numbers");
    auto [input_test, output_test] = readDataset(dataset_folder + "test_labels", dataset_folder + "test_images");

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