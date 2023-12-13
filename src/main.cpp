#include "OpenCLManager.h"
#include "RedFish.h"

#include <iostream>
#include <thread>

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

int main(int, char**)
{
    std::string category[] = { "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
    auto [train_samples, train_results] = readCIFRA10Dataset(dataset_folder + "CIFRA10/data_batch_1.bin");
    auto [ test_samples,  test_results] = readCIFRA10Dataset(dataset_folder + "CIFRA10/test_batch.bin");
    train_results.reshape({train_results.getSize(), 1});
    test_results.reshape({test_results.getSize(), 1});
    
    // GPU please 
    // Model.toDevice();
    Model model({{Layer::Descriptor::CONV2D, {(size_t)3,  (size_t)16, Tuple2d(3), Tuple2d(1), Tuple2d(1), Tuple2d(1), (int8_t)PaddingMode::ZERO}},
                 {Layer::Descriptor::LEAKY_RELU},
                 {Layer::Descriptor::CONV2D, {(size_t)16, (size_t)32, Tuple2d(3), Tuple2d(1), Tuple2d(1), Tuple2d(1), (int8_t)PaddingMode::ZERO}},
                 {Layer::Descriptor::LEAKY_RELU},
                 {Layer::Descriptor::MAXPOOL2D, {Tuple2d(2), Tuple2d(2), Tuple2d(0), Tuple2d(1)}},
                 {Layer::Descriptor::FLATTEN, {(size_t)1, (size_t)-1}},
                 {Layer::Descriptor::LINEAR,  {(size_t)256*32, (size_t)256}},
                 {Layer::Descriptor::DROPOUT, {(float64).2}},
                 {Layer::Descriptor::RELU},
                 {Layer::Descriptor::LINEAR, {(size_t)256, (size_t)10}},
                 {Layer::Descriptor::SOFTMAX}},
                 CROSS_ENTROPY_LOSS,
                 ADAM_OPTIMIZER);
    // Model model("img_classifier.mod");
    //model.layers.insert(model.layers.begin() + 7, make_layer({Layer::Descriptor::DROPOUT, {(float64).5}}, model.optimizer));
    //model.layers.erase(model.layers.begin() + 7);

    for (size_t i = 0; i < 100; i++)
    {
        model.train(train_samples, train_results, 1000, .0005, 20);
        std::string name = "img_classifier_v0.2.mod";
        model.save(name, false);

        auto drop = model.layers[7];
        model.layers.erase(model.layers.begin() + 7);

        float64 accuracy = model.test(test_samples, test_results, correctly_classified);
        std::cout << "Accuracy: " << accuracy * 100 << " %\n";
        model.layers.insert(model.layers.begin() + 7, drop);
    }

    float64 accuracy = model.test(test_samples, test_results, correctly_classified);
    std::cout << "Accuracy: " << accuracy * 100 << " %\n";

    model.save("img_classifier.mod", true);

    return 0;
}