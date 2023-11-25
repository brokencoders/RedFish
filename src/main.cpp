#include "RedFish.h"
#include "gnuplot-iostream.h"

using namespace std;
using namespace RedFish;

std::string dataset_folder = "../dataset/";
std::string model_folder = "../models/";

int main(int, char**)
{
    /*
    std::srand(std::time(nullptr));

    Model model({{Layer::Descriptor::LINEAR, {(size_t)784, (size_t)784}},
                 {Layer::Descriptor::RELU},
                 {Layer::Descriptor::LINEAR, {(size_t)784, (size_t)10}},
                 {Layer::Descriptor::SOFTMAX}},
                 CROSS_ENTROPY_LOSS,
                 ADAM_OPTIMIZER);

    auto [input, output] = readMINSTDataset(dataset_folder + "MNIST/train_labels", dataset_folder + "MNIST/train_images");

    model.train(input, output, 500, .02, 50);
    
    // model.save(model_folder + "numbers");
    auto [input_test, output_test] = readMINSTDataset(dataset_folder + "MNIST/test_labels", dataset_folder + "MNIST/test_images");

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
    */

    std::string category[] = { "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
    auto [batch, data] = readCIFRA10Dataset(dataset_folder + "CIFRA10/data_batch_1.bin");
    Model model({/* {Layer::Descriptor::CONV2D, {(size_t)3,  (size_t)16, Tuple2d(3), Tuple2d(1), Tuple2d(1), Tuple2d(1), (int8_t)PaddingMode::ZERO}},
                 {Layer::Descriptor::LEAKY_RELU},
                 {Layer::Descriptor::CONV2D, {(size_t)16, (size_t)32, Tuple2d(3), Tuple2d(1), Tuple2d(1), Tuple2d(1), (int8_t)PaddingMode::ZERO}},
                 {Layer::Descriptor::LEAKY_RELU},
                 {Layer::Descriptor::MAXPOOL2D, {Tuple2d(2), Tuple2d(2), Tuple2d(0), Tuple2d(1)}}, */
                 {Layer::Descriptor::FLATTEN, {(size_t)1, (size_t)-1}},
                 {Layer::Descriptor::LINEAR, {(size_t)/* 256*32 */32*32*3, (size_t)256}},
                 {Layer::Descriptor::RELU},
                 {Layer::Descriptor::LINEAR, {(size_t)256, (size_t)10}},
                 {Layer::Descriptor::SOFTMAX}},
                 CROSS_ENTROPY_LOSS,
                 ADAM_OPTIMIZER);

    data.reshape({data.getSize(), 1});
    model.train(batch, data, {3,32,32}, 10, .03, 5);

    auto [test_batch, test_data] = readCIFRA10Dataset(dataset_folder + "CIFRA10/data_batch_1.bin");
    test_data.reshape({data.getSize(), 1});

    double accuracy = model.test(test_batch, test_data, [](const Tensor& m1, const Tensor& m2) {
        double max = 0; 
        int max_index = 0;
        int result_index = m2((size_t)0);
        ///cout << m1 << m2;
        for (size_t i = 0; i < m1.getSize(); i++)
        {
            if (m1(i) > max) max = m1(i), max_index = i; 
        }
        return result_index == max_index;
    });

    std::cout << "Accuracy: " << accuracy * 100 << " %\n";

    return 0;
}