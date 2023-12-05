#include "OpenCLManager.h"
#include "RedFish.h"

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

    /*
    std::string category[] = { "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
    auto [batch, data] = readCIFRA10Dataset(dataset_folder + "CIFRA10/data_batch_1.bin");
    Model model({{Layer::Descriptor::CONV2D, {(size_t)3,  (size_t)16, Tuple2d(3), Tuple2d(1), Tuple2d(1), Tuple2d(1), (int8_t)PaddingMode::ZERO}},
                 {Layer::Descriptor::LEAKY_RELU},
                 {Layer::Descriptor::CONV2D, {(size_t)16, (size_t)32, Tuple2d(3), Tuple2d(1), Tuple2d(1), Tuple2d(1), (int8_t)PaddingMode::ZERO}},
                 {Layer::Descriptor::LEAKY_RELU},
                 {Layer::Descriptor::MAXPOOL2D, {Tuple2d(2), Tuple2d(2), Tuple2d(0), Tuple2d(1)}},
                 {Layer::Descriptor::FLATTEN, {(size_t)1, (size_t)-1}},
                 {Layer::Descriptor::LINEAR, {(size_t)256*32, (size_t)256}},
                 {Layer::Descriptor::RELU},
                 {Layer::Descriptor::LINEAR, {(size_t)256, (size_t)10}},
                 {Layer::Descriptor::SOFTMAX}},
                 CROSS_ENTROPY_LOSS,
                 ADAM_OPTIMIZER);

    data.reshape({data.getSize(), 1});
    model.train(batch, data, 300, .0005, 10);

    auto [test_batch, test_data] = readCIFRA10Dataset(dataset_folder + "CIFRA10/data_batch_1.bin");
    test_data.reshape({data.getSize(), 1});

    double accuracy = model.test(/* test_batch,  test_ data, [](const Tensor& pred, const Tensor& gt) {
        double max = 0; 
        int max_index = 0;
        int result_index = gt((size_t)0);
        ///cout << m1 << m2;
        for (size_t i = 0; i < pred.getSize(); i++)
        {
            if (pred(i) > max) max = pred(i), max_index = i; 
        }
        return result_index == max_index;
    });

    std::cout << "Accuracy: " << accuracy * 100 << " %\n";
    */

    /*
    Tensor image = loadImageTensor("../dataset/pinkfloyd.jpg");
    plot_image(image, image.getShape()[1]);
    sleep(2);  

    Tensor up_scale = resizeImage(image, 500, 500);
    plot_image(up_scale, image.getShape()[1]);
    sleep(2);  

    Tensor down_scale = resizeImage(image, 100, 100);
    plot_image(down_scale, image.getShape()[1]);
    sleep(2);  
    */

    int C[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int A[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int B[] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};

    OpenCLManager::init();
    OpenCLManager::createSourceFromFile("../src/kernels/TensorBasic.cl");
    OpenCLManager::createProgram();

    Buffer buffer_A = OpenCLManager::createBuffer<int>(10);
    Buffer buffer_B = OpenCLManager::createBuffer<int>(10);
    Buffer buffer_C = OpenCLManager::createBuffer<int>(10);

    OpenCLManager::loadWriteBuffer<int>(buffer_A, 10, A);
    OpenCLManager::loadWriteBuffer<int>(buffer_B, 10, B);
    
    Kernel kernel = OpenCLManager::createKernel("tensor_tensor_add");

    OpenCLManager::execute(kernel, {buffer_A, buffer_B, buffer_C});

    OpenCLManager::loadReadBuffer<int>(buffer_C, 10, C);
    
    std::cout << "C: \n";
    for(int i=0;i<10;i++){
        std::cout<< C[i] <<" ";
    }

    // model.save("img_classifier.mod", true);

    return 0;
}