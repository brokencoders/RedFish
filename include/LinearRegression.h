#pragma once 

#include "Parser.h"
#define ALGEBRA_IMPL
#include "Algebra.h"
#include "LinearLayer.h"

namespace RedFish {

    class LinearRegression {
    public:
        LinearRegression();

        void splitTrainTestBatch(Parser::CSV::File& csv, int percentage);

        void train(size_t);
        void test();
    public:
        Algebra::Matrix x_train;
        Algebra::Matrix y_train;
        //Algebra::Matrix weights;
        Neuron<1> ne;
    };

    LinearRegression::LinearRegression() { }

    void LinearRegression::splitTrainTestBatch(Parser::CSV::File& csv, int percentageTest)
    {
        // Get All the numbers numbers 
        if(percentageTest > 100 || percentageTest < 0)
            std::runtime_error("The percentage number must be between zero and one hundred");

        std::cout << "We have " << csv.data.size() << " data\n";

        int train_batch_size = csv.data.size() / 100. * percentageTest;
        int test_batch_size = csv.data.size() - train_batch_size;

        std::cout << "We are going to use " << train_batch_size << " to train the model\n";
        std::cout << "We are going to use " << test_batch_size << " to test the model\n";

        // Split Train Data and Test Data
        // x_train, y_train, x_test, y_test 
        
        x_train = Algebra::Matrix(train_batch_size, csv.data.at(0).numbers.size() - 1);
        y_train = Algebra::Matrix(train_batch_size);
        //weights = Algebra::Matrix(csv.data.at(0).numbers.size());

        for(int i = 0; i < train_batch_size; i++)
        {
            for(int j = 0; j < csv.data[i].numbers.size() - 1; j++)
                x_train(i, j) = csv.data[i].numbers[j];
            //x_train(i, csv.data[i].numbers.size() - 1) = 1;
            y_train(i) = csv.data[i].numbers[csv.data[i].numbers.size() - 1];
        }

        x_train.print();
        y_train.print();
        ne.print();

    }

    void LinearRegression::train(size_t epoch)
    {
        // forward linear regression 
        // loss gradient 
        // backwords pass

        using namespace Algebra;

        double learning_rate = 0.01;

        for (size_t i = 0; i < epoch; i++)
        {
            /* Matrix N = x_train * weights;
            double loss = (N - y_train).normSquare();
            auto dLdP = 2 * (N - y_train);
            auto dNdW = x_train.T();

            auto grad = dNdW * dLdP;

            weights -= learning_rate * grad; */
            /* weights.print();
            grad.print(); */

            auto N = ne.farward(x_train);
            double loss = (N - y_train).normSquare();
            ne.backward(x_train, (N - y_train), learning_rate);

            std::cout << "Loss: " << loss << "\n";
        }
        
    }

    void LinearRegression::test()
    {
        
    }


}




