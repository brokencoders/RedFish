#pragma once
#include "With.h"
#include <iostream>

namespace RedFish
{
    
    template<typename LayerType, typename OptType = SGD, typename... Args>
    void test_grad(std::vector<size_t> input_shape, Args... args)
    {
        With w(Layer::training, true);
        OptType opt;
        SquareLoss sl;
        LayerType layer(args...);
        layer.useOptimizer(opt);

        Tensor a = Tensor::empty_like(input_shape).randNormal();
        Tensor out = layer.forward(a);
        Tensor gt  = Tensor::empty_like(out).randNormal();
        Tensor grad = layer.backward(sl.backward(out, gt));
        Tensor grad_es = Tensor::empty_like(a);
        double loss = sl.forward(out, gt);
        double delta = 1e-8;
        Layer::training = false;

        for (size_t i = 0; i < grad_es.getSize(); i++)
        {
            a(i) += delta;
            grad_es(i) = (sl.forward(layer.forward(a), gt) - loss) / delta;
            a(i) -= delta;
        }

        std::cout << "X grad error: " << (grad - grad_es).squareSum() / grad.getSize() << std::endl;
        
        for (size_t k = 0; k < opt.grads.size(); k++)
        {
            grad_es = Tensor::empty_like(opt.grads[k]);
            for (size_t i = 0; i < grad_es.getSize(); i++)
            {
                Tensor& param = *opt.parameters[k];
                param(i) += delta;
                grad_es(i) = (sl.forward(layer.forward(a), gt) - loss) / delta;
                param(i) -= delta;
            }
            std::cout << "Layer grad error: " << (opt.grads[k] - grad_es).squareSum() / grad.getSize() << std::endl;
        }
    }

    template<typename LayerType, typename OptType = SGD, typename... Args>
    void test_learning(std::vector<size_t> input_shape, size_t iterations, Args... args)
    {
        bool tr = Layer::training;
        Layer::training = true;
        OptType opt, fake_opt;
        opt.setLearningRate(.01);
        SquareLoss sl;
        LayerType objective(args...);
        LayerType learner(args...);
        Tensor X(input_shape);
        objective.useOptimizer(fake_opt);
        learner.useOptimizer(opt);

        for (size_t i = 0; i < iterations; i++)
        {
            X.randNormal();
            Tensor gt = objective.forward(X);
            Tensor pred = learner.forward(X);
            auto loss = sl.forward(pred, gt);
            learner.backward(sl.backward(pred, gt));
            opt.step();

            float64 diff = 0;
            for (size_t i = 0; i < opt.parameters.size(); i++)
                diff += (*fake_opt.parameters[i] - *opt.parameters[i]).squareSum();

            /* std::cout << *fake_opt.parameters[0] - *opt.parameters[0];
            std::cout << *fake_opt.parameters[1] - *opt.parameters[1]; */

            std::cout << "loss: " << loss << "\t"
                      << "diff: " << diff << std::endl;
        }
        Layer::training = tr;
    }

}
