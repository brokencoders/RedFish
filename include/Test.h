#pragma once

namespace RedFish
{
    
    template<typename LayerType, typename... Args>
    void test_grad(std::vector<size_t> input_shape, Args... args)
    {
        Adam opt;
        SquareLoss sl;
        LayerType layer(args..., &opt);

        Tensor a(input_shape);
        a.randNormal();
        Tensor out = layer.forward(a);
        Tensor gt  = Tensor::empty_like(out);
        gt.randNormal();
        Tensor& param = a;
        Tensor grad = layer.backward(sl.backward(out, gt));
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

    template<typename LayerType, typename OptType = SGD, typename... Args>
    void test_learning(std::vector<size_t> input_shape, size_t iterations, Args... args)
    {
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
    }

}
