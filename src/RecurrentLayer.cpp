#include "RecurrentLayer.h"

namespace RedFish {
    
    template<typename Act>
    RecurrentLayer<Act>::RecurrentLayer(size_t input_size, size_t output_size, Optimizer* optimizer)
        : Wi({input_size, output_size}), Wh({output_size, output_size}), b({output_size}), Y({1, output_size}), h({1, output_size}), optimizer(optimizer)
    {
        Wi.randUniform(-.5, .5);
        Wh.randUniform(-.5, .5);
        b.randUniform(-.5, .5);
        Y.zero();
        h.zero();
        Wi_id = optimizer->allocateParameter(Wi);
        Wh_id = optimizer->allocateParameter(Wh);
        b_id = optimizer->allocateParameter(b);
    }

    template<typename Act>
    RecurrentLayer<Act>::RecurrentLayer(std::ifstream &file, Optimizer* optimizer)
        : optimizer(optimizer)
    {
        const std::string name = "Layer::Recurrent";
        char rname[sizeof("Layer::Recurrent")];
        file.read(rname, sizeof(rname));

        if (name != rname)
            throw std::runtime_error("Invalid file content in RecurrentLayer(std::ifstream&)");

        uint64_t size = 0;
        file.read((char*)&size, sizeof(size));

        file.read((char*)&Wi_id, sizeof(Wi_id));
        file.read((char*)&Wh_id, sizeof(Wh_id));
        file.read((char*)&b_id, sizeof(b_id));
 
        Wi = Tensor(file);
        Wh = Tensor(file);
        b  = Tensor(file);

        std::cout << Wi << Wh << b;
    }

    template<typename Act>
    Tensor RecurrentLayer<Act>::forward(const Tensor &X)
    {
        Y = X.matmul(Wi) + b;
        h = Tensor::empty_like(Y);
        h.getMatrix({0}) = f.forward(Y.getMatrix({0}));
        size_t time_length = Y.getShape().size() > 2 ? Y.getShape().end()[-3] : 1;
        for (size_t t = 1; t < time_length; t++)
        {
            Y.getMatrix({t}) += h.getMatrix({t-1}).matmul(Wh);
            h.getMatrix({t}) = f.forward(Y.getMatrix({t}));
        }
        
        return h;
    }

    template<typename Act>
    Tensor RecurrentLayer<Act>::backward(const Tensor &X, const Tensor &D)
    {
        h = h.shift(2,1);
        Tensor d = f.backward(Y, D);
        Tensor grad_X  = d.matmul(Wi, Transpose::RIGHT);
        Tensor grad_Wi = X.matmul(d,  Transpose::LEFT).sum(2);
        Tensor grad_Wh = h.matmul(d,  Transpose::LEFT).sum(2);
        Tensor grad_b  = d.sum(1).sum(2);

        size_t backward_steps = Y.getShape().size() > 2 ? Y.getShape().end()[-3] : 1;

        for (size_t t = 1; t < backward_steps; t++)
        {
            d = f.backward(Y, d.matmul(Wh, Transpose::RIGHT).shift(2, -1));
            grad_X  += d.matmul(Wi, Transpose::RIGHT);
            grad_Wi += X.matmul(d,  Transpose::LEFT).sum(2);
            grad_Wh += h.matmul(d,  Transpose::LEFT).sum(2);
            grad_b  += d.sum(1).sum(2);
        }

        optimizer->updateParameter(Wi_id, Wi, grad_Wi);
        optimizer->updateParameter(Wh_id, Wh, grad_Wh);
        optimizer->updateParameter(b_id,  b,  grad_b);

        return grad_X;
    }

    template<typename Act>
    uint64_t RecurrentLayer<Act>::save(std::ofstream &file) const
    {
        const char name[] = "Layer::Recurrent";
        file.write(name, sizeof(name));
        uint64_t size = sizeof(Wi_id) + sizeof(Wh_id) + sizeof(b_id);

        auto spos = file.tellp();
        file.write((char*)&size, sizeof(size));

        file.write((char*)&Wi_id, sizeof(Wi_id));
        file.write((char*)&Wh_id, sizeof(Wh_id));
        file.write((char*)&b_id, sizeof(b_id));
 
        size += Wi.save(file);
        size += Wh.save(file);
        size += b.save(file);

        auto cpos = file.tellp();
        file.seekp(spos);
        file.write((char*)&size, sizeof(size));
        file.seekp(cpos);
        
        return size + sizeof(uint64_t) + sizeof(name);
    }

    template<typename Act>
    void RecurrentLayer<Act>::print()
    {
        std::cout << "Wi = \n" << Wi << "Wh = \n" << Wh << "b = " << b << "\n";
    }

    void dummy()
    {
        using namespace RedFish;
        using namespace Activation;
        RecurrentLayer<Identity>  r0(0,0,0);
        RecurrentLayer<ReLU>      r1(0,0,0);
        RecurrentLayer<LeakyReLU> r2(0,0,0);
        RecurrentLayer<Sigmoid>   r3(0,0,0);
        RecurrentLayer<TanH>      r4(0,0,0);
        RecurrentLayer<Softplus>  r5(0,0,0);
        RecurrentLayer<SiLU>      r6(0,0,0);
        RecurrentLayer<Gaussian>  r7(0,0,0);
        RecurrentLayer<Softmax>   r8(0,0,0);
    }
}

