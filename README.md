# RedFish

Simple framework for deep learning in C++ (no python allowed), it's a simple project made to learn better how deep learning works. Feel free to contribute. It is meant to have a simple way to extend the base layer class and create custom ones and to create models in few lines of code.
Some GPU acceleration is also provided through OpenCL

#### List of working layers

- Linear (fully connected)
- Convolutional (1d, 2d)
- Max Pooling (1d, 2d)
- Dropout
- Flatten
- Activation
  - Identity
  - ReLU
  - LeakyReLU
  - PReLU
  - Sigmoid
  - TanH
  - Softplus
  - SiLU
  - Gaussian
  - Softmax
  
#### List of working losses

- Square loss
- Cross entropy loss

#### List of working optimizers

- SGD
- Adam

### Working with models

```cpp
Model model({{Layer::Descriptor::LINEAR, {784, 784}},
             {Layer::Descriptor::RELU},
             {Layer::Descriptor::LINEAR, {784, 10}},
             {Layer::Descriptor::SOFTMAX}},
             CROSS_ENTROPY_LOSS,
             ADAM_OPTIMIZER);

model.train(train_samples, train_results, epochs, learning_rate, mini_batch_size);
double accuracy = model.test(test_samples, test_results, accuracy_test_function);

model.save("path/to/file", save_training_data);
```


### How to extend the layer class
```cpp
class CustomLayer : public RedFish::Layer
{
    public:
        CustomLayer(whatever ...);
        ~CustomLayer(); // optional

        Tensor farward(const Tensor& input) override;
        Tensor backward(const Tensor& input, const Tensor& gradient) override;
        uint64_t save(std::ofstream& file) const override;

    private:
        custom variables ...;
};
```

### To-do
- Get full GPU support
- Recurrent layers
- Attention layers
- More losses