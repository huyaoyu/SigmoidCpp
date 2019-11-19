#include <torch/extension.h>

#include <iostream>
#include <vector>

std::vector<torch::Tensor> sigmoid_cpp_forward( torch::Tensor input )
{
    return { torch::sigmoid( input ) };
}

std::vector<torch::Tensor> sigmoid_cpp_backward( torch::Tensor grad, torch::Tensor s )
{
    auto sp  = (1 - s) * s;

    return { grad * sp };
}

PYBIND11_MODULE( TORCH_EXTENSION_NAME, m )
{
    m.def("forward", &sigmoid_cpp_forward, "SigmoidCpp forward");
    m.def("backward", &sigmoid_cpp_backward, "SigmoidCpp backward");
}

