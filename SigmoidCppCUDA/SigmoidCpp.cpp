#include <torch/extension.h>

#include <vector>

// CUDA interfaces.
std::vector<torch::Tensor> sigmoid_cpp_forward_cuda( torch::Tensor input );
std::vector<torch::Tensor> sigmoid_cpp_backward_cuda( torch::Tensor grad, torch::Tensor s );

// C++ interfaces.

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> sigmoid_cpp_forward( torch::Tensor input )
{
    CHECK_INPUT(input);

    return sigmoid_cpp_forward_cuda(input);
}

std::vector<torch::Tensor> sigmoid_cpp_backward( torch::Tensor grad, torch::Tensor s )
{
    CHECK_INPUT(grad);
    CHECK_INPUT(s);

    return sigmoid_cpp_backward_cuda(grad, s);
}

PYBIND11_MODULE( TORCH_EXTENSION_NAME, m )
{
    m.def("forward", &sigmoid_cpp_forward, "SigmoidCpp forward, CUDA version. ");
    m.def("backward", &sigmoid_cpp_backward, "SigmoidCpp backward, CUDA version. ");
}

