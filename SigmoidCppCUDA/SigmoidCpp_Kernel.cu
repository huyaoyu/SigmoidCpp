#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

// ========== Device functions. ==========

template <typename scalar_t> 
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t x)
{
    return 1.0 / ( 1.0 + exp(-x) );
}

// ========== Kernel functions. ==========

template <typename scalar_t>
__global__ void k_sigmoid_cpp_forward( 
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> input,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> output )
{
    const int idxX    = blockIdx.x * blockDim.x + threadIdx.x;
    const int idxY    = blockIdx.y * blockDim.y + threadIdx.y;
    const int idxZ    = blockIdx.z * blockDim.z + threadIdx.z;
    const int strideX = gridDim.x * blockDim.x;
    const int strideY = gridDim.y * blockDim.y;
    const int strideZ = gridDim.z * blockDim.z;

    const int b = input.size(0);
    const int h = input.size(1);
    const int w = input.size(2);

    for ( int z = idxZ; z < b; z += strideZ )
    {
        for ( int y = idxY; y < h; y += strideY )
        {
            for ( int x = idxX; x < w; x += strideX )
            {
                output[z][y][x] = d_sigmoid( input[z][y][x] );
            }
        }
    }
}

template <typename scalar_t> 
__global__ void k_sigmoid_cpp_backward(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> grad,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> s,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> output )
{
    const int idxX    = blockIdx.x * blockDim.x + threadIdx.x;
    const int idxY    = blockIdx.y * blockDim.y + threadIdx.y;
    const int idxZ    = blockIdx.z * blockDim.z + threadIdx.z;
    const int strideX = gridDim.x * blockDim.x;
    const int strideY = gridDim.y * blockDim.y;
    const int strideZ = gridDim.z * blockDim.z;

    const int b = s.size(0);
    const int h = s.size(1);
    const int w = s.size(2);

    for (int z = idxZ; z < b; z += strideZ )
    {
        for ( int y = idxY; y < h; y += strideY )
        {
            for ( int x = idxX; x < w; x += strideX )
            {
                output[z][y][x] = 
                    grad[z][y][x] * 
                    ( 1.0 - s[z][y][x] ) * s[z][y][x];
            }
        }
    }
}

// ========== Interface functions. ==========

std::vector<torch::Tensor> sigmoid_cpp_forward_cuda( torch::Tensor input )
{
    // Get the batch size.
    auto b = input.size(0);

    // The 2D tensor dimensions.
    auto h = input.size(1);
    auto w = input.size(2);

    // Prepare output.
    auto output = torch::zeros_like(input);

    const int threadsX = 2;
    const int threadsY = 2;

    // Kernal launch dimensions.
    const dim3 blocks( ( w + threadsX - 1 ) / threadsX, ( h + threadsY - 1 ) / threadsY, b );
    const dim3 thrds( threadsX, threadsY, 1 );

    // Kernal launch.
    AT_DISPATCH_FLOATING_TYPES( input.type(), "sigmoid_cpp_forwrd_cuda", ([&] {
        k_sigmoid_cpp_forward<scalar_t><<<blocks, thrds>>>( 
            input.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            output.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>() );
    }) );

    return { output };
}

std::vector<torch::Tensor> sigmoid_cpp_backward_cuda( torch::Tensor grad, torch::Tensor s )
{
    // Get the batch size.
    auto b = s.size(0);

    // Get the 2D tensor dimesnions.
    auto h = s.size(1);
    auto w = s.size(2);

    // The result.
    auto output = torch::zeros_like(s);

    const int threadsX = 2;
    const int threadsY = 2;

    // Kernal launch dimensions.
    const dim3 blocks( ( w + threadsX - 1 ) / threadsX, ( h + threadsY - 1 ) / threadsY, b );
    const dim3 thrds( threadsX, threadsY, 1 );

    // Kernal launch.
    AT_DISPATCH_FLOATING_TYPES( s.type(), "sigmoid_cpp_backward_cuda", ( [&] {
        k_sigmoid_cpp_backward<scalar_t><<<blocks, thrds>>>( 
            grad.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            s.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            output.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>() );
    } ) );

    return { output };
}
