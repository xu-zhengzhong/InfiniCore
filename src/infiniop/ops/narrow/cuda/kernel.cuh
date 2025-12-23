#ifndef __NARROW_CUDA_H__
#define __NARROW_CUDA_H__

#include <cstdint>

namespace op::narrow::cuda {

template <typename T>
__global__ void narrow_kernel(
    T * __restrict__ output,
    const T * __restrict__ input,
    size_t outer_size,
    size_t input_dim_size,
    size_t output_dim_size,
    size_t inner_size,
    int64_t start) {
    
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    size_t total_elements = outer_size * output_dim_size * inner_size;
    
    for (size_t idx = tid; idx < total_elements; idx += stride) {
        size_t inner_idx = idx % inner_size;
        size_t temp = idx / inner_size;
        size_t dim_idx = temp % output_dim_size;
        size_t outer_idx = temp / output_dim_size;
        
        size_t in_offset = (outer_idx * input_dim_size + start + dim_idx) * inner_size + inner_idx;
        output[idx] = input[in_offset];
    }
}

} // namespace op::narrow::cuda

#endif // __NARROW_CUDA_H__