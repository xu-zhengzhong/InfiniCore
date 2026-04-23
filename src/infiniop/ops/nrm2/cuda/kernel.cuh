#ifndef __NRM2_CUDA_CUH__
#define __NRM2_CUDA_CUH__

#include "../../../reduce/cuda/reduce.cuh"
#include <type_traits>

namespace op::cuda {

template <typename T>
__device__ __forceinline__ T nrm2_sqrt(T v) {
    return sqrt(v);
}

template <>
__device__ __forceinline__ float nrm2_sqrt<float>(float v) {
    return sqrtf(v);
}

template <>
__device__ __forceinline__ double nrm2_sqrt<double>(double v) {
    return sqrt(v);
}

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tout, typename Tcompute>
__global__ void nrm2_kernel(
    size_t n,
    const Tdata *x,
    ptrdiff_t incx,
    Tout *result) {

    Tcompute sum_sq = 0;

    for (size_t i = threadIdx.x; i < n; i += BLOCK_SIZE) {
        const Tcompute value = static_cast<Tcompute>(x[i * incx]);
        sum_sq += value * value;
    }

    using BlockReduce = cub::BlockReduce<Tcompute, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    Tcompute block_sum = BlockReduce(temp_storage).Sum(sum_sq);

    if (threadIdx.x == 0) {
        result[0] = static_cast<Tout>(nrm2_sqrt<Tcompute>(block_sum));
    }
}

} // namespace op::cuda

#endif // __NRM2_CUDA_CUH__