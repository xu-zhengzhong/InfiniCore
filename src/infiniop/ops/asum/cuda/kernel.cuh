#ifndef __ASUM_CUDA_CUH__
#define __ASUM_CUDA_CUH__

#include "../../../reduce/cuda/reduce.cuh"
#include <type_traits>

namespace op::cuda {

template <typename T>
__device__ __forceinline__ T asum_abs(T v) {
    return v < T(0) ? -v : v;
}

template <>
__device__ __forceinline__ float asum_abs<float>(float v) {
    return fabsf(v);
}

template <>
__device__ __forceinline__ double asum_abs<double>(double v) {
    return fabs(v);
}

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tout, typename Tcompute>
__global__ void asum_kernel(
    size_t n,
    const Tdata *x,
    ptrdiff_t incx,
    Tout *result) {

    Tcompute sum = 0;

    for (size_t i = threadIdx.x; i < n; i += BLOCK_SIZE) {
        sum += asum_abs<Tcompute>(Tcompute(x[i * incx]));
    }

    using BlockReduce = cub::BlockReduce<Tcompute, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    Tcompute block_sum = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
        result[0] = static_cast<Tout>(block_sum);
    }
}

} // namespace op::cuda

#endif // __ASUM_CUDA_CUH__