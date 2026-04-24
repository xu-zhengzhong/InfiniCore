#ifndef __SCAL_CUDA_CUH__
#define __SCAL_CUDA_CUH__

#include <cstddef>

namespace op::cuda {

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__global__ void scal_kernel(
    size_t n,
    const Tdata *alpha,
    Tdata *x,
    ptrdiff_t incx, ) {

    const Tcompute alpha_v = static_cast<Tcompute>(alpha[0]);

    for (size_t i = threadIdx.x; i < n; i += BLOCK_SIZE) {
        const Tcompute x_v = static_cast<Tcompute>(x[i * incx]);
        x[i * incx] = static_cast<Tdata>(alpha_v * x_v);
    }
}

} // namespace op::cuda

#endif // __SCAL_CUDA_CUH__
