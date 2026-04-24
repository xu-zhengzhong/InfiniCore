#ifndef __AXPY_CUDA_CUH__
#define __AXPY_CUDA_CUH__

#include <cstddef>

namespace op::cuda {

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__global__ void axpy_kernel(
    size_t n,
    const Tdata *alpha,
    const Tdata *x,
    ptrdiff_t incx,
    Tdata *y,
    ptrdiff_t incy) {

    const Tcompute alpha_v = static_cast<Tcompute>(alpha[0]);

    for (size_t i = threadIdx.x; i < n; i += BLOCK_SIZE) {
        const Tcompute x_v = static_cast<Tcompute>(x[i * incx]);
        const Tcompute y_v = static_cast<Tcompute>(y[i * incy]);
        y[i * incy] = static_cast<Tdata>(alpha_v * x_v + y_v);
    }
}

} // namespace op::cuda

#endif // __AXPY_CUDA_CUH__
