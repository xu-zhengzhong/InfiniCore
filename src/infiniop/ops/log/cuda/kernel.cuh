#ifndef __LOG_CUDA_H__
#define __LOG_CUDA_H__

#include <cmath>

namespace op::log::cuda {

typedef struct LogOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            // half2 vectorized optimization
            return h2log(x);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // BF16
            const float x_f = __bfloat162float(x);
            return __float2bfloat16(::logf(x_f));
        } else if constexpr (std::is_same_v<T, half>) {
            // FP16
            const float x_f = __half2float(x);
            return __float2half(::logf(x_f));
        } else if constexpr (std::is_same_v<T, float>) {
            // FP32
            return ::logf(x);
        } else if constexpr (std::is_same_v<T, double>) {
            // FP64
            return ::log(x);
        }
    }
} LogOp;

} // namespace op::log::cuda

#endif // __LOG_CUDA_H__
