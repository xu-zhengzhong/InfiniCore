#ifndef __LOG_MOORE_KERNEL_H__
#define __LOG_MOORE_KERNEL_H__

namespace op::log::moore {

struct LogOp {
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &input) const {
        if constexpr (std::is_same_v<T, half2>) {
            float f1 = __low2float(input);
            float f2 = __high2float(input);
            return __floats2half2_rn(::logf(f1), ::logf(f2));
        } else if constexpr (std::is_same_v<T, half>) {
            float v = __half2float(input);
            return __float2half(::logf(v));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float v = __bfloat162float(input);
            return __float2bfloat16(::logf(v));
        } else if constexpr (std::is_same_v<T, float>) {
            return ::logf(input);
        } else {
            return ::log(input);
        }
    }
};

} // namespace op::log::moore
#endif // __LOG_MOORE_KERNEL_H__