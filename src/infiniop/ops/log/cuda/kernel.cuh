#ifndef __LOG_CUDA_H__
#define __LOG_CUDA_H__
#if ENABLE_METAX_API
  #include <maca_fp16.h>
  #include <maca_bfloat16.h>
  using nv_bfloat162 = __maca_bfloat162;
#else
  #include <cuda_fp16.h>
  #include <cuda_bf16.h>
#endif
#include <cmath>
#include <type_traits>

namespace op::log::cuda {

template<typename T>
__device__ __forceinline__ T log_impl(T v);

template<>
__device__ __forceinline__ float log_impl<float>(float v) { return logf(v); }

template<>
__device__ __forceinline__ half log_impl<half>(half v) {
    float f = __half2float(v);
    return __float2half(logf(f));
}

template<>
__device__ __forceinline__ half2 log_impl<half2>(half2 v) {
    float2 f = __half22float2(v);
    f.x = logf(f.x); f.y = logf(f.y);
    return __float22half2_rn(f);
}

template<>
__device__ __forceinline__ cuda_bfloat16 log_impl<cuda_bfloat16>(cuda_bfloat16 v) {
    float f = __bfloat162float(v);
    return __float2bfloat16(logf(f));
}

template<typename T>
__device__ __forceinline__ T log_impl(T v) {
    return static_cast<T>(logf(static_cast<float>(v)));
}

struct LogOp {
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a) const { return log_impl(a); }
};

} // namespace op::log::cuda
#endif