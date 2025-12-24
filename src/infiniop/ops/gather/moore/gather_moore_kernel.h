#ifndef __GATHER_MOORE_KERNEL_H__
#define __GATHER_MOORE_KERNEL_H__

#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>
#include <cstddef>
#include <cstdint>

namespace op::gather::moore {

template <typename T, typename TIdx>
struct GatherOp {
    __device__ __forceinline__ void operator()(
        size_t t,                       // linear index in output/index
        T *output,
        const T *input,
        const TIdx *index,
        size_t num_out,
        int ndim,
        int dim,
        const size_t *out_shape,
        const size_t *in_shape,
        const ptrdiff_t *in_strides) const {

        (void)num_out;

        // inner = prod(out_shape[dim+1:])
        size_t inner = 1;
        for (int k = dim + 1; k < ndim; ++k) inner *= out_shape[k];

        size_t dim_size_out = out_shape[dim];
        size_t dim_size_in = in_shape[dim];

        TIdx g = index[t];
        if (g < 0 || (size_t)g >= dim_size_in) {
            output[t] = (T)0;
            return;
        }

        size_t inner_idx = (inner == 0) ? 0 : (t % inner);
        size_t tmp = (inner == 0) ? t : (t / inner);

        size_t outer_flat = (dim_size_out == 0) ? tmp : (tmp / dim_size_out);

        ptrdiff_t in_offset = 0;

        // prefix dims
        size_t rem = outer_flat;
        for (int k = dim - 1; k >= 0; --k) {
            size_t sz = out_shape[k];
            size_t coord = (sz == 0) ? 0 : (rem % sz);
            rem = (sz == 0) ? 0 : (rem / sz);
            in_offset += (ptrdiff_t)coord * in_strides[k];
            if (k == 0) break;
        }

        // dim
        in_offset += (ptrdiff_t)g * in_strides[dim];

        // suffix dims
        size_t rem2 = inner_idx;
        for (int k = ndim - 1; k >= dim + 1; --k) {
            size_t sz = out_shape[k];
            size_t coord = (sz == 0) ? 0 : (rem2 % sz);
            rem2 = (sz == 0) ? 0 : (rem2 / sz);
            in_offset += (ptrdiff_t)coord * in_strides[k];
        }

        output[t] = input[in_offset];
    }
};

} // namespace op::gather::moore

#endif // __GATHER_MOORE_KERNEL_H__