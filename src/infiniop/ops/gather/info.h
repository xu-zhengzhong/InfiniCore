#ifndef __GATHER_INFO_H__
#define __GATHER_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>
#include <cstddef>

namespace op::gather {

class GatherInfo {
    GatherInfo() = default;

public:
    int _dtype;
    int _idx_dtype;
    int _ndim;
    int _dim;

    // output / index shape (same)
    std::vector<size_t> _out_shape;

    // input shape/strides
    std::vector<size_t> _in_shape;
    std::vector<ptrdiff_t> _in_strides;

    // index strides (usually contiguous in tests, but keep for completeness)
    std::vector<ptrdiff_t> _idx_strides;

    size_t _num_out;

    int dtype() const { return _dtype; }
    int idx_dtype() const { return _idx_dtype; }
    int ndim() const { return _ndim; }
    int dim() const { return _dim; }
    size_t num_out() const { return _num_out; }

    static utils::Result<GatherInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t in_desc,
        infiniopTensorDescriptor_t idx_desc,
        int dim) {

        // dtype check
        if (out_desc->dtype() != in_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // index dtype
        int idx_type = idx_desc->dtype();
        if (idx_type != INFINI_DTYPE_I32 && idx_type != INFINI_DTYPE_I64) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // ndim check
        if (out_desc->ndim() != idx_desc->ndim() || in_desc->ndim() != idx_desc->ndim()) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        int ndim = static_cast<int>(in_desc->ndim());

        // normalize dim
        int actual_dim = dim;
        if (actual_dim < 0) actual_dim += ndim;
        if (actual_dim < 0 || actual_dim >= ndim) {
            return INFINI_STATUS_BAD_PARAM;
        }

        // out shape == idx shape
        auto out_shape = out_desc->shape();
        auto idx_shape = idx_desc->shape();
        if (out_shape.size() != idx_shape.size()) return INFINI_STATUS_BAD_TENSOR_SHAPE;
        for (size_t i = 0; i < out_shape.size(); ++i) {
            if (out_shape[i] != idx_shape[i]) return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // torch.gather requires: idx.size(d) <= input.size(d) for d != dim
        auto in_shape = in_desc->shape();
        for (int d = 0; d < ndim; ++d) {
            if (d == actual_dim) continue;
            if (idx_shape[d] > in_shape[d]) return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        auto in_strides = in_desc->strides();
        auto idx_strides = idx_desc->strides();

        size_t num_out = out_desc->numel();

        return utils::Result<GatherInfo>(GatherInfo{
            static_cast<int>(in_desc->dtype()),
            idx_type,
            ndim,
            actual_dim,
            std::move(out_shape),
            std::move(in_shape),
            std::move(in_strides),
            std::move(idx_strides),
            num_out});
    }
};

} // namespace op::gather

#endif // __GATHER_INFO_H__