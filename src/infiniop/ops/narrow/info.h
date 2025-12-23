#ifndef __NARROW_INFO_H__
#define __NARROW_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op:: narrow {

class NarrowInfo {
    NarrowInfo() = default;

public:
    int _dtype;
    int _dim;
    int64_t _start;
    int64_t _length;
    std::vector<size_t> _input_shape;
    std::vector<size_t> _output_shape;
    size_t _outer_size;  // 外层循环大小
    size_t _inner_size;  // 内层循环大小

    int dtype() const { return _dtype; }
    int dim() const { return _dim; }
    int64_t start() const { return _start; }
    int64_t length() const { return _length; }
    size_t outer_size() const { return _outer_size; }
    size_t inner_size() const { return _inner_size; }
    const std::vector<size_t>& input_shape() const { return _input_shape; }
    const std::vector<size_t>& output_shape() const { return _output_shape; }

    static utils::Result<NarrowInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t in_desc,
        int dim,
        int64_t start,
        int64_t length) {

        // 1. 检查数据类型一致性
        if (out_desc->dtype() != in_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // 2. 获取输入形状
        int ndim = in_desc->ndim();
        if (dim < 0 || dim >= ndim) {
            return INFINI_STATUS_BAD_PARAM;
        }

        const auto &in_shape = in_desc->shape();
        int64_t dim_size = in_shape[dim];

        // 3. 处理负索引
        if (start < 0) {
            start += dim_size;
        }

        // 4. 边界检查
        if (start < 0 || start >= dim_size) {
            return INFINI_STATUS_BAD_PARAM;
        }
        if (length <= 0 || start + length > dim_size) {
            return INFINI_STATUS_BAD_PARAM;
        }

        // 5. 验证输出形状
        const auto &out_shape = out_desc->shape();
        if (out_desc->ndim() != static_cast<uint64_t>(ndim)) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        for (int i = 0; i < ndim; ++i) {
            if (i == dim) {
                if (out_shape[i] != static_cast<size_t>(length)) {
                    return INFINI_STATUS_BAD_TENSOR_SHAPE;
                }
            } else {
                if (out_shape[i] != in_shape[i]) {
                    return INFINI_STATUS_BAD_TENSOR_SHAPE;
                }
            }
        }

        // 6. 计算 outer_size 和 inner_size
        size_t outer_size = 1;
        for (int i = 0; i < dim; ++i) {
            outer_size *= in_shape[i];
        }

        size_t inner_size = 1;
        for (int i = dim + 1; i < ndim; ++i) {
            inner_size *= in_shape[i];
        }

        // 7. 构造 Info 对象
        std::vector<size_t> input_shape(in_shape.begin(), in_shape.end());
        std::vector<size_t> output_shape(out_shape.begin(), out_shape.end());

        return utils::Result<NarrowInfo>(NarrowInfo{
            in_desc->dtype(),
            dim,
            start,
            length,
            std::move(input_shape),
            std::move(output_shape),
            outer_size,
            inner_size
        });
    }
};

} // namespace op:: narrow

#endif // __NARROW_INFO_H__