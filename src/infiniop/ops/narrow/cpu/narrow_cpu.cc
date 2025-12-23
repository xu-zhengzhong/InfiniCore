#include "narrow_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <omp.h>
#include <cstring>

namespace op::narrow:: cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t in_desc,
    int dim,
    int64_t start,
    int64_t length) {
    
    auto handle = reinterpret_cast<device:: cpu::Handle *>(handle_);

    auto result = NarrowInfo::create(out_desc, in_desc, dim, start, length);
    CHECK_RESULT(result);
    
    *desc_ptr = new Descriptor(
        nullptr,
        result. take(),
        0,
        handle->device,
        handle->device_id
    );

    return INFINI_STATUS_SUCCESS;
}

template <typename T>
void calculate_cpu_impl(
    const NarrowInfo &info,
    void *output,
    const void *input) {

    auto out_ptr = reinterpret_cast<T *>(output);
    auto in_ptr = reinterpret_cast<const T *>(input);

    size_t outer_size = info.outer_size();
    size_t inner_size = info.inner_size();
    int64_t start = info.start();
    int64_t length = info.length();
    size_t input_dim_size = info.input_shape()[info.dim()];

#pragma omp parallel for collapse(2)
    for (size_t outer = 0; outer < outer_size; ++outer) {
        for (int64_t i = 0; i < length; ++i) {
            size_t in_offset = (outer * input_dim_size + start + i) * inner_size;
            size_t out_offset = (outer * length + i) * inner_size;
            
            std::memcpy(out_ptr + out_offset, 
                       in_ptr + in_offset, 
                       inner_size * sizeof(T));
        }
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {

    auto dtype = _info.dtype();

    switch (dtype) {
    case INFINI_DTYPE_F16: 
        cpu::calculate_cpu_impl<fp16_t>(_info, output, input);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_BF16:
        cpu::calculate_cpu_impl<bf16_t>(_info, output, input);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F32:
        cpu::calculate_cpu_impl<float>(_info, output, input);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F64:
        cpu::calculate_cpu_impl<double>(_info, output, input);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_I8:
        cpu::calculate_cpu_impl<int8_t>(_info, output, input);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_U8:
        cpu::calculate_cpu_impl<uint8_t>(_info, output, input);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_I16:
        cpu::calculate_cpu_impl<int16_t>(_info, output, input);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_U16:
        cpu::calculate_cpu_impl<uint16_t>(_info, output, input);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_I32:
        cpu::calculate_cpu_impl<int32_t>(_info, output, input);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_U32:
        cpu::calculate_cpu_impl<uint32_t>(_info, output, input);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_I64:
        cpu::calculate_cpu_impl<int64_t>(_info, output, input);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_U64:
        cpu::calculate_cpu_impl<uint64_t>(_info, output, input);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::narrow::cpu