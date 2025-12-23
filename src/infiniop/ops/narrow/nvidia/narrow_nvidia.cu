#include "narrow_nvidia.cuh"
#include "../cuda/kernel.cuh"
#include "../../../handle.h"
#include <cstdint>

namespace op::narrow::nvidia {

template <typename T>
void launch_kernel(
    void *output,
    const void *input,
    const NarrowInfo &info,
    void *stream) {

    auto out_ptr = reinterpret_cast<T *>(output);
    auto in_ptr = reinterpret_cast<const T *>(input);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    size_t outer_size = info.outer_size();
    size_t inner_size = info.inner_size();
    size_t input_dim_size = info.input_shape()[info.dim()];
    size_t output_dim_size = info. length();
    int64_t start = info.start();

    size_t total_elements = outer_size * output_dim_size * inner_size;
    
    size_t block_size = 256;
    size_t grid_size = (total_elements + block_size - 1) / block_size;

    op::narrow:: cuda::narrow_kernel<T>
        <<<grid_size, block_size, 0, cuda_stream>>>(
        out_ptr, in_ptr, outer_size, input_dim_size, output_dim_size, 
        inner_size, start
    );
}

struct Descriptor:: Opaque {};

Descriptor::~Descriptor() {
    if (_opaque) delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t in_desc,
    int dim,
    int64_t start,
    int64_t length) {

    auto info_result = NarrowInfo::create(out_desc, in_desc, dim, start, length);
    if (! info_result) return info_result. status();

    *desc_ptr = new Descriptor(
        new Opaque(), info_result.take(), 0, handle->device, handle->device_id
    );
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {

    auto dtype = _info.dtype();

    switch (dtype) {
    case INFINI_DTYPE_I8:
    case INFINI_DTYPE_U8:
        launch_kernel<uint8_t>(output, input, _info, stream);
        break;
    case INFINI_DTYPE_F16:
    case INFINI_DTYPE_BF16:
    case INFINI_DTYPE_I16:
    case INFINI_DTYPE_U16:
        launch_kernel<uint16_t>(output, input, _info, stream);
        break;
    case INFINI_DTYPE_F32:
    case INFINI_DTYPE_I32:
    case INFINI_DTYPE_U32:
        launch_kernel<uint32_t>(output, input, _info, stream);
        break;
    case INFINI_DTYPE_F64:
    case INFINI_DTYPE_I64:
    case INFINI_DTYPE_U64:
        launch_kernel<uint64_t>(output, input, _info, stream);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::narrow::nvidia