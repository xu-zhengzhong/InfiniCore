#include "spvv_metax.h"
#include "../../../devices/metax/metax_handle.h"

namespace op::spvv::metax {

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopSpVecDescriptor_t a_desc,
    infiniopTensorDescriptor_t x_desc) {
    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    auto dtype = y_desc->dtype();
    auto index_dtype = a_desc->indicesDesc()->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F32);

    auto result = SpVVInfo::create(y_desc, a_desc, x_desc);
    CHECK_RESULT(result);
    auto info = result.take();

    CHECK_OR_RETURN(info.x_vector.stride == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);

    auto opaque = new Opaque();

    *desc_ptr = new Descriptor(
        dtype,
        index_dtype,
        info,
        a_desc,
        0,
        opaque,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    float alpha,
    float beta,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    (void)workspace;

    switch (_index_dtype) {
    case INFINI_DTYPE_I32:
        return launchCalculateI32(
            y,
            _a_desc->values(),
            _a_desc->indices(),
            x,
            _info.nnz,
            alpha,
            beta,
            stream);
    case INFINI_DTYPE_I64:
        return launchCalculateI64(
            y,
            _a_desc->values(),
            _a_desc->indices(),
            x,
            _info.nnz,
            alpha,
            beta,
            stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::spvv::metax
