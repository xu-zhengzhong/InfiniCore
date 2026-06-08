#include "axpby_metax.h"
#include "../../../devices/metax/metax_handle.h"

namespace op::axpby::metax {

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc) {
    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    CHECK_DTYPE(x_desc->dtype(), INFINI_DTYPE_F32, INFINI_DTYPE_F64);
    auto result = AxpbyInfo::create(x_desc, y_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        result.take(),
        0,
        new Opaque(),
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *x,
    void *y,
    float alpha,
    float beta,
    void *stream) const {
    (void)workspace;
    (void)workspace_size;
    return launchAxpby(_info, x, y, alpha, beta, stream);
}

} // namespace op::axpby::metax
