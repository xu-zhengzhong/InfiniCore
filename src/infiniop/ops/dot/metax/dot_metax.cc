#include "dot_metax.h"
#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"

namespace op::dot::metax {

struct Descriptor::Opaque {
    std::shared_ptr<device::metax::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc) {

    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    auto result = DotInfo::createDotInfo(x_desc, y_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        result.take(),
        0,
        new Opaque{handle->internal()},
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *x,
    const void *y,
    void *result,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;

    CHECK_STATUS(_opaque->internal->useMcblas(
        (hcStream_t)stream,
        [&](hcblasHandle_t handle) {
            switch (_info.getDtype()) {
            case INFINI_DTYPE_F32:
                CHECK_MCBLAS(hcblasSdot(handle, _info.getSize(), (const float *)x, _info.getIncx(), (const float *)y, _info.getIncy(), (float *)result));
                break;
            case INFINI_DTYPE_F64:
                CHECK_MCBLAS(hcblasDdot(handle, _info.getSize(), (const double *)x, _info.getIncx(), (const double *)y, _info.getIncy(), (double *)result));
                break;
            default:
                return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
            }

            return INFINI_STATUS_SUCCESS;
        }));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::dot::metax