#include "rot_metax.h"
#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"

namespace op::rot::metax {

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
    auto result = RotInfo::createRotInfo(x_desc, y_desc);
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
    void *x,
    void *y,
    const void *c,
    const void *s,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;

    CHECK_STATUS(_opaque->internal->useMcblas(
        (hcStream_t)stream,
        [&](hcblasHandle_t handle) {
            CHECK_MCBLAS(hcblasSetPointerMode(handle, HCBLAS_POINTER_MODE_HOST));

            switch (_info.getDtype()) {
            case INFINI_DTYPE_F32:
                CHECK_MCBLAS(hcblasSrot(handle, _info.getSize(), (float *)x, _info.getIncx(), (float *)y, _info.getIncy(), (const float *)c, (const float *)s));
                break;
            case INFINI_DTYPE_F64:
                CHECK_MCBLAS(hcblasDrot(handle, _info.getSize(), (double *)x, _info.getIncx(), (double *)y, _info.getIncy(), (const double *)c, (const double *)s));
                break;
            default:
                return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
            }

            return INFINI_STATUS_SUCCESS;
        }));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::rot::metax
