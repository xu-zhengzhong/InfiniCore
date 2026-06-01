#include "hpmv_metax.h"
#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"

namespace op::hpmv::metax {

struct Descriptor::Opaque {
    std::shared_ptr<device::metax::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopBlasFillMode_t uplo,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t AP_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t beta_desc,
    infiniopTensorDescriptor_t y_desc) {

    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    auto result = HpmvInfo::createHpmvInfo(uplo, alpha_desc, AP_desc, x_desc, beta_desc, y_desc);
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
    const void *alpha,
    const void *AP,
    const void *x,
    const void *beta,
    void *y,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;

    const int n = utils::cast<int>(_info.n);
    const int incx = utils::cast<int>(_info.incx);
    const int incy = utils::cast<int>(_info.incy);
    const infiniDtype_t data_type = _info.data_type;

    auto uplo = _info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER ? HCBLAS_FILL_MODE_UPPER : HCBLAS_FILL_MODE_LOWER;

    CHECK_STATUS(_opaque->internal->useMcblas(
        (hcStream_t)stream,
        [&](hcblasHandle_t handle) {
            CHECK_MCBLAS(hcblasSetPointerMode(handle, HCBLAS_POINTER_MODE_DEVICE));

            switch (data_type) {
            case INFINI_DTYPE_C64:
                CHECK_MCBLAS(hcblasChpmv(
                    handle, uplo, n,
                    static_cast<const hcFloatComplex *>(alpha),
                    static_cast<const hcFloatComplex *>(AP),
                    static_cast<const hcFloatComplex *>(x), incx,
                    static_cast<const hcFloatComplex *>(beta),
                    static_cast<hcFloatComplex *>(y), incy));
                break;
            case INFINI_DTYPE_C128:
                CHECK_MCBLAS(hcblasZhpmv(
                    handle, uplo, n,
                    static_cast<const hcDoubleComplex *>(alpha),
                    static_cast<const hcDoubleComplex *>(AP),
                    static_cast<const hcDoubleComplex *>(x), incx,
                    static_cast<const hcDoubleComplex *>(beta),
                    static_cast<hcDoubleComplex *>(y), incy));
                break;
            default:
                return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
            }

            return INFINI_STATUS_SUCCESS;
        }));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::hpmv::metax
