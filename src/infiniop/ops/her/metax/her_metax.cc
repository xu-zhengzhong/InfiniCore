#include "her_metax.h"
#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"

namespace op::her::metax {

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
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t A_desc) {

    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    auto result = HerInfo::createHerInfo(uplo, alpha_desc, x_desc, A_desc);
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
    const void *x,
    void *A,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;

    const int n = utils::cast<int>(_info.n);
    const int incx = utils::cast<int>(_info.incx);
    const int lda = utils::cast<int>(_info.A_col_stride);
    const infiniDtype_t data_type = _info.data_type;

    auto uplo = _info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER ? HCBLAS_FILL_MODE_UPPER : HCBLAS_FILL_MODE_LOWER;

    CHECK_STATUS(_opaque->internal->useMcblas(
        (hcStream_t)stream,
        [&](hcblasHandle_t handle) {
            CHECK_MCBLAS(hcblasSetPointerMode(handle, HCBLAS_POINTER_MODE_DEVICE));

            switch (data_type) {
            case INFINI_DTYPE_C64:
                CHECK_MCBLAS(hcblasCher(
                    handle, uplo, n,
                    static_cast<const float *>(alpha),
                    static_cast<const hcFloatComplex *>(x), incx,
                    static_cast<hcFloatComplex *>(A), lda));
                break;
            case INFINI_DTYPE_C128:
                CHECK_MCBLAS(hcblasZher(
                    handle, uplo, n,
                    static_cast<const double *>(alpha),
                    static_cast<const hcDoubleComplex *>(x), incx,
                    static_cast<hcDoubleComplex *>(A), lda));
                break;
            default:
                return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
            }

            return INFINI_STATUS_SUCCESS;
        }));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::her::metax
