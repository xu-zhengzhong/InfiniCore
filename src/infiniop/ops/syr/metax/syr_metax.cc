#include "syr_metax.h"
#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"

namespace op::syr::metax {

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
    auto result = SyrInfo::createSyrInfo(uplo, alpha_desc, x_desc, A_desc);
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
    const int A_row_stride = utils::cast<int>(_info.A_row_stride);
    const int A_col_stride = utils::cast<int>(_info.A_col_stride);
    const infiniDtype_t data_type = _info.data_type;

    auto uplo = _info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER ? HCBLAS_FILL_MODE_UPPER : HCBLAS_FILL_MODE_LOWER;
    auto lda = A_col_stride;

    // Row-major tensors are interpreted as column-major transposed matrices.
    if (A_col_stride == 1) {
        uplo = _info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER ? HCBLAS_FILL_MODE_LOWER : HCBLAS_FILL_MODE_UPPER;
        lda = A_row_stride;
    }

    CHECK_STATUS(_opaque->internal->useMcblas(
        (hcStream_t)stream,
        [&](hcblasHandle_t handle) {
            CHECK_MCBLAS(hcblasSetPointerMode(handle, HCBLAS_POINTER_MODE_DEVICE));

            switch (data_type) {
            case INFINI_DTYPE_F32:
                CHECK_MCBLAS(hcblasSsyr(
                    handle, uplo, n,
                    static_cast<const float *>(alpha),
                    static_cast<const float *>(x), incx,
                    static_cast<float *>(A), lda));
                break;
            case INFINI_DTYPE_F64:
                CHECK_MCBLAS(hcblasDsyr(
                    handle, uplo, n,
                    static_cast<const double *>(alpha),
                    static_cast<const double *>(x), incx,
                    static_cast<double *>(A), lda));
                break;
            default:
                return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
            }

            return INFINI_STATUS_SUCCESS;
        }));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::syr::metax
