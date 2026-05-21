#include "trsv_metax.h"
#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"

namespace op::trsv::metax {

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
    infiniopBlasOperation_t trans,
    infiniopBlasDiagType_t diag,
    infiniopTensorDescriptor_t A_desc,
    infiniopTensorDescriptor_t x_desc) {

    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    auto result = TrsvInfo::createTrsvInfo(uplo, trans, diag, A_desc, x_desc);
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
    const void *A,
    void *x,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;

    const int n = utils::cast<int>(_info.n);
    const int incx = utils::cast<int>(_info.incx);
    const int A_row_stride = utils::cast<int>(_info.A_row_stride);
    const int A_col_stride = utils::cast<int>(_info.A_col_stride);
    const infiniDtype_t data_type = _info.data_type;

    auto uplo = _info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER ? HCBLAS_FILL_MODE_UPPER : HCBLAS_FILL_MODE_LOWER;
    auto trans = _info.trans == INFINIOP_BLAS_OP_N ? HCBLAS_OP_N : HCBLAS_OP_T;
    auto diag = _info.diag == INFINIOP_BLAS_DIAG_UNIT ? HCBLAS_DIAG_UNIT : HCBLAS_DIAG_NON_UNIT;
    auto lda = A_col_stride;

    // Row-major tensors are interpreted as column-major transposed matrices.
    if (A_col_stride == 1) {
        uplo = _info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER ? HCBLAS_FILL_MODE_LOWER : HCBLAS_FILL_MODE_UPPER;
        trans = _info.trans == INFINIOP_BLAS_OP_N ? HCBLAS_OP_T : HCBLAS_OP_N;
        lda = A_row_stride;
    }

    CHECK_STATUS(_opaque->internal->useMcblas(
        (hcStream_t)stream,
        [&](hcblasHandle_t handle) {
            switch (data_type) {
            case INFINI_DTYPE_F32:
                CHECK_MCBLAS(hcblasStrsv(
                    handle, uplo, trans, diag, n,
                    static_cast<const float *>(A), lda,
                    static_cast<float *>(x), incx));
                break;
            case INFINI_DTYPE_F64:
                CHECK_MCBLAS(hcblasDtrsv(
                    handle, uplo, trans, diag, n,
                    static_cast<const double *>(A), lda,
                    static_cast<double *>(x), incx));
                break;
            default:
                return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
            }

            return INFINI_STATUS_SUCCESS;
        }));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::trsv::metax
