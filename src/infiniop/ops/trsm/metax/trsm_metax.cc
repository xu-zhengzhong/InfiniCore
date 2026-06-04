#include "trsm_metax.h"
#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"

#include <utility>

namespace op::trsm::metax {

struct Descriptor::Opaque {
    std::shared_ptr<device::metax::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopBlasSideMode_t side,
    infiniopBlasFillMode_t uplo,
    infiniopBlasOperation_t trans,
    infiniopBlasDiagType_t diag,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t A_desc,
    infiniopTensorDescriptor_t B_desc) {

    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    auto result = TrsmInfo::createTrsmInfo(side, uplo, trans, diag, alpha_desc, A_desc, B_desc);
    CHECK_RESULT(result);
    auto info = result.take();

    *desc_ptr = new Descriptor(
        info,
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
    const void *A,
    void *B,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;

    auto side = _info.side == INFINIOP_BLAS_SIDE_LEFT ? HCBLAS_SIDE_LEFT : HCBLAS_SIDE_RIGHT;
    auto uplo = _info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER ? HCBLAS_FILL_MODE_UPPER : HCBLAS_FILL_MODE_LOWER;
    auto trans = _info.trans == INFINIOP_BLAS_OP_N ? HCBLAS_OP_N : HCBLAS_OP_T;
    auto diag = _info.diag == INFINIOP_BLAS_DIAG_UNIT ? HCBLAS_DIAG_UNIT : HCBLAS_DIAG_NON_UNIT;
    int m = utils::cast<int>(_info.m);
    int n = utils::cast<int>(_info.n);
    int ldb;

    int lda = utils::cast<int>(_info.A_col_stride);
    if (_info.A_col_stride == 1) {
        uplo = _info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER ? HCBLAS_FILL_MODE_LOWER : HCBLAS_FILL_MODE_UPPER;
        trans = _info.trans == INFINIOP_BLAS_OP_N ? HCBLAS_OP_T : HCBLAS_OP_N;
        lda = utils::cast<int>(_info.A_row_stride);
    }

    if (_info.B_row_stride == 1) {
        ldb = utils::cast<int>(_info.B_col_stride);
    } else {
        side = _info.side == INFINIOP_BLAS_SIDE_LEFT ? HCBLAS_SIDE_RIGHT : HCBLAS_SIDE_LEFT;
        trans = trans == HCBLAS_OP_N ? HCBLAS_OP_T : HCBLAS_OP_N;
        ldb = utils::cast<int>(_info.B_row_stride);
        std::swap(m, n);
    }

    const infiniDtype_t data_type = _info.data_type;

    CHECK_STATUS(_opaque->internal->useMcblas(
        (hcStream_t)stream,
        [&](hcblasHandle_t handle) {
            CHECK_MCBLAS(hcblasSetPointerMode(handle, HCBLAS_POINTER_MODE_DEVICE));

            switch (data_type) {
            case INFINI_DTYPE_F32:
                CHECK_MCBLAS(hcblasStrsm(
                    handle, side, uplo, trans, diag,
                    m, n,
                    static_cast<const float *>(alpha),
                    static_cast<const float *>(A), lda,
                    static_cast<float *>(B), ldb));
                break;
            case INFINI_DTYPE_F64:
                CHECK_MCBLAS(hcblasDtrsm(
                    handle, side, uplo, trans, diag,
                    m, n,
                    static_cast<const double *>(alpha),
                    static_cast<const double *>(A), lda,
                    static_cast<double *>(B), ldb));
                break;
            default:
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }

            return INFINI_STATUS_SUCCESS;
        }));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::trsm::metax
