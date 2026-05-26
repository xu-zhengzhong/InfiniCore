#include "syrk_metax.h"
#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"

namespace op::syrk::metax {

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
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t A_desc,
    infiniopTensorDescriptor_t beta_desc,
    infiniopTensorDescriptor_t C_desc) {

    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    auto result = SyrkInfo::createSyrkInfo(uplo, trans, alpha_desc, A_desc, beta_desc, C_desc);
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
    const void *A,
    const void *beta,
    void *C,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;

    auto uplo = _info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER ? HCBLAS_FILL_MODE_UPPER : HCBLAS_FILL_MODE_LOWER;
    auto trans = _info.trans == INFINIOP_BLAS_OP_N ? HCBLAS_OP_N : HCBLAS_OP_T;
    const int n = utils::cast<int>(_info.n);
    const int k = utils::cast<int>(_info.k);
    const int A_cols = _info.trans == INFINIOP_BLAS_OP_N ? k : n;

    int lda = utils::cast<int>(_info.A_col_stride);
    if (_info.A_col_stride == 1) {
        trans = _info.trans == INFINIOP_BLAS_OP_N ? HCBLAS_OP_T : HCBLAS_OP_N;
        lda = std::max(utils::cast<int>(_info.A_row_stride), A_cols);
    }

    int ldc = utils::cast<int>(_info.C_col_stride);
    if (_info.C_col_stride == 1) {
        uplo = _info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER ? HCBLAS_FILL_MODE_LOWER : HCBLAS_FILL_MODE_UPPER;
        ldc = utils::cast<int>(_info.C_row_stride);
    }

    const infiniDtype_t data_type = _info.data_type;

    CHECK_STATUS(_opaque->internal->useMcblas(
        (hcStream_t)stream,
        [&](hcblasHandle_t handle) {
            CHECK_MCBLAS(hcblasSetPointerMode(handle, HCBLAS_POINTER_MODE_DEVICE));

            switch (data_type) {
            case INFINI_DTYPE_F32:
                CHECK_MCBLAS(hcblasSsyrk(
                    handle, uplo, trans,
                    n, k,
                    static_cast<const float *>(alpha),
                    static_cast<const float *>(A), lda,
                    static_cast<const float *>(beta),
                    static_cast<float *>(C), ldc));
                break;
            case INFINI_DTYPE_F64:
                CHECK_MCBLAS(hcblasDsyrk(
                    handle, uplo, trans,
                    n, k,
                    static_cast<const double *>(alpha),
                    static_cast<const double *>(A), lda,
                    static_cast<const double *>(beta),
                    static_cast<double *>(C), ldc));
                break;
            default:
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }

            return INFINI_STATUS_SUCCESS;
        }));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::syrk::metax
