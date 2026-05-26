#include "hemm_metax.h"
#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"

namespace op::hemm::metax {

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
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t A_desc,
    infiniopTensorDescriptor_t B_desc,
    infiniopTensorDescriptor_t beta_desc,
    infiniopTensorDescriptor_t C_desc) {

    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    auto result = HemmInfo::createHemmInfo(side, uplo, alpha_desc, A_desc, B_desc, beta_desc, C_desc);
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
    const void *B,
    const void *beta,
    void *C,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;

    auto side = _info.side == INFINIOP_BLAS_SIDE_LEFT ? HCBLAS_SIDE_LEFT : HCBLAS_SIDE_RIGHT;
    auto uplo = _info.uplo == INFINIOP_BLAS_FILL_MODE_UPPER ? HCBLAS_FILL_MODE_UPPER : HCBLAS_FILL_MODE_LOWER;
    const int m = utils::cast<int>(_info.m);
    const int n = utils::cast<int>(_info.n);
    const int lda = utils::cast<int>(_info.A_col_stride);
    const int ldb = utils::cast<int>(_info.B_col_stride);
    const int ldc = utils::cast<int>(_info.C_col_stride);

    const infiniDtype_t data_type = _info.data_type;

    CHECK_STATUS(_opaque->internal->useMcblas(
        (hcStream_t)stream,
        [&](hcblasHandle_t handle) {
            CHECK_MCBLAS(hcblasSetPointerMode(handle, HCBLAS_POINTER_MODE_DEVICE));

            switch (data_type) {
            case INFINI_DTYPE_C64:
                CHECK_MCBLAS(hcblasChemm(
                    handle, side, uplo,
                    m, n,
                    static_cast<const hcFloatComplex *>(alpha),
                    static_cast<const hcFloatComplex *>(A), lda,
                    static_cast<const hcFloatComplex *>(B), ldb,
                    static_cast<const hcFloatComplex *>(beta),
                    static_cast<hcFloatComplex *>(C), ldc));
                break;
            case INFINI_DTYPE_C128:
                CHECK_MCBLAS(hcblasZhemm(
                    handle, side, uplo,
                    m, n,
                    static_cast<const hcDoubleComplex *>(alpha),
                    static_cast<const hcDoubleComplex *>(A), lda,
                    static_cast<const hcDoubleComplex *>(B), ldb,
                    static_cast<const hcDoubleComplex *>(beta),
                    static_cast<hcDoubleComplex *>(C), ldc));
                break;
            default:
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }

            return INFINI_STATUS_SUCCESS;
        }));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::hemm::metax
