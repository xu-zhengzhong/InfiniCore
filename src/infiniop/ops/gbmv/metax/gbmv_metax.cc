#include "gbmv_metax.h"
#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"

namespace op::gbmv::metax {

struct Descriptor::Opaque {
    std::shared_ptr<device::metax::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopBlasOperation_t trans,
    size_t kl,
    size_t ku,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t A_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t beta_desc,
    infiniopTensorDescriptor_t y_desc) {

    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    auto result = GbmvInfo::createGbmvInfo(trans, kl, ku, alpha_desc, A_desc, x_desc, beta_desc, y_desc);
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
    const void *x,
    const void *beta,
    void *y,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;

    const int m = utils::cast<int>(_info.m);
    const int n = utils::cast<int>(_info.n);
    const int kl = utils::cast<int>(_info.kl);
    const int ku = utils::cast<int>(_info.ku);
    const int incx = utils::cast<int>(_info.incx);
    const int incy = utils::cast<int>(_info.incy);
    const int A_row_stride = utils::cast<int>(_info.A_row_stride);
    const int A_col_stride = utils::cast<int>(_info.A_col_stride);
    const infiniDtype_t data_type = _info.data_type;

    CHECK_OR_RETURN(A_row_stride == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);

    auto trans = _info.trans == INFINIOP_BLAS_OP_N ? HCBLAS_OP_N : HCBLAS_OP_T;
    auto lda = A_col_stride;

    CHECK_STATUS(_opaque->internal->useMcblas(
        (hcStream_t)stream,
        [&](hcblasHandle_t handle) {
            CHECK_MCBLAS(hcblasSetPointerMode(handle, HCBLAS_POINTER_MODE_DEVICE));

            switch (data_type) {
            case INFINI_DTYPE_F32:
                CHECK_MCBLAS(hcblasSgbmv(
                    handle, trans, m, n, kl, ku,
                    static_cast<const float *>(alpha),
                    static_cast<const float *>(A), lda,
                    static_cast<const float *>(x), incx,
                    static_cast<const float *>(beta),
                    static_cast<float *>(y), incy));
                break;
            case INFINI_DTYPE_F64:
                CHECK_MCBLAS(hcblasDgbmv(
                    handle, trans, m, n, kl, ku,
                    static_cast<const double *>(alpha),
                    static_cast<const double *>(A), lda,
                    static_cast<const double *>(x), incx,
                    static_cast<const double *>(beta),
                    static_cast<double *>(y), incy));
                break;
            default:
                return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
            }

            return INFINI_STATUS_SUCCESS;
        }));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gbmv::metax
