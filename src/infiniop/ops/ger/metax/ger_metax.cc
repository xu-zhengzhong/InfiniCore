#include "ger_metax.h"
#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"

namespace op::ger::metax {

struct Descriptor::Opaque {
    std::shared_ptr<device::metax::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t A_desc) {

    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    auto result = GerInfo::createGerInfo(alpha_desc, x_desc, y_desc, A_desc);
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
    const void *y,
    void *A,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;

    int m = utils::cast<int>(_info.m);
    int n = utils::cast<int>(_info.n);
    int incx = utils::cast<int>(_info.incx);
    int incy = utils::cast<int>(_info.incy);

    const int A_row_stride = utils::cast<int>(_info.A_row_stride);
    const int A_col_stride = utils::cast<int>(_info.A_col_stride);
    const infiniDtype_t data_type = _info.data_type;

    int lda = A_col_stride;

    // Row-major to Column-major transpose trick
    if (A_col_stride == 1) {
        lda = A_row_stride;
        std::swap(m, n);
        std::swap(x, y);
        std::swap(incx, incy);
    }

    CHECK_STATUS(_opaque->internal->useMcblas(
        (hcStream_t)stream,
        [&](hcblasHandle_t handle) {
            CHECK_MCBLAS(hcblasSetPointerMode(handle, HCBLAS_POINTER_MODE_DEVICE));

            switch (data_type) {
            case INFINI_DTYPE_F32:
                CHECK_MCBLAS(hcblasSger(
                    handle, m, n,
                    static_cast<const float *>(alpha),
                    static_cast<const float *>(x), incx,
                    static_cast<const float *>(y), incy,
                    static_cast<float *>(A), lda));
                break;
            case INFINI_DTYPE_F64:
                CHECK_MCBLAS(hcblasDger(
                    handle, m, n,
                    static_cast<const double *>(alpha),
                    static_cast<const double *>(x), incx,
                    static_cast<const double *>(y), incy,
                    static_cast<double *>(A), lda));
                break;
            default:
                return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
            }

            return INFINI_STATUS_SUCCESS;
        }));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::ger::metax
