#ifndef __SYR2_INFO_H__
#define __SYR2_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include "infiniop/ops/syr2.h"

struct Syr2Info {
private:
    Syr2Info() = default;

public:
    size_t n;
    infiniopBlasFillMode_t uplo;
    ptrdiff_t incx;
    ptrdiff_t incy;
    ptrdiff_t A_row_stride;
    ptrdiff_t A_col_stride;
    infiniDtype_t data_type;

    static utils::Result<Syr2Info> createSyr2Info(
        infiniopBlasFillMode_t uplo,
        infiniopTensorDescriptor_t alpha_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t A_desc) {

        CHECK_OR_RETURN(uplo == INFINIOP_BLAS_FILL_MODE_UPPER || uplo == INFINIOP_BLAS_FILL_MODE_LOWER, INFINI_STATUS_BAD_PARAM);
        CHECK_OR_RETURN(alpha_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(x_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(y_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(A_desc != nullptr, INFINI_STATUS_NULL_POINTER);

        CHECK_OR_RETURN(alpha_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(x_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(y_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(A_desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto data_type = A_desc->dtype();
        CHECK_DTYPE(data_type, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
        CHECK_OR_RETURN(alpha_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(x_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(y_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);

        auto n = x_desc->dim(0);
        CHECK_OR_RETURN(y_desc->dim(0) == n, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(A_desc->dim(0) == n, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(A_desc->dim(1) == n, INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto A_row_stride = A_desc->stride(0);
        auto A_col_stride = A_desc->stride(1);
        CHECK_OR_RETURN(A_row_stride == 1 || A_col_stride == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);

        auto incx = x_desc->stride(0);
        auto incy = y_desc->stride(0);

        return utils::Result<Syr2Info>(Syr2Info{
            n,
            uplo,
            incx,
            incy,
            A_row_stride,
            A_col_stride,
            data_type});
    }
};

#endif // __SYR2_INFO_H__
