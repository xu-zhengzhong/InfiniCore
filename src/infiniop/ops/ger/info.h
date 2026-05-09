#ifndef __GER_INFO_H__
#define __GER_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

struct GerInfo {
private:
    GerInfo() = default;

public:
    size_t m;
    size_t n;
    ptrdiff_t incx;
    ptrdiff_t incy;
    ptrdiff_t A_row_stride;
    ptrdiff_t A_col_stride;
    infiniDtype_t data_type;

    static utils::Result<GerInfo> createGerInfo(
        infiniopTensorDescriptor_t alpha_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t A_desc) {

        CHECK_OR_RETURN(alpha_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(x_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(y_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(A_desc != nullptr, INFINI_STATUS_NULL_POINTER);

        CHECK_OR_RETURN(alpha_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(A_desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(x_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(y_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto data_type = A_desc->dtype();
        CHECK_DTYPE(data_type, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
        CHECK_OR_RETURN(alpha_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(x_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(y_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);

        auto m = A_desc->dim(0);
        auto n = A_desc->dim(1);
        CHECK_OR_RETURN(x_desc->dim(0) == m, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(y_desc->dim(0) == n, INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto A_row_stride = A_desc->stride(0);
        auto A_col_stride = A_desc->stride(1);
        auto incx = x_desc->stride(0);
        auto incy = y_desc->stride(0);

        return utils::Result<GerInfo>(GerInfo{
            m,
            n,
            incx,
            incy,
            A_row_stride,
            A_col_stride,
            data_type});
    }
};

#endif // __GER_INFO_H__
