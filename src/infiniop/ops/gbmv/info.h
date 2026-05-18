#ifndef __GBMV_INFO_H__
#define __GBMV_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include "infiniop/ops/gbmv.h"

struct GbmvInfo {
private:
    GbmvInfo() = default;

public:
    size_t m;
    size_t n;
    size_t kl;
    size_t ku;
    ptrdiff_t A_row_stride;
    ptrdiff_t A_col_stride;
    ptrdiff_t incx;
    ptrdiff_t incy;
    infiniDtype_t data_type;
    infiniopBlasOperation_t trans;

    static utils::Result<GbmvInfo> createGbmvInfo(
        infiniopBlasOperation_t trans,
        size_t kl,
        size_t ku,
        infiniopTensorDescriptor_t alpha_desc,
        infiniopTensorDescriptor_t A_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t beta_desc,
        infiniopTensorDescriptor_t y_desc) {

        CHECK_OR_RETURN(trans == INFINIOP_BLAS_OP_N || trans == INFINIOP_BLAS_OP_T, INFINI_STATUS_BAD_PARAM);
        CHECK_OR_RETURN(alpha_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(A_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(x_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(beta_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(y_desc != nullptr, INFINI_STATUS_NULL_POINTER);

        CHECK_OR_RETURN(alpha_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(beta_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(A_desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(x_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(y_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto data_type = A_desc->dtype();
        CHECK_DTYPE(data_type, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
        CHECK_OR_RETURN(alpha_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(x_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(beta_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(y_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);

        auto band_rows = A_desc->dim(0);
        auto n = A_desc->dim(1);
        CHECK_OR_RETURN(band_rows >= kl + ku + 1, INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto m = y_desc->dim(0);
        if (trans == INFINIOP_BLAS_OP_N) {
            CHECK_OR_RETURN(x_desc->dim(0) == n, INFINI_STATUS_BAD_TENSOR_SHAPE);
        } else {
            m = x_desc->dim(0);
            CHECK_OR_RETURN(y_desc->dim(0) == n, INFINI_STATUS_BAD_TENSOR_SHAPE);
        }

        auto A_row_stride = A_desc->stride(0);
        auto A_col_stride = A_desc->stride(1);
        CHECK_OR_RETURN(A_row_stride == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);

        auto incx = x_desc->stride(0);
        auto incy = y_desc->stride(0);

        return utils::Result<GbmvInfo>(GbmvInfo{
            m,
            n,
            kl,
            ku,
            A_row_stride,
            A_col_stride,
            incx,
            incy,
            data_type,
            trans});
    }
};

#endif // __GBMV_INFO_H__
