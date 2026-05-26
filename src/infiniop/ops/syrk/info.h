#ifndef __SYRK_INFO_H__
#define __SYRK_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include "infiniop/ops/syrk.h"

struct SyrkInfo {
private:
    SyrkInfo() = default;

public:
    size_t n;
    size_t k;
    infiniopBlasFillMode_t uplo;
    infiniopBlasOperation_t trans;
    ptrdiff_t A_row_stride;
    ptrdiff_t A_col_stride;
    ptrdiff_t C_row_stride;
    ptrdiff_t C_col_stride;
    infiniDtype_t data_type;

    static utils::Result<SyrkInfo> createSyrkInfo(
        infiniopBlasFillMode_t uplo,
        infiniopBlasOperation_t trans,
        infiniopTensorDescriptor_t alpha_desc,
        infiniopTensorDescriptor_t A_desc,
        infiniopTensorDescriptor_t beta_desc,
        infiniopTensorDescriptor_t C_desc) {

        CHECK_OR_RETURN(uplo == INFINIOP_BLAS_FILL_MODE_UPPER || uplo == INFINIOP_BLAS_FILL_MODE_LOWER, INFINI_STATUS_BAD_PARAM);
        CHECK_OR_RETURN(trans == INFINIOP_BLAS_OP_N || trans == INFINIOP_BLAS_OP_T || trans == INFINIOP_BLAS_OP_C, INFINI_STATUS_BAD_PARAM);
        CHECK_OR_RETURN(alpha_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(A_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(beta_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(C_desc != nullptr, INFINI_STATUS_NULL_POINTER);

        CHECK_OR_RETURN(alpha_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(beta_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(A_desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(C_desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto data_type = C_desc->dtype();
        CHECK_DTYPE(data_type, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
        CHECK_OR_RETURN(alpha_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(A_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(beta_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);

        const auto rows = C_desc->dim(0);
        const auto n = C_desc->dim(1);
        CHECK_OR_RETURN(rows == n, INFINI_STATUS_BAD_TENSOR_SHAPE);

        size_t k;
        if (trans == INFINIOP_BLAS_OP_N) {
            CHECK_OR_RETURN(A_desc->dim(0) == n, INFINI_STATUS_BAD_TENSOR_SHAPE);
            k = A_desc->dim(1);
        } else {
            CHECK_OR_RETURN(A_desc->dim(1) == n, INFINI_STATUS_BAD_TENSOR_SHAPE);
            k = A_desc->dim(0);
        }

        const auto A_row_stride = A_desc->stride(0);
        const auto A_col_stride = A_desc->stride(1);
        const auto C_row_stride = C_desc->stride(0);
        const auto C_col_stride = C_desc->stride(1);
        CHECK_OR_RETURN(A_row_stride == 1 || A_col_stride == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(C_row_stride == 1 || C_col_stride == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);

        return utils::Result<SyrkInfo>(SyrkInfo{
            n,
            k,
            uplo,
            trans,
            A_row_stride,
            A_col_stride,
            C_row_stride,
            C_col_stride,
            data_type});
    }
};

#endif // __SYRK_INFO_H__
