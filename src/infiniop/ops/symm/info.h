#ifndef __SYMM_INFO_H__
#define __SYMM_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include "infiniop/ops/symm.h"

struct SymmInfo {
private:
    SymmInfo() = default;

public:
    size_t m;
    size_t n;
    infiniopBlasSideMode_t side;
    infiniopBlasFillMode_t uplo;
    ptrdiff_t A_row_stride;
    ptrdiff_t A_col_stride;
    ptrdiff_t B_row_stride;
    ptrdiff_t B_col_stride;
    ptrdiff_t C_row_stride;
    ptrdiff_t C_col_stride;
    infiniDtype_t data_type;

    static utils::Result<SymmInfo> createSymmInfo(
        infiniopBlasSideMode_t side,
        infiniopBlasFillMode_t uplo,
        infiniopTensorDescriptor_t alpha_desc,
        infiniopTensorDescriptor_t A_desc,
        infiniopTensorDescriptor_t B_desc,
        infiniopTensorDescriptor_t beta_desc,
        infiniopTensorDescriptor_t C_desc) {

        CHECK_OR_RETURN(side == INFINIOP_BLAS_SIDE_LEFT || side == INFINIOP_BLAS_SIDE_RIGHT, INFINI_STATUS_BAD_PARAM);
        CHECK_OR_RETURN(uplo == INFINIOP_BLAS_FILL_MODE_UPPER || uplo == INFINIOP_BLAS_FILL_MODE_LOWER, INFINI_STATUS_BAD_PARAM);
        CHECK_OR_RETURN(alpha_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(A_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(B_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(beta_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(C_desc != nullptr, INFINI_STATUS_NULL_POINTER);

        CHECK_OR_RETURN(alpha_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(beta_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(A_desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(B_desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(C_desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto data_type = C_desc->dtype();
        CHECK_DTYPE(data_type, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
        CHECK_OR_RETURN(alpha_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(A_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(B_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(beta_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);

        const auto m = C_desc->dim(0);
        const auto n = C_desc->dim(1);
        CHECK_OR_RETURN(B_desc->dim(0) == m, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(B_desc->dim(1) == n, INFINI_STATUS_BAD_TENSOR_SHAPE);

        const auto a_dim = side == INFINIOP_BLAS_SIDE_LEFT ? m : n;
        CHECK_OR_RETURN(A_desc->dim(0) == a_dim, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(A_desc->dim(1) == a_dim, INFINI_STATUS_BAD_TENSOR_SHAPE);

        const auto A_row_stride = A_desc->stride(0);
        const auto A_col_stride = A_desc->stride(1);
        const auto B_row_stride = B_desc->stride(0);
        const auto B_col_stride = B_desc->stride(1);
        const auto C_row_stride = C_desc->stride(0);
        const auto C_col_stride = C_desc->stride(1);
        CHECK_OR_RETURN(A_row_stride == 1 || A_col_stride == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(B_row_stride == 1 || B_col_stride == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(C_row_stride == 1 || C_col_stride == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);

        return utils::Result<SymmInfo>(SymmInfo{
            m,
            n,
            side,
            uplo,
            A_row_stride,
            A_col_stride,
            B_row_stride,
            B_col_stride,
            C_row_stride,
            C_col_stride,
            data_type});
    }
};

#endif // __SYMM_INFO_H__
