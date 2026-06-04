#ifndef __INFINIOP_TRSM_API_H__
#define __INFINIOP_TRSM_API_H__

#include "../operator_descriptor.h"
#include "blas_enum.h"

typedef struct InfiniopDescriptor *infiniopTrsmDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateTrsmDescriptor(infiniopHandle_t handle,
                                                                infiniopTrsmDescriptor_t *desc_ptr,
                                                                infiniopBlasSideMode_t side,
                                                                infiniopBlasFillMode_t uplo,
                                                                infiniopBlasOperation_t trans,
                                                                infiniopBlasDiagType_t diag,
                                                                infiniopTensorDescriptor_t alpha_desc,
                                                                infiniopTensorDescriptor_t A_desc,
                                                                infiniopTensorDescriptor_t B_desc);

__INFINI_C __export infiniStatus_t infiniopGetTrsmWorkspaceSize(infiniopTrsmDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopTrsm(infiniopTrsmDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                const void *alpha,
                                                const void *A,
                                                void *B,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyTrsmDescriptor(infiniopTrsmDescriptor_t desc);

#endif
