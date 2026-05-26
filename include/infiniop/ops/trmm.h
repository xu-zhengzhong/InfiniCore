#ifndef __INFINIOP_TRMM_API_H__
#define __INFINIOP_TRMM_API_H__

#include "../operator_descriptor.h"
#include "blas_enum.h"

typedef struct InfiniopDescriptor *infiniopTrmmDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateTrmmDescriptor(infiniopHandle_t handle,
                                                                infiniopTrmmDescriptor_t *desc_ptr,
                                                                infiniopBlasSideMode_t side,
                                                                infiniopBlasFillMode_t uplo,
                                                                infiniopBlasOperation_t trans,
                                                                infiniopBlasDiagType_t diag,
                                                                infiniopTensorDescriptor_t alpha_desc,
                                                                infiniopTensorDescriptor_t A_desc,
                                                                infiniopTensorDescriptor_t B_desc);

__INFINI_C __export infiniStatus_t infiniopGetTrmmWorkspaceSize(infiniopTrmmDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopTrmm(infiniopTrmmDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                const void *alpha,
                                                const void *A,
                                                void *B,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyTrmmDescriptor(infiniopTrmmDescriptor_t desc);

#endif
