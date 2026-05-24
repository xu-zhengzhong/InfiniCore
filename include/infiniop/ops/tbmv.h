#ifndef __INFINIOP_TBMV_API_H__
#define __INFINIOP_TBMV_API_H__

#include "../operator_descriptor.h"
#include "blas_enum.h"

typedef struct InfiniopDescriptor *infiniopTbmvDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateTbmvDescriptor(infiniopHandle_t handle,
                                                                infiniopTbmvDescriptor_t *desc_ptr,
                                                                infiniopBlasFillMode_t uplo,
                                                                infiniopBlasOperation_t trans,
                                                                infiniopBlasDiagType_t diag,
                                                                size_t k,
                                                                infiniopTensorDescriptor_t A_desc,
                                                                infiniopTensorDescriptor_t x_desc);

__INFINI_C __export infiniStatus_t infiniopGetTbmvWorkspaceSize(infiniopTbmvDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopTbmv(infiniopTbmvDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                const void *A,
                                                void *x,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyTbmvDescriptor(infiniopTbmvDescriptor_t desc);

#endif
