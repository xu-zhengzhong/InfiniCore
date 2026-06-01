#ifndef __INFINIOP_TBSV_API_H__
#define __INFINIOP_TBSV_API_H__

#include "../operator_descriptor.h"
#include "blas_enum.h"

typedef struct InfiniopDescriptor *infiniopTbsvDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateTbsvDescriptor(infiniopHandle_t handle,
                                                                infiniopTbsvDescriptor_t *desc_ptr,
                                                                infiniopBlasFillMode_t uplo,
                                                                infiniopBlasOperation_t trans,
                                                                infiniopBlasDiagType_t diag,
                                                                size_t k,
                                                                infiniopTensorDescriptor_t A_desc,
                                                                infiniopTensorDescriptor_t x_desc);

__INFINI_C __export infiniStatus_t infiniopGetTbsvWorkspaceSize(infiniopTbsvDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopTbsv(infiniopTbsvDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                const void *A,
                                                void *x,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyTbsvDescriptor(infiniopTbsvDescriptor_t desc);

#endif
