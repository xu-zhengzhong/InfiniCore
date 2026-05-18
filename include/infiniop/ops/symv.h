#ifndef __INFINIOP_SYMV_API_H__
#define __INFINIOP_SYMV_API_H__

#include "../operator_descriptor.h"
#include "blas_enum.h"

typedef struct InfiniopDescriptor *infiniopSymvDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSymvDescriptor(infiniopHandle_t handle,
                                                                infiniopSymvDescriptor_t *desc_ptr,
                                                                infiniopBlasFillMode_t uplo,
                                                                infiniopTensorDescriptor_t alpha_desc,
                                                                infiniopTensorDescriptor_t A_desc,
                                                                infiniopTensorDescriptor_t x_desc,
                                                                infiniopTensorDescriptor_t beta_desc,
                                                                infiniopTensorDescriptor_t y_desc);

__INFINI_C __export infiniStatus_t infiniopGetSymvWorkspaceSize(infiniopSymvDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopSymv(infiniopSymvDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                const void *alpha,
                                                const void *A,
                                                const void *x,
                                                const void *beta,
                                                void *y,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySymvDescriptor(infiniopSymvDescriptor_t desc);

#endif
