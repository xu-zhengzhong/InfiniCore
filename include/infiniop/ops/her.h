#ifndef __INFINIOP_HER_API_H__
#define __INFINIOP_HER_API_H__

#include "../operator_descriptor.h"
#include "blas_enum.h"

typedef struct InfiniopDescriptor *infiniopHerDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateHerDescriptor(infiniopHandle_t handle,
                                                               infiniopHerDescriptor_t *desc_ptr,
                                                               infiniopBlasFillMode_t uplo,
                                                               infiniopTensorDescriptor_t alpha_desc,
                                                               infiniopTensorDescriptor_t x_desc,
                                                               infiniopTensorDescriptor_t A_desc);

__INFINI_C __export infiniStatus_t infiniopGetHerWorkspaceSize(infiniopHerDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopHer(infiniopHerDescriptor_t desc,
                                               void *workspace,
                                               size_t workspace_size,
                                               const void *alpha,
                                               const void *x,
                                               void *A,
                                               void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyHerDescriptor(infiniopHerDescriptor_t desc);

#endif
