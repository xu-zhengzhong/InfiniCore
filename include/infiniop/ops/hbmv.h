#ifndef __INFINIOP_HBMV_API_H__
#define __INFINIOP_HBMV_API_H__

#include "../operator_descriptor.h"
#include "blas_enum.h"

typedef struct InfiniopDescriptor *infiniopHbmvDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateHbmvDescriptor(infiniopHandle_t handle,
                                                                infiniopHbmvDescriptor_t *desc_ptr,
                                                                infiniopBlasFillMode_t uplo,
                                                                size_t k,
                                                                infiniopTensorDescriptor_t alpha_desc,
                                                                infiniopTensorDescriptor_t A_desc,
                                                                infiniopTensorDescriptor_t x_desc,
                                                                infiniopTensorDescriptor_t beta_desc,
                                                                infiniopTensorDescriptor_t y_desc);

__INFINI_C __export infiniStatus_t infiniopGetHbmvWorkspaceSize(infiniopHbmvDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopHbmv(infiniopHbmvDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                const void *alpha,
                                                const void *A,
                                                const void *x,
                                                const void *beta,
                                                void *y,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyHbmvDescriptor(infiniopHbmvDescriptor_t desc);

#endif
