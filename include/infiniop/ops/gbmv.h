#ifndef __INFINIOP_GBMV_API_H__
#define __INFINIOP_GBMV_API_H__

#include "../operator_descriptor.h"
#include "blas_enum.h"

typedef struct InfiniopDescriptor *infiniopGbmvDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateGbmvDescriptor(infiniopHandle_t handle,
                                                                infiniopGbmvDescriptor_t *desc_ptr,
                                                                infiniopBlasOperation_t trans,
                                                                size_t kl,
                                                                size_t ku,
                                                                infiniopTensorDescriptor_t alpha_desc,
                                                                infiniopTensorDescriptor_t A_desc,
                                                                infiniopTensorDescriptor_t x_desc,
                                                                infiniopTensorDescriptor_t beta_desc,
                                                                infiniopTensorDescriptor_t y_desc);

__INFINI_C __export infiniStatus_t infiniopGetGbmvWorkspaceSize(infiniopGbmvDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopGbmv(infiniopGbmvDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                const void *alpha,
                                                const void *A,
                                                const void *x,
                                                const void *beta,
                                                void *y,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyGbmvDescriptor(infiniopGbmvDescriptor_t desc);

#endif
