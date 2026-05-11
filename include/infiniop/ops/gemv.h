#ifndef __INFINIOP_GEMV_API_H__
#define __INFINIOP_GEMV_API_H__

#include "../operator_descriptor.h"
#include "blas_enum.h"

typedef struct InfiniopDescriptor *infiniopGemvDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateGemvDescriptor(infiniopHandle_t handle,
                                                                infiniopGemvDescriptor_t *desc_ptr,
                                                                infiniopBlasOperation_t trans,
                                                                infiniopTensorDescriptor_t alpha_desc,
                                                                infiniopTensorDescriptor_t A_desc,
                                                                infiniopTensorDescriptor_t x_desc,
                                                                infiniopTensorDescriptor_t beta_desc,
                                                                infiniopTensorDescriptor_t y_desc);

__INFINI_C __export infiniStatus_t infiniopGetGemvWorkspaceSize(infiniopGemvDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopGemv(infiniopGemvDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                const void *alpha,
                                                const void *A,
                                                const void *x,
                                                const void *beta,
                                                void *y,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyGemvDescriptor(infiniopGemvDescriptor_t desc);

#endif
