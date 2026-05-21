#ifndef __INFINIOP_HEMV_API_H__
#define __INFINIOP_HEMV_API_H__

#include "../operator_descriptor.h"
#include "blas_enum.h"

typedef struct InfiniopDescriptor *infiniopHemvDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateHemvDescriptor(infiniopHandle_t handle,
                                                                infiniopHemvDescriptor_t *desc_ptr,
                                                                infiniopBlasFillMode_t uplo,
                                                                infiniopTensorDescriptor_t alpha_desc,
                                                                infiniopTensorDescriptor_t A_desc,
                                                                infiniopTensorDescriptor_t x_desc,
                                                                infiniopTensorDescriptor_t beta_desc,
                                                                infiniopTensorDescriptor_t y_desc);

__INFINI_C __export infiniStatus_t infiniopGetHemvWorkspaceSize(infiniopHemvDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopHemv(infiniopHemvDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                const void *alpha,
                                                const void *A,
                                                const void *x,
                                                const void *beta,
                                                void *y,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyHemvDescriptor(infiniopHemvDescriptor_t desc);

#endif
