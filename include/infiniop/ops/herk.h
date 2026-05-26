#ifndef __INFINIOP_HERK_API_H__
#define __INFINIOP_HERK_API_H__

#include "../operator_descriptor.h"
#include "blas_enum.h"

typedef struct InfiniopDescriptor *infiniopHerkDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateHerkDescriptor(infiniopHandle_t handle,
                                                                infiniopHerkDescriptor_t *desc_ptr,
                                                                infiniopBlasFillMode_t uplo,
                                                                infiniopBlasOperation_t trans,
                                                                infiniopTensorDescriptor_t alpha_desc,
                                                                infiniopTensorDescriptor_t A_desc,
                                                                infiniopTensorDescriptor_t beta_desc,
                                                                infiniopTensorDescriptor_t C_desc);

__INFINI_C __export infiniStatus_t infiniopGetHerkWorkspaceSize(infiniopHerkDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopHerk(infiniopHerkDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                const void *alpha,
                                                const void *A,
                                                const void *beta,
                                                void *C,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyHerkDescriptor(infiniopHerkDescriptor_t desc);

#endif
