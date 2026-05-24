#ifndef __INFINIOP_SYMM_API_H__
#define __INFINIOP_SYMM_API_H__

#include "../operator_descriptor.h"
#include "blas_enum.h"

typedef struct InfiniopDescriptor *infiniopSymmDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSymmDescriptor(infiniopHandle_t handle,
                                                                infiniopSymmDescriptor_t *desc_ptr,
                                                                infiniopBlasSideMode_t side,
                                                                infiniopBlasFillMode_t uplo,
                                                                infiniopTensorDescriptor_t alpha_desc,
                                                                infiniopTensorDescriptor_t A_desc,
                                                                infiniopTensorDescriptor_t B_desc,
                                                                infiniopTensorDescriptor_t beta_desc,
                                                                infiniopTensorDescriptor_t C_desc);

__INFINI_C __export infiniStatus_t infiniopGetSymmWorkspaceSize(infiniopSymmDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopSymm(infiniopSymmDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                const void *alpha,
                                                const void *A,
                                                const void *B,
                                                const void *beta,
                                                void *C,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySymmDescriptor(infiniopSymmDescriptor_t desc);

#endif
