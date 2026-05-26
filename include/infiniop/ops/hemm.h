#ifndef __INFINIOP_HEMM_API_H__
#define __INFINIOP_HEMM_API_H__

#include "../operator_descriptor.h"
#include "blas_enum.h"

typedef struct InfiniopDescriptor *infiniopHemmDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateHemmDescriptor(infiniopHandle_t handle,
                                                                infiniopHemmDescriptor_t *desc_ptr,
                                                                infiniopBlasSideMode_t side,
                                                                infiniopBlasFillMode_t uplo,
                                                                infiniopTensorDescriptor_t alpha_desc,
                                                                infiniopTensorDescriptor_t A_desc,
                                                                infiniopTensorDescriptor_t B_desc,
                                                                infiniopTensorDescriptor_t beta_desc,
                                                                infiniopTensorDescriptor_t C_desc);

__INFINI_C __export infiniStatus_t infiniopGetHemmWorkspaceSize(infiniopHemmDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopHemm(infiniopHemmDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                const void *alpha,
                                                const void *A,
                                                const void *B,
                                                const void *beta,
                                                void *C,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyHemmDescriptor(infiniopHemmDescriptor_t desc);

#endif
