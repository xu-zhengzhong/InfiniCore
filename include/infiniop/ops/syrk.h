#ifndef __INFINIOP_SYRK_API_H__
#define __INFINIOP_SYRK_API_H__

#include "../operator_descriptor.h"
#include "blas_enum.h"

typedef struct InfiniopDescriptor *infiniopSyrkDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSyrkDescriptor(infiniopHandle_t handle,
                                                                infiniopSyrkDescriptor_t *desc_ptr,
                                                                infiniopBlasFillMode_t uplo,
                                                                infiniopBlasOperation_t trans,
                                                                infiniopTensorDescriptor_t alpha_desc,
                                                                infiniopTensorDescriptor_t A_desc,
                                                                infiniopTensorDescriptor_t beta_desc,
                                                                infiniopTensorDescriptor_t C_desc);

__INFINI_C __export infiniStatus_t infiniopGetSyrkWorkspaceSize(infiniopSyrkDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopSyrk(infiniopSyrkDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                const void *alpha,
                                                const void *A,
                                                const void *beta,
                                                void *C,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySyrkDescriptor(infiniopSyrkDescriptor_t desc);

#endif
