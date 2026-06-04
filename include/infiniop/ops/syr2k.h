#ifndef __INFINIOP_SYR2K_API_H__
#define __INFINIOP_SYR2K_API_H__

#include "../operator_descriptor.h"
#include "blas_enum.h"

typedef struct InfiniopDescriptor *infiniopSyr2kDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSyr2kDescriptor(infiniopHandle_t handle,
                                                                 infiniopSyr2kDescriptor_t *desc_ptr,
                                                                 infiniopBlasFillMode_t uplo,
                                                                 infiniopBlasOperation_t trans,
                                                                 infiniopTensorDescriptor_t alpha_desc,
                                                                 infiniopTensorDescriptor_t A_desc,
                                                                 infiniopTensorDescriptor_t B_desc,
                                                                 infiniopTensorDescriptor_t beta_desc,
                                                                 infiniopTensorDescriptor_t C_desc);

__INFINI_C __export infiniStatus_t infiniopGetSyr2kWorkspaceSize(infiniopSyr2kDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopSyr2k(infiniopSyr2kDescriptor_t desc,
                                                 void *workspace,
                                                 size_t workspace_size,
                                                 const void *alpha,
                                                 const void *A,
                                                 const void *B,
                                                 const void *beta,
                                                 void *C,
                                                 void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySyr2kDescriptor(infiniopSyr2kDescriptor_t desc);

#endif
