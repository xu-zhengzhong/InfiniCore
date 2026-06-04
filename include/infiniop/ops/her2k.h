#ifndef __INFINIOP_HER2K_API_H__
#define __INFINIOP_HER2K_API_H__

#include "../operator_descriptor.h"
#include "blas_enum.h"

typedef struct InfiniopDescriptor *infiniopHer2kDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateHer2kDescriptor(infiniopHandle_t handle,
                                                                 infiniopHer2kDescriptor_t *desc_ptr,
                                                                 infiniopBlasFillMode_t uplo,
                                                                 infiniopBlasOperation_t trans,
                                                                 infiniopTensorDescriptor_t alpha_desc,
                                                                 infiniopTensorDescriptor_t A_desc,
                                                                 infiniopTensorDescriptor_t B_desc,
                                                                 infiniopTensorDescriptor_t beta_desc,
                                                                 infiniopTensorDescriptor_t C_desc);

__INFINI_C __export infiniStatus_t infiniopGetHer2kWorkspaceSize(infiniopHer2kDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopHer2k(infiniopHer2kDescriptor_t desc,
                                                 void *workspace,
                                                 size_t workspace_size,
                                                 const void *alpha,
                                                 const void *A,
                                                 const void *B,
                                                 const void *beta,
                                                 void *C,
                                                 void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyHer2kDescriptor(infiniopHer2kDescriptor_t desc);

#endif
