#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/syr2k.h"

#ifdef ENABLE_CPU_API
#include "cpu/syr2k_cpu.h"
#endif
#ifdef ENABLE_METAX_API
#include "metax/syr2k_metax.h"
#endif
#ifdef ENABLE_CAMBRICON_API
#include "bang/syr2k_bang.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateSyr2kDescriptor(
    infiniopHandle_t handle,
    infiniopSyr2kDescriptor_t *desc_ptr,
    infiniopBlasFillMode_t uplo,
    infiniopBlasOperation_t trans,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t A_desc,
    infiniopTensorDescriptor_t B_desc,
    infiniopTensorDescriptor_t beta_desc,
    infiniopTensorDescriptor_t C_desc) {

#define CREATE(CASE, NAMESPACE)                                              \
    case CASE:                                                               \
        return op::syr2k::NAMESPACE::Descriptor::create(                     \
            handle,                                                          \
            reinterpret_cast<op::syr2k::NAMESPACE::Descriptor **>(desc_ptr), \
            uplo,                                                            \
            trans,                                                           \
            alpha_desc,                                                      \
            A_desc,                                                          \
            B_desc,                                                          \
            beta_desc,                                                       \
            C_desc)

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_CAMBRICON_API
        CREATE(INFINI_DEVICE_CAMBRICON, bang);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetSyr2kWorkspaceSize(infiniopSyr2kDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                       \
    case CASE:                                                                                     \
        *size = reinterpret_cast<const op::syr2k::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_CAMBRICON_API
        GET(INFINI_DEVICE_CAMBRICON, bang);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GET
}

__INFINI_C infiniStatus_t infiniopSyr2k(
    infiniopSyr2kDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    const void *alpha,
    const void *A,
    const void *B,
    const void *beta,
    void *C,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                              \
    case CASE:                                                                  \
        return reinterpret_cast<const op::syr2k::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, alpha, A, B, beta, C, stream)

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_CAMBRICON_API
        CALCULATE(INFINI_DEVICE_CAMBRICON, bang);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroySyr2kDescriptor(infiniopSyr2kDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                  \
    case CASE:                                                                   \
        delete reinterpret_cast<const op::syr2k::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_METAX_API
        DELETE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_CAMBRICON_API
        DELETE(INFINI_DEVICE_CAMBRICON, bang);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
