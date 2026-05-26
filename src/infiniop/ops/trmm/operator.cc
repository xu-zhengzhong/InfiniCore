#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/trmm.h"

#ifdef ENABLE_CPU_API
#include "cpu/trmm_cpu.h"
#endif
#ifdef ENABLE_METAX_API
#include "metax/trmm_metax.h"
#endif
#ifdef ENABLE_CAMBRICON_API
#include "bang/trmm_bang.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateTrmmDescriptor(
    infiniopHandle_t handle,
    infiniopTrmmDescriptor_t *desc_ptr,
    infiniopBlasSideMode_t side,
    infiniopBlasFillMode_t uplo,
    infiniopBlasOperation_t trans,
    infiniopBlasDiagType_t diag,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t A_desc,
    infiniopTensorDescriptor_t B_desc) {

#define CREATE(CASE, NAMESPACE)                                             \
    case CASE:                                                              \
        return op::trmm::NAMESPACE::Descriptor::create(                     \
            handle,                                                         \
            reinterpret_cast<op::trmm::NAMESPACE::Descriptor **>(desc_ptr), \
            side,                                                           \
            uplo,                                                           \
            trans,                                                          \
            diag,                                                           \
            alpha_desc,                                                     \
            A_desc,                                                         \
            B_desc)

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

__INFINI_C infiniStatus_t infiniopGetTrmmWorkspaceSize(infiniopTrmmDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                      \
    case CASE:                                                                                    \
        *size = reinterpret_cast<const op::trmm::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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

__INFINI_C infiniStatus_t infiniopTrmm(
    infiniopTrmmDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    const void *alpha,
    const void *A,
    void *B,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                             \
    case CASE:                                                                 \
        return reinterpret_cast<const op::trmm::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, alpha, A, B, stream)

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

__INFINI_C infiniStatus_t infiniopDestroyTrmmDescriptor(infiniopTrmmDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                 \
    case CASE:                                                                  \
        delete reinterpret_cast<const op::trmm::NAMESPACE::Descriptor *>(desc); \
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
