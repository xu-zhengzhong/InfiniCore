#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/tbmv.h"

#ifdef ENABLE_CPU_API
#include "cpu/tbmv_cpu.h"
#endif
#ifdef ENABLE_METAX_API
#include "metax/tbmv_metax.h"
#endif
#ifdef ENABLE_CAMBRICON_API
#include "bang/tbmv_bang.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateTbmvDescriptor(
    infiniopHandle_t handle,
    infiniopTbmvDescriptor_t *desc_ptr,
    infiniopBlasFillMode_t uplo,
    infiniopBlasOperation_t trans,
    infiniopBlasDiagType_t diag,
    size_t k,
    infiniopTensorDescriptor_t A_desc,
    infiniopTensorDescriptor_t x_desc) {

#define CREATE(CASE, NAMESPACE)                                             \
    case CASE:                                                              \
        return op::tbmv::NAMESPACE::Descriptor::create(                     \
            handle,                                                         \
            reinterpret_cast<op::tbmv::NAMESPACE::Descriptor **>(desc_ptr), \
            uplo,                                                           \
            trans,                                                          \
            diag,                                                           \
            k,                                                              \
            A_desc,                                                         \
            x_desc)

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

__INFINI_C infiniStatus_t infiniopGetTbmvWorkspaceSize(infiniopTbmvDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                      \
    case CASE:                                                                                    \
        *size = reinterpret_cast<const op::tbmv::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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

__INFINI_C infiniStatus_t infiniopTbmv(
    infiniopTbmvDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    const void *A,
    void *x,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                             \
    case CASE:                                                                 \
        return reinterpret_cast<const op::tbmv::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, A, x, stream)

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

__INFINI_C infiniStatus_t infiniopDestroyTbmvDescriptor(infiniopTbmvDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                 \
    case CASE:                                                                  \
        delete reinterpret_cast<const op::tbmv::NAMESPACE::Descriptor *>(desc); \
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
