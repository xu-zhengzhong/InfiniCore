#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/syr.h"

#ifdef ENABLE_CPU_API
#include "cpu/syr_cpu.h"
#endif
#ifdef ENABLE_METAX_API
#include "metax/syr_metax.h"
#endif
#ifdef ENABLE_CAMBRICON_API
#include "bang/syr_bang.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateSyrDescriptor(
    infiniopHandle_t handle,
    infiniopSyrDescriptor_t *desc_ptr,
    infiniopBlasFillMode_t uplo,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t A_desc) {

#define CREATE(CASE, NAMESPACE)                                            \
    case CASE:                                                             \
        return op::syr::NAMESPACE::Descriptor::create(                     \
            handle,                                                        \
            reinterpret_cast<op::syr::NAMESPACE::Descriptor **>(desc_ptr), \
            uplo,                                                          \
            alpha_desc,                                                    \
            x_desc,                                                        \
            A_desc)

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

__INFINI_C infiniStatus_t infiniopGetSyrWorkspaceSize(infiniopSyrDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                     \
    case CASE:                                                                                   \
        *size = reinterpret_cast<const op::syr::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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

__INFINI_C infiniStatus_t infiniopSyr(
    infiniopSyrDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    const void *alpha,
    const void *x,
    void *A,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                            \
    case CASE:                                                                \
        return reinterpret_cast<const op::syr::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, alpha, x, A, stream)

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

__INFINI_C infiniStatus_t infiniopDestroySyrDescriptor(infiniopSyrDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                \
    case CASE:                                                                 \
        delete reinterpret_cast<const op::syr::NAMESPACE::Descriptor *>(desc); \
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
