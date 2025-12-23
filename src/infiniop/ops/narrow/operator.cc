#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/narrow.h"
#include <stdint.h>

// Backend implementation headers
#ifdef ENABLE_CPU_API
#include "cpu/narrow_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/narrow_nvidia.cuh"
#endif

#ifdef ENABLE_METAX_API
#include "metax/narrow_metax.h"
#endif
#ifdef ENABLE_MOORE_API
#include "moore/narrow_moore.h"
#endif

extern "C" {

// =======================================================================
// 1. Create operator descriptor
// =======================================================================
__C infiniStatus_t infiniopCreateNarrowDescriptor(
    infiniopHandle_t handle,
    infiniopNarrowDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t input,
    int64_t dim,
    int64_t start,
    int64_t length) {

    #define CREATE(CASE, NAMESPACE)                                               \
        case CASE:                                                                \
            return op::narrow::NAMESPACE::Descriptor::create(                     \
                handle,                                                           \
                reinterpret_cast<op::narrow::NAMESPACE::Descriptor **>(desc_ptr), \
                output,                                                           \
                input,                                                            \
                dim,                                                              \
                start,                                                            \
                length)

    switch (handle->device) {
    #ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
    #endif
    #ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
    #endif
    #ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
    #endif
    #ifdef ENABLE_QY_API
        CREATE(INFINI_DEVICE_QY, nvidia);
    #endif
    #ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
    #endif
    #ifdef ENABLE_MOORE_API
        CREATE(INFINI_DEVICE_MOORE, moore);
    #endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
    #undef CREATE
}

// =======================================================================
// 2. Get workspace size
// =======================================================================
__C infiniStatus_t infiniopGetNarrowWorkspaceSize(infiniopNarrowDescriptor_t desc, size_t *size) {

    #define GET(CASE, NAMESPACE)                                                                      \
        case CASE:                                                                                    \
            *size = reinterpret_cast<op::narrow::NAMESPACE::Descriptor *>(desc)->workspaceSize();    \
            return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
    #ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
    #endif
    #ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
    #endif
    #ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
    #endif
    #ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia);
    #endif
    #ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
    #endif
    #ifdef ENABLE_MOORE_API
        GET(INFINI_DEVICE_MOORE, moore);
    #endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
    #undef GET
}

// =======================================================================
// 3. Execute calculation
// =======================================================================
__C infiniStatus_t infiniopNarrow(
    infiniopNarrowDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) {

    #define CALCULATE(CASE, NAMESPACE)                                            \
        case CASE:                                                                \
            return reinterpret_cast<const op::narrow::NAMESPACE::Descriptor *>(desc) \
                ->calculate(workspace, workspace_size, output, input, stream)

    switch (desc->device_type) {
    #ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
    #endif
    #ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
    #endif
    #ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
    #endif
    #ifdef ENABLE_QY_API
        CALCULATE(INFINI_DEVICE_QY, nvidia);
    #endif
    #ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
    #endif
    #ifdef ENABLE_MOORE_API
        CALCULATE(INFINI_DEVICE_MOORE, moore);
    #endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
    #undef CALCULATE
}

// =======================================================================
// 4. Destroy descriptor
// =======================================================================
__C infiniStatus_t infiniopDestroyNarrowDescriptor(infiniopNarrowDescriptor_t desc) {

    #define DELETE(CASE, NAMESPACE)                                                            \
        case CASE:                                                                             \
            delete reinterpret_cast<const op::narrow::NAMESPACE::Descriptor *>(desc);          \
            return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
    #ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
    #endif
    #ifdef ENABLE_NVIDIA_API
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
    #endif
    #ifdef ENABLE_ILUVATAR_API
        DELETE(INFINI_DEVICE_ILUVATAR, nvidia);
    #endif
    #ifdef ENABLE_QY_API
        DELETE(INFINI_DEVICE_QY, nvidia);
    #endif
    #ifdef ENABLE_METAX_API
        DELETE(INFINI_DEVICE_METAX, metax);
    #endif
    #ifdef ENABLE_MOORE_API
        DELETE(INFINI_DEVICE_MOORE, moore);
    #endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
    #undef DELETE
}

} // extern "C"
