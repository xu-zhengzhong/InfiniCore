#ifndef __TRMM_H__
#define __TRMM_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::trmm::NAMESPACE {                              \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        TrmmInfo _info;                                          \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            TrmmInfo info,                                       \
            size_t workspace_size_,                              \
            Opaque *opaque,                                      \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _info(std::move(info)),                            \
              _workspace_size(workspace_size_) {}                \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
                                                                 \
        size_t workspaceSize() const { return _workspace_size; } \
                                                                 \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopBlasSideMode_t side,                         \
            infiniopBlasFillMode_t uplo,                         \
            infiniopBlasOperation_t trans,                       \
            infiniopBlasDiagType_t diag,                         \
            infiniopTensorDescriptor_t alpha_desc,               \
            infiniopTensorDescriptor_t A_desc,                   \
            infiniopTensorDescriptor_t B_desc);                  \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            const void *alpha,                                   \
            const void *A,                                       \
            void *B,                                             \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __TRMM_H__
