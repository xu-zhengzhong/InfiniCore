#ifndef __SYR_H__
#define __SYR_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::syr::NAMESPACE {                               \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        SyrInfo _info;                                           \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            SyrInfo info,                                        \
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
            infiniopBlasFillMode_t uplo,                         \
            infiniopTensorDescriptor_t alpha_desc,               \
            infiniopTensorDescriptor_t x_desc,                   \
            infiniopTensorDescriptor_t A_desc);                  \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            const void *alpha,                                   \
            const void *x,                                       \
            void *A,                                             \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __SYR_H__
