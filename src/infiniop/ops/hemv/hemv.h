#ifndef __HEMV_H__
#define __HEMV_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::hemv::NAMESPACE {                              \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        HemvInfo _info;                                          \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            HemvInfo info,                                       \
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
            infiniopTensorDescriptor_t A_desc,                   \
            infiniopTensorDescriptor_t x_desc,                   \
            infiniopTensorDescriptor_t beta_desc,                \
            infiniopTensorDescriptor_t y_desc);                  \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            const void *alpha,                                   \
            const void *A,                                       \
            const void *x,                                       \
            const void *beta,                                    \
            void *y,                                             \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __HEMV_H__
