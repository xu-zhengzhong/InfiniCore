#ifndef __SYMM_H__
#define __SYMM_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::symm::NAMESPACE {                              \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        SymmInfo _info;                                          \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            SymmInfo info,                                       \
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
            infiniopTensorDescriptor_t alpha_desc,               \
            infiniopTensorDescriptor_t A_desc,                   \
            infiniopTensorDescriptor_t B_desc,                   \
            infiniopTensorDescriptor_t beta_desc,                \
            infiniopTensorDescriptor_t C_desc);                  \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            const void *alpha,                                   \
            const void *A,                                       \
            const void *B,                                       \
            const void *beta,                                    \
            void *C,                                             \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __SYMM_H__
