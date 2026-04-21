#ifndef __ROTG_H__
#define __ROTG_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "infiniop/ops/rotg.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::rotg::NAMESPACE {                              \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        RotgInfo _info;                                          \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            RotgInfo info,                                       \
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
            infiniopTensorDescriptor_t a_desc,                   \
            infiniopTensorDescriptor_t b_desc,                   \
            infiniopTensorDescriptor_t c_desc,                   \
            infiniopTensorDescriptor_t s_desc);                  \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *a,                                             \
            void *b,                                             \
            void *c,                                             \
            void *s,                                             \
            void *stream) const;                                 \
    };                                                           \
    }

class RotgInfo {
private:
    infiniDtype_t _dtype;

public:
    RotgInfo() = default;
    explicit RotgInfo(infiniDtype_t dtype) : _dtype(dtype) {}

    inline infiniDtype_t getDtype() const { return _dtype; }

    static utils::Result<RotgInfo> createRotgInfo(
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc,
        infiniopTensorDescriptor_t c_desc,
        infiniopTensorDescriptor_t s_desc) {

        CHECK_OR_RETURN(a_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(b_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(c_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(s_desc != nullptr, INFINI_STATUS_NULL_POINTER);

        CHECK_OR_RETURN(a_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(b_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(c_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(s_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);

        CHECK_OR_RETURN(a_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(b_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(c_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(s_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);

        CHECK_OR_RETURN(a_desc->dtype() == b_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(a_desc->dtype() == c_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(a_desc->dtype() == s_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);

        return utils::Result<RotgInfo>(RotgInfo(a_desc->dtype()));
    }
};

#endif // __ROTG_H__
