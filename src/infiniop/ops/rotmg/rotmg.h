#ifndef __ROTMG_H__
#define __ROTMG_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "infiniop/ops/rotmg.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::rotmg::NAMESPACE {                             \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        RotmgInfo _info;                                         \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            RotmgInfo info,                                      \
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
            infiniopTensorDescriptor_t d1_desc,                  \
            infiniopTensorDescriptor_t d2_desc,                  \
            infiniopTensorDescriptor_t x1_desc,                  \
            infiniopTensorDescriptor_t y1_desc,                  \
            infiniopTensorDescriptor_t param_desc);              \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *d1,                                            \
            void *d2,                                            \
            void *x1,                                            \
            const void *y1,                                      \
            void *param,                                         \
            void *stream) const;                                 \
    };                                                           \
    }

class RotmgInfo {
private:
    infiniDtype_t _dtype;

public:
    RotmgInfo() = default;
    explicit RotmgInfo(infiniDtype_t dtype) : _dtype(dtype) {}

    inline infiniDtype_t getDtype() const { return _dtype; }

    static utils::Result<RotmgInfo> createRotmgInfo(
        infiniopTensorDescriptor_t d1_desc,
        infiniopTensorDescriptor_t d2_desc,
        infiniopTensorDescriptor_t x1_desc,
        infiniopTensorDescriptor_t y1_desc,
        infiniopTensorDescriptor_t param_desc) {

        CHECK_OR_RETURN(d1_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(d2_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(x1_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(y1_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(param_desc != nullptr, INFINI_STATUS_NULL_POINTER);

        CHECK_OR_RETURN(d1_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(d2_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(x1_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(y1_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(param_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);

        CHECK_OR_RETURN(d1_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(d2_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(x1_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(y1_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(param_desc->numel() == 5, INFINI_STATUS_BAD_TENSOR_SHAPE);

        CHECK_OR_RETURN(d1_desc->dtype() == d2_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(d1_desc->dtype() == x1_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(d1_desc->dtype() == y1_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(d1_desc->dtype() == param_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);

        return utils::Result<RotmgInfo>(RotmgInfo(d1_desc->dtype()));
    }
};

#endif // __ROTMG_H__
