#ifndef __DOT_H__
#define __DOT_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "infiniop/ops/dot.h"
#include <cstring>
#include <vector>

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::dot::NAMESPACE {                               \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        DotInfo _info;                                           \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            DotInfo info,                                        \
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
            infiniopTensorDescriptor_t x_desc,                   \
            infiniopTensorDescriptor_t y_desc);                  \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            const void *x,                                       \
            const void *y,                                       \
            void *result,                                        \
            void *stream) const;                                 \
    };                                                           \
    }

class DotInfo {
private:
    size_t _size;
    ptrdiff_t _incx;
    ptrdiff_t _incy;
    infiniDtype_t _dtype;

public:
    DotInfo() = default;

    DotInfo(size_t size,
            ptrdiff_t incx,
            ptrdiff_t incy,
            infiniDtype_t dtype)
        : _size(size), _incx(incx), _incy(incy), _dtype(dtype) {}

    inline size_t getSize() const { return _size; }
    inline ptrdiff_t getIncx() const { return _incx; }
    inline ptrdiff_t getIncy() const { return _incy; }
    inline infiniDtype_t getDtype() const { return _dtype; }

    static utils::Result<DotInfo> createDotInfo(
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t y_desc) {
        CHECK_OR_RETURN(x_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(y_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(x_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(y_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(x_desc->dtype() == y_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(x_desc->numel() == y_desc->numel(), INFINI_STATUS_BAD_TENSOR_SHAPE);

        return utils::Result<DotInfo>(DotInfo(
            x_desc->numel(),
            x_desc->stride(0),
            y_desc->stride(0),
            x_desc->dtype()));
    }
};

#endif // __DOT_H__