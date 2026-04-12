#ifndef __AXPY_H__
#define __AXPY_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "infiniop/ops/axpy.h"
#include <vector>
#include <cstring>

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::axpy::NAMESPACE {                              \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        AxpyInfo _info;                                          \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            AxpyInfo info,                                       \
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
            const void *alpha,                                   \
            const void *x,                                       \
            void *y,                                             \
            void *stream) const;                                 \
    };                                                           \
    }

class AxpyInfo {
private:
    size_t _size;
    size_t _incx;
    size_t _incy;
    infiniDtype_t _dtype;

public:
    AxpyInfo() = default;

    AxpyInfo(size_t size,
             size_t incx,
             size_t incy,
             infiniDtype_t dtype)
        : _size(size), _incx(incx), _incy(incy), _dtype(dtype) {}

    inline size_t getSize() const { return _size; }
    inline size_t getIncx() const { return _incx; }
    inline size_t getIncy() const { return _incy; }
    inline infiniDtype_t getDtype() const { return _dtype; }

    static utils::Result<AxpyInfo> createAxpyInfo(
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t y_desc) {

        CHECK_OR_RETURN(x_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(x_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(y_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(y_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(x_desc->dtype() == y_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(x_desc->numel() == y_desc->numel(), INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto size = x_desc->numel();
        auto dtype = x_desc->dtype();
        auto incx = x_desc->stride(0);
        auto incy = y_desc->stride(0);

        return utils::Result<AxpyInfo>(AxpyInfo(size, incx, incy, dtype));
    }
};

#endif // __AXPY_H__