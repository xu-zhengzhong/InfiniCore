#ifndef __ASUM_H__
#define __ASUM_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "infiniop/ops/asum.h"
#include <cstring>
#include <vector>

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::asum::NAMESPACE {                              \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        AsumInfo _info;                                          \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            AsumInfo info,                                       \
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
            infiniopTensorDescriptor_t x_desc);                  \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            const void *x,                                       \
            void *result,                                        \
            void *stream) const;                                 \
    };                                                           \
    }

class AsumInfo {
private:
    size_t _size;
    size_t _incx;
    infiniDtype_t _dtype;

public:
    AsumInfo() = default;

    AsumInfo(size_t size,
             size_t incx,
             infiniDtype_t dtype)
        : _size(size), _incx(incx), _dtype(dtype) {}

    inline size_t getSize() const { return _size; }
    inline size_t getIncx() const { return _incx; }
    inline infiniDtype_t getDtype() const { return _dtype; }

    static utils::Result<AsumInfo> createAsumInfo(
        infiniopTensorDescriptor_t x_desc) {
        CHECK_OR_RETURN(x_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(x_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto size = x_desc->numel();
        auto dtype = x_desc->dtype();
        auto incx = x_desc->stride(0);

        return utils::Result<AsumInfo>(AsumInfo(size, incx, dtype));
    }
};

#endif // __ASUM_H__