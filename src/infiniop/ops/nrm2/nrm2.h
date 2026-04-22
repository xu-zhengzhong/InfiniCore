#ifndef __NRM2_H__
#define __NRM2_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "infiniop/ops/nrm2.h"
#include <cstring>
#include <vector>

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::nrm2::NAMESPACE {                              \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        Nrm2Info _info;                                          \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            Nrm2Info info,                                       \
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

class Nrm2Info {
private:
    size_t _size;
    size_t _incx;
    infiniDtype_t _dtype;

public:
    Nrm2Info() = default;

    Nrm2Info(size_t size,
             size_t incx,
             infiniDtype_t dtype)
        : _size(size), _incx(incx), _dtype(dtype) {}

    inline size_t getSize() const { return _size; }
    inline size_t getIncx() const { return _incx; }
    inline infiniDtype_t getDtype() const { return _dtype; }

    static utils::Result<Nrm2Info> createNrm2Info(
        infiniopTensorDescriptor_t x_desc) {
        CHECK_OR_RETURN(x_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(x_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto size = x_desc->numel();
        auto dtype = x_desc->dtype();
        auto incx = x_desc->stride(0);

        return utils::Result<Nrm2Info>(Nrm2Info(size, incx, dtype));
    }
};

#endif // __NRM2_H__