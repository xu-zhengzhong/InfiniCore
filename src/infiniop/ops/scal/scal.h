#ifndef __SCAL_H__
#define __SCAL_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "infiniop/ops/scal.h"
#include <vector>
#include <cstring>

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::scal::NAMESPACE {                              \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        ScalInfo _info;                                          \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            ScalInfo info,                                       \
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
            void *x,                                             \
            float alpha,                                         \
            void *stream) const;                                 \
    };                                                           \
    }

class ScalInfo {
private:
    std::vector<size_t> _meta;
    size_t _size;
    size_t _ndim;
    bool _contiguous;
    infiniDtype_t _dtype;

public:
    ScalInfo() = default;

    ScalInfo(std::vector<size_t> meta,
             size_t size,
             size_t ndim,
             bool contiguous,
             infiniDtype_t dtype)
        : _meta(std::move(meta)), _size(size),
          _ndim(ndim), _contiguous(contiguous), _dtype(dtype) {}

    inline size_t getSize() const { return _size; }
    inline size_t getNdim() const { return _ndim; }
    inline bool isContiguous() const { return _contiguous; }
    inline infiniDtype_t getDtype() const { return _dtype; }
    inline const size_t *getShape() const { return reinterpret_cast<const size_t *>(_meta.data()); }
    inline const ptrdiff_t *getStrides() const { return reinterpret_cast<const ptrdiff_t *>(getShape() + _ndim); }

    static utils::Result<ScalInfo> createScalInfo(infiniopTensorDescriptor_t x_desc) {
        CHECK_OR_RETURN(x_desc != nullptr, INFINI_STATUS_NULL_POINTER);

        auto ndim = x_desc->ndim();
        auto size = x_desc->numel();
        auto contiguous = x_desc->isContiguous();
        auto dtype = x_desc->dtype();

        auto shape_unit = x_desc->dim(0);
        auto stride_unit = x_desc->stride(0);
        size_t meta_mem_size = ndim * (sizeof(shape_unit) + sizeof(stride_unit));
        std::vector<size_t> meta(CEIL_DIV(meta_mem_size, sizeof(size_t)));
        
        int8_t *meta_ptr = reinterpret_cast<int8_t *>(meta.data());
        size_t *shape_p = reinterpret_cast<size_t *>(meta_ptr);
        ptrdiff_t *strides_p = reinterpret_cast<ptrdiff_t *>(shape_p + ndim);

        std::memcpy(shape_p, x_desc->shape().data(), ndim * sizeof(*shape_p));
        std::memcpy(strides_p, x_desc->strides().data(), ndim * sizeof(*strides_p));

        return utils::Result<ScalInfo>(ScalInfo(std::move(meta), size, ndim, contiguous, dtype));
    }
};

#endif // __SCAL_H__