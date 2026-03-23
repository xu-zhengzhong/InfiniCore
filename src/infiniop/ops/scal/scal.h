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
            infiniopTensorDescriptor_t y_desc,                   \
            infiniopTensorDescriptor_t x_desc);                  \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *y,                                             \
            const void *x,                                       \
            const void *alpha,                                   \
            void *stream) const;                                 \
    };                                                           \
    }

class ScalInfo {
private:
    std::vector<size_t> _meta; // Layout: [Shape] [Y_Strides] [X_Strides]
    size_t _size;
    size_t _ndim;
    bool _y_contiguous;
    bool _x_contiguous;
    infiniDtype_t _dtype;

public:
    ScalInfo() = default;

    ScalInfo(std::vector<size_t> meta,
             size_t size,
             size_t ndim,
             bool y_contiguous,
             bool x_contiguous,
             infiniDtype_t dtype)
        : _meta(std::move(meta)), _size(size), _ndim(ndim),
          _y_contiguous(y_contiguous), _x_contiguous(x_contiguous), _dtype(dtype) {}

    inline size_t getSize() const { return _size; }
    inline size_t getNdim() const { return _ndim; }
    inline bool isYContiguous() const { return _y_contiguous; }
    inline bool isXContiguous() const { return _x_contiguous; }
    inline infiniDtype_t getDtype() const { return _dtype; }
    
    inline const size_t *getShape() const { return reinterpret_cast<const size_t *>(_meta.data()); }
    inline const ptrdiff_t *getYStrides() const { return reinterpret_cast<const ptrdiff_t *>(getShape() + _ndim); }
    inline const ptrdiff_t *getXStrides() const { return reinterpret_cast<const ptrdiff_t *>(getYStrides() + _ndim); }

    static utils::Result<ScalInfo> createScalInfo(infiniopTensorDescriptor_t y_desc, 
                                                  infiniopTensorDescriptor_t x_desc) {
        CHECK_OR_RETURN(y_desc != nullptr && x_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(y_desc->dtype() == x_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(y_desc->ndim() == x_desc->ndim(), INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(y_desc->numel() == x_desc->numel(), INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto ndim = y_desc->ndim();
        auto size = y_desc->numel();
        auto y_contiguous = y_desc->isContiguous();
        auto x_contiguous = x_desc->isContiguous();
        auto dtype = y_desc->dtype();

        // Ensure shapes match
        for (size_t i = 0; i < ndim; ++i) {
            CHECK_OR_RETURN(y_desc->dim(i) == x_desc->dim(i), INFINI_STATUS_BAD_TENSOR_SHAPE);
        }

        size_t meta_mem_size = ndim * sizeof(size_t) + 2 * ndim * sizeof(ptrdiff_t);
        std::vector<size_t> meta(CEIL_DIV(meta_mem_size, sizeof(size_t)));
        
        int8_t *meta_ptr = reinterpret_cast<int8_t *>(meta.data());
        size_t *shape_p = reinterpret_cast<size_t *>(meta_ptr);
        ptrdiff_t *y_strides_p = reinterpret_cast<ptrdiff_t *>(shape_p + ndim);
        ptrdiff_t *x_strides_p = reinterpret_cast<ptrdiff_t *>(y_strides_p + ndim);

        std::memcpy(shape_p, y_desc->shape().data(), ndim * sizeof(*shape_p));
        std::memcpy(y_strides_p, y_desc->strides().data(), ndim * sizeof(*y_strides_p));
        std::memcpy(x_strides_p, x_desc->strides().data(), ndim * sizeof(*x_strides_p));

        return utils::Result<ScalInfo>(ScalInfo(std::move(meta), size, ndim, y_contiguous, x_contiguous, dtype));
    }
};

#endif // __SCAL_H__