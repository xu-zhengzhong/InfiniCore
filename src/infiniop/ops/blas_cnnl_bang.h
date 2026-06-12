#ifndef __BLAS_CNNL_BANG_H__
#define __BLAS_CNNL_BANG_H__

#include "../devices/bang/common_bang.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace op::blas_cnnl::bang {

class TensorDescriptor {
public:
    cnnlTensorDescriptor_t desc = nullptr;

    TensorDescriptor() = default;
    TensorDescriptor(const TensorDescriptor &) = delete;
    TensorDescriptor &operator=(const TensorDescriptor &) = delete;

    TensorDescriptor(TensorDescriptor &&other) noexcept : desc(other.desc) {
        other.desc = nullptr;
    }

    TensorDescriptor &operator=(TensorDescriptor &&other) noexcept {
        if (this != &other) {
            reset();
            desc = other.desc;
            other.desc = nullptr;
        }
        return *this;
    }

    ~TensorDescriptor() { reset(); }

    void reset() {
        if (desc != nullptr) {
            cnnlDestroyTensorDescriptor(desc);
            desc = nullptr;
        }
    }

    infiniStatus_t create() {
        reset();
        CHECK_BANG(cnnlCreateTensorDescriptor(&desc));
        return INFINI_STATUS_SUCCESS;
    }

    cnnlTensorDescriptor_t get() const { return desc; }
};

class OpTensorDescriptor {
public:
    cnnlOpTensorDescriptor_t desc = nullptr;

    OpTensorDescriptor() = default;
    OpTensorDescriptor(const OpTensorDescriptor &) = delete;
    OpTensorDescriptor &operator=(const OpTensorDescriptor &) = delete;

    OpTensorDescriptor(OpTensorDescriptor &&other) noexcept : desc(other.desc) {
        other.desc = nullptr;
    }

    OpTensorDescriptor &operator=(OpTensorDescriptor &&other) noexcept {
        if (this != &other) {
            reset();
            desc = other.desc;
            other.desc = nullptr;
        }
        return *this;
    }

    ~OpTensorDescriptor() { reset(); }

    void reset() {
        if (desc != nullptr) {
            cnnlDestroyOpTensorDescriptor(desc);
            desc = nullptr;
        }
    }

    infiniStatus_t create(cnnlOpTensorDesc_t op) {
        reset();
        CHECK_BANG(cnnlCreateOpTensorDescriptor(&desc));
        CHECK_BANG(cnnlSetOpTensorDescriptor(
            desc,
            op,
            CNNL_DTYPE_FLOAT,
            CNNL_NOT_PROPAGATE_NAN));
        return INFINI_STATUS_SUCCESS;
    }

    cnnlOpTensorDescriptor_t get() const { return desc; }
};

class ReduceDescriptor {
public:
    cnnlReduceDescriptor_t desc = nullptr;

    ReduceDescriptor() = default;
    ReduceDescriptor(const ReduceDescriptor &) = delete;
    ReduceDescriptor &operator=(const ReduceDescriptor &) = delete;

    ReduceDescriptor(ReduceDescriptor &&other) noexcept : desc(other.desc) {
        other.desc = nullptr;
    }

    ReduceDescriptor &operator=(ReduceDescriptor &&other) noexcept {
        if (this != &other) {
            reset();
            desc = other.desc;
            other.desc = nullptr;
        }
        return *this;
    }

    ~ReduceDescriptor() { reset(); }

    void reset() {
        if (desc != nullptr) {
            cnnlDestroyReduceDescriptor(desc);
            desc = nullptr;
        }
    }

    infiniStatus_t create(cnnlReduceOp_t op) {
        reset();
        CHECK_BANG(cnnlCreateReduceDescriptor(&desc));
        int axis = 0;
        CHECK_BANG(cnnlSetReduceDescriptor_v2(
            desc,
            &axis,
            1,
            op,
            CNNL_DTYPE_FLOAT,
            CNNL_NOT_PROPAGATE_NAN,
            CNNL_REDUCE_NO_INDICES,
            CNNL_32BIT_INDICES,
            0.0f));
        return INFINI_STATUS_SUCCESS;
    }

    cnnlReduceDescriptor_t get() const { return desc; }
};

class MatMulDescriptor {
public:
    cnnlMatMulDescriptor_t desc = nullptr;

    MatMulDescriptor() = default;
    MatMulDescriptor(const MatMulDescriptor &) = delete;
    MatMulDescriptor &operator=(const MatMulDescriptor &) = delete;

    MatMulDescriptor(MatMulDescriptor &&other) noexcept : desc(other.desc) {
        other.desc = nullptr;
    }

    MatMulDescriptor &operator=(MatMulDescriptor &&other) noexcept {
        if (this != &other) {
            reset();
            desc = other.desc;
            other.desc = nullptr;
        }
        return *this;
    }

    ~MatMulDescriptor() { reset(); }

    void reset() {
        if (desc != nullptr) {
            cnnlDestroyMatMulDescriptor(desc);
            desc = nullptr;
        }
    }

    infiniStatus_t create(
        bool trans_a,
        bool trans_b,
        bool use_beta,
        int lda = 0,
        int ldb = 0,
        int ldc = 0) {
        reset();
        CHECK_BANG(cnnlCreateMatMulDescriptor(&desc));
        const cnnlDataType_t compute_type = CNNL_DTYPE_FLOAT;
        const int transa = trans_a ? 1 : 0;
        const int transb = trans_b ? 1 : 0;
        const int beta = use_beta ? 1 : 0;
        CHECK_BANG(cnnlSetMatMulDescAttr(
            desc,
            CNNL_MATMUL_DESC_COMPUTE_TYPE,
            &compute_type,
            sizeof(compute_type)));
        CHECK_BANG(cnnlSetMatMulDescAttr(
            desc,
            CNNL_MATMUL_DESC_TRANSA,
            &transa,
            sizeof(transa)));
        CHECK_BANG(cnnlSetMatMulDescAttr(
            desc,
            CNNL_MATMUL_DESC_TRANSB,
            &transb,
            sizeof(transb)));
        CHECK_BANG(cnnlSetMatMulDescAttr(
            desc,
            CNNL_MATMUL_USE_BETA,
            &beta,
            sizeof(beta)));
        if (lda > 0) {
            CHECK_BANG(cnnlSetMatMulDescAttr(
                desc,
                CNNL_MATMUL_DESC_LDA,
                &lda,
                sizeof(lda)));
        }
        if (ldb > 0) {
            CHECK_BANG(cnnlSetMatMulDescAttr(
                desc,
                CNNL_MATMUL_DESC_LDB,
                &ldb,
                sizeof(ldb)));
        }
        if (ldc > 0) {
            CHECK_BANG(cnnlSetMatMulDescAttr(
                desc,
                CNNL_MATMUL_DESC_LDC,
                &ldc,
                sizeof(ldc)));
        }
        return INFINI_STATUS_SUCCESS;
    }

    cnnlMatMulDescriptor_t get() const { return desc; }
};

class MatMulAlgo {
public:
    cnnlMatMulAlgo_t algo = nullptr;

    MatMulAlgo() = default;
    MatMulAlgo(const MatMulAlgo &) = delete;
    MatMulAlgo &operator=(const MatMulAlgo &) = delete;

    MatMulAlgo(MatMulAlgo &&other) noexcept : algo(other.algo) {
        other.algo = nullptr;
    }

    MatMulAlgo &operator=(MatMulAlgo &&other) noexcept {
        if (this != &other) {
            reset();
            algo = other.algo;
            other.algo = nullptr;
        }
        return *this;
    }

    ~MatMulAlgo() { reset(); }

    void reset() {
        if (algo != nullptr) {
            cnnlDestroyMatMulAlgo(algo);
            algo = nullptr;
        }
    }

    infiniStatus_t create() {
        reset();
        CHECK_BANG(cnnlCreateMatMulAlgo(&algo));
        return INFINI_STATUS_SUCCESS;
    }

    cnnlMatMulAlgo_t get() const { return algo; }
};

class MatMulHeuristicResult {
public:
    cnnlMatMulHeuristicResult_t result = nullptr;

    MatMulHeuristicResult() = default;
    MatMulHeuristicResult(const MatMulHeuristicResult &) = delete;
    MatMulHeuristicResult &operator=(const MatMulHeuristicResult &) = delete;

    MatMulHeuristicResult(MatMulHeuristicResult &&other) noexcept : result(other.result) {
        other.result = nullptr;
    }

    MatMulHeuristicResult &operator=(MatMulHeuristicResult &&other) noexcept {
        if (this != &other) {
            reset();
            result = other.result;
            other.result = nullptr;
        }
        return *this;
    }

    ~MatMulHeuristicResult() { reset(); }

    void reset() {
        if (result != nullptr) {
            cnnlDestroyMatMulHeuristicResult(result);
            result = nullptr;
        }
    }

    infiniStatus_t create() {
        reset();
        CHECK_BANG(cnnlCreateMatMulHeuristicResult(&result));
        return INFINI_STATUS_SUCCESS;
    }

    cnnlMatMulHeuristicResult_t get() const { return result; }
};

inline ptrdiff_t positiveStride(ptrdiff_t stride) {
    return stride < 0 ? -stride : stride;
}

inline size_t vectorBytes(size_t n, infiniDtype_t dtype) {
    return utils::align(n * infiniSizeOf(dtype), ALIGN_SIZE);
}

inline void *workspaceAt(void *workspace, size_t offset) {
    if (offset == 0) {
        return workspace;
    }
    return reinterpret_cast<char *>(workspace) + offset;
}

inline const void *logicalVectorPtr(const void *ptr, size_t n, ptrdiff_t stride, infiniDtype_t dtype) {
    if (stride >= 0 || n == 0) {
        return ptr;
    }
    const auto offset = (static_cast<ptrdiff_t>(1) - static_cast<ptrdiff_t>(n)) * stride;
    return reinterpret_cast<const char *>(ptr) + offset * static_cast<ptrdiff_t>(infiniSizeOf(dtype));
}

inline void *logicalVectorPtr(void *ptr, size_t n, ptrdiff_t stride, infiniDtype_t dtype) {
    return const_cast<void *>(logicalVectorPtr(static_cast<const void *>(ptr), n, stride, dtype));
}

inline const void *logicalMatrixPtr(
    const void *ptr,
    size_t rows,
    size_t cols,
    ptrdiff_t row_stride,
    ptrdiff_t col_stride,
    infiniDtype_t dtype) {
    ptrdiff_t offset = 0;
    if (row_stride < 0 && rows > 0) {
        offset += (static_cast<ptrdiff_t>(1) - static_cast<ptrdiff_t>(rows)) * row_stride;
    }
    if (col_stride < 0 && cols > 0) {
        offset += (static_cast<ptrdiff_t>(1) - static_cast<ptrdiff_t>(cols)) * col_stride;
    }
    return reinterpret_cast<const char *>(ptr) + offset * static_cast<ptrdiff_t>(infiniSizeOf(dtype));
}

inline void *logicalMatrixPtr(
    void *ptr,
    size_t rows,
    size_t cols,
    ptrdiff_t row_stride,
    ptrdiff_t col_stride,
    infiniDtype_t dtype) {
    return const_cast<void *>(logicalMatrixPtr(
        static_cast<const void *>(ptr),
        rows,
        cols,
        row_stride,
        col_stride,
        dtype));
}

inline infiniStatus_t setTensor(
    TensorDescriptor &desc,
    infiniDtype_t dtype,
    const std::vector<int> &dims,
    const std::vector<int> &strides) {
    CHECK_OR_RETURN(dims.size() == strides.size(), INFINI_STATUS_BAD_PARAM);
    if (desc.get() == nullptr) {
        CHECK_STATUS(desc.create());
    }
    CHECK_BANG(cnnlSetTensorDescriptorEx(
        desc.get(),
        CNNL_LAYOUT_ARRAY,
        device::bang::getCnnlDtype(dtype),
        static_cast<int>(dims.size()),
        const_cast<int *>(dims.data()),
        const_cast<int *>(strides.data())));
    return INFINI_STATUS_SUCCESS;
}

inline infiniStatus_t setVectorTensor(
    TensorDescriptor &desc,
    size_t n,
    ptrdiff_t stride,
    infiniDtype_t dtype) {
    return setTensor(
        desc,
        dtype,
        {utils::cast<int>(n)},
        {utils::cast<int>(positiveStride(stride))});
}

inline infiniStatus_t setContiguousVectorTensor(
    TensorDescriptor &desc,
    size_t n,
    infiniDtype_t dtype) {
    return setTensor(desc, dtype, {utils::cast<int>(n)}, {1});
}

inline infiniStatus_t setScalarTensor(TensorDescriptor &desc, infiniDtype_t dtype) {
    return setTensor(desc, dtype, {1}, {1});
}

inline infiniStatus_t setMatrixTensor(
    TensorDescriptor &desc,
    size_t rows,
    size_t cols,
    ptrdiff_t row_stride,
    ptrdiff_t col_stride,
    infiniDtype_t dtype) {
    return setTensor(
        desc,
        dtype,
        {utils::cast<int>(rows), utils::cast<int>(cols)},
        {utils::cast<int>(positiveStride(row_stride)), utils::cast<int>(positiveStride(col_stride))});
}

inline infiniStatus_t copyDeviceToHost(
    void *host,
    const void *device,
    size_t bytes,
    cnrtQueue_t queue) {
    if (queue != nullptr) {
        CHECK_INTERNAL(cnrtMemcpyAsync_V2(host, const_cast<void *>(device), bytes, queue, cnrtMemcpyDevToHost), cnrtSuccess);
        CHECK_INTERNAL(cnrtQueueSync(queue), cnrtSuccess);
    } else {
        CHECK_INTERNAL(cnrtMemcpy(host, const_cast<void *>(device), bytes, cnrtMemcpyDevToHost), cnrtSuccess);
    }
    return INFINI_STATUS_SUCCESS;
}

inline infiniStatus_t copyScalarToFloat(
    float *host,
    const void *device,
    infiniDtype_t dtype,
    cnrtQueue_t queue) {
    CHECK_OR_RETURN(host != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(device != nullptr, INFINI_STATUS_NULL_POINTER);

    switch (dtype) {
    case INFINI_DTYPE_F16: {
        fp16_t value;
        CHECK_STATUS(copyDeviceToHost(&value, device, sizeof(value), queue));
        *host = utils::cast<float>(value);
        return INFINI_STATUS_SUCCESS;
    }
    case INFINI_DTYPE_BF16: {
        bf16_t value;
        CHECK_STATUS(copyDeviceToHost(&value, device, sizeof(value), queue));
        *host = utils::cast<float>(value);
        return INFINI_STATUS_SUCCESS;
    }
    case INFINI_DTYPE_F32:
        CHECK_STATUS(copyDeviceToHost(host, device, sizeof(float), queue));
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

inline infiniStatus_t copyVectorToFloat(
    float *host,
    const void *device,
    size_t n,
    infiniDtype_t dtype,
    cnrtQueue_t queue) {
    CHECK_OR_RETURN(host != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(device != nullptr, INFINI_STATUS_NULL_POINTER);

    switch (dtype) {
    case INFINI_DTYPE_F16: {
        std::vector<fp16_t> values(n);
        CHECK_STATUS(copyDeviceToHost(values.data(), device, values.size() * sizeof(fp16_t), queue));
        for (size_t i = 0; i < n; ++i) {
            host[i] = utils::cast<float>(values[i]);
        }
        return INFINI_STATUS_SUCCESS;
    }
    case INFINI_DTYPE_BF16: {
        std::vector<bf16_t> values(n);
        CHECK_STATUS(copyDeviceToHost(values.data(), device, values.size() * sizeof(bf16_t), queue));
        for (size_t i = 0; i < n; ++i) {
            host[i] = utils::cast<float>(values[i]);
        }
        return INFINI_STATUS_SUCCESS;
    }
    case INFINI_DTYPE_F32:
        CHECK_STATUS(copyDeviceToHost(host, device, n * sizeof(float), queue));
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

inline infiniStatus_t syncQueue(cnrtQueue_t queue) {
    CHECK_INTERNAL(cnrtQueueSync(queue), cnrtSuccess);
    return INFINI_STATUS_SUCCESS;
}

inline infiniStatus_t getCopyWorkspaceSize(
    const std::shared_ptr<device::bang::Handle::Internal> &internal,
    const TensorDescriptor &src,
    const TensorDescriptor &dst,
    size_t *workspace_size) {
    CHECK_STATUS(internal->useCnnl(
        static_cast<cnrtQueue_t>(nullptr),
        [&](cnnlHandle_t handle) {
            CHECK_BANG(cnnlGetCopyWorkspaceSize(
                handle,
                src.get(),
                dst.get(),
                workspace_size));
            return INFINI_STATUS_SUCCESS;
        }));
    return INFINI_STATUS_SUCCESS;
}

inline infiniStatus_t getOpTensorWorkspaceSize(
    const std::shared_ptr<device::bang::Handle::Internal> &internal,
    const TensorDescriptor &a,
    const TensorDescriptor &b,
    const TensorDescriptor &c,
    size_t *workspace_size) {
    const float one = 1.0f;
    const float zero = 0.0f;
    CHECK_STATUS(internal->useCnnl(
        static_cast<cnrtQueue_t>(nullptr),
        [&](cnnlHandle_t handle) {
            CHECK_BANG(cnnlGetOpTensorWorkspaceSize_v3(
                handle,
                &one,
                a.get(),
                &one,
                b.get(),
                &zero,
                c.get(),
                workspace_size));
            return INFINI_STATUS_SUCCESS;
        }));
    return INFINI_STATUS_SUCCESS;
}

inline infiniStatus_t getReduceWorkspaceSize(
    const std::shared_ptr<device::bang::Handle::Internal> &internal,
    const TensorDescriptor &input,
    const TensorDescriptor &output,
    const ReduceDescriptor &reduce,
    size_t *workspace_size) {
    CHECK_STATUS(internal->useCnnl(
        static_cast<cnrtQueue_t>(nullptr),
        [&](cnnlHandle_t handle) {
            CHECK_BANG(cnnlGetReduceOpWorkspaceSize_v2(
                handle,
                input.get(),
                output.get(),
                nullptr,
                reduce.get(),
                workspace_size));
            return INFINI_STATUS_SUCCESS;
        }));
    return INFINI_STATUS_SUCCESS;
}

inline infiniStatus_t initMatMulAlgo(
    const std::shared_ptr<device::bang::Handle::Internal> &internal,
    const MatMulDescriptor &op,
    const TensorDescriptor &a,
    const TensorDescriptor &b,
    const TensorDescriptor &c,
    const TensorDescriptor &d,
    MatMulAlgo &algo,
    MatMulHeuristicResult &heuristic,
    size_t *workspace_size) {
    CHECK_STATUS(algo.create());
    CHECK_STATUS(heuristic.create());

    int count = 0;
    cnnlMatMulHeuristicResult_t results[] = {heuristic.get()};
    CHECK_STATUS(internal->useCnnl(
        static_cast<cnrtQueue_t>(nullptr),
        [&](cnnlHandle_t handle) {
            CHECK_BANG(cnnlGetMatMulAlgoHeuristic(
                handle,
                op.get(),
                a.get(),
                b.get(),
                c.get(),
                d.get(),
                nullptr,
                1,
                results,
                &count));
            return INFINI_STATUS_SUCCESS;
        }));
    CHECK_OR_RETURN(count > 0, INFINI_STATUS_INTERNAL_ERROR);
    CHECK_BANG(cnnlGetMatMulHeuristicResult(heuristic.get(), algo.get(), workspace_size));
    return INFINI_STATUS_SUCCESS;
}

inline infiniStatus_t cnnlCopy(
    const std::shared_ptr<device::bang::Handle::Internal> &internal,
    cnrtQueue_t queue,
    const TensorDescriptor &src_desc,
    const void *src,
    const TensorDescriptor &dst_desc,
    void *dst,
    void *workspace,
    size_t workspace_size) {
    CHECK_STATUS(internal->useCnnl(
        queue,
        [&](cnnlHandle_t handle) {
            CHECK_BANG(cnnlCopy_v2(
                handle,
                src_desc.get(),
                src,
                dst_desc.get(),
                dst,
                workspace,
                workspace_size));
            return INFINI_STATUS_SUCCESS;
        }));
    return INFINI_STATUS_SUCCESS;
}

inline infiniStatus_t cnnlOpTensor(
    const std::shared_ptr<device::bang::Handle::Internal> &internal,
    cnrtQueue_t queue,
    const OpTensorDescriptor &op,
    float alpha1,
    const TensorDescriptor &a_desc,
    const void *a,
    float alpha2,
    const TensorDescriptor &b_desc,
    const void *b,
    void *workspace,
    size_t workspace_size,
    float beta,
    const TensorDescriptor &c_desc,
    void *c) {
    CHECK_STATUS(internal->useCnnl(
        queue,
        [&](cnnlHandle_t handle) {
            CHECK_BANG(cnnlOpTensor(
                handle,
                op.get(),
                &alpha1,
                a_desc.get(),
                a,
                &alpha2,
                b_desc.get(),
                b,
                workspace,
                workspace_size,
                &beta,
                c_desc.get(),
                c));
            return INFINI_STATUS_SUCCESS;
        }));
    return INFINI_STATUS_SUCCESS;
}

inline infiniStatus_t cnnlReduce(
    const std::shared_ptr<device::bang::Handle::Internal> &internal,
    cnrtQueue_t queue,
    const ReduceDescriptor &reduce,
    const TensorDescriptor &input_desc,
    const void *input,
    void *workspace,
    size_t workspace_size,
    const TensorDescriptor &output_desc,
    void *output) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHECK_STATUS(internal->useCnnl(
        queue,
        [&](cnnlHandle_t handle) {
            CHECK_BANG(cnnlReduce_v2(
                handle,
                reduce.get(),
                input_desc.get(),
                input,
                &alpha,
                &beta,
                workspace,
                workspace_size,
                output_desc.get(),
                output,
                nullptr,
                nullptr));
            return INFINI_STATUS_SUCCESS;
        }));
    return INFINI_STATUS_SUCCESS;
}

inline infiniStatus_t cnnlMatMul(
    const std::shared_ptr<device::bang::Handle::Internal> &internal,
    cnrtQueue_t queue,
    const MatMulDescriptor &op,
    const MatMulAlgo &algo,
    float alpha,
    const TensorDescriptor &a_desc,
    const void *a,
    const TensorDescriptor &b_desc,
    const void *b,
    float beta,
    const TensorDescriptor &c_desc,
    void *c,
    void *workspace,
    size_t workspace_size,
    const TensorDescriptor &d_desc,
    void *d) {
    CHECK_STATUS(internal->useCnnl(
        queue,
        [&](cnnlHandle_t handle) {
            CHECK_BANG(cnnlMatMul_v2(
                handle,
                op.get(),
                algo.get(),
                &alpha,
                a_desc.get(),
                a,
                b_desc.get(),
                b,
                &beta,
                c_desc.get(),
                c,
                workspace,
                workspace_size,
                d_desc.get(),
                d));
            return INFINI_STATUS_SUCCESS;
        }));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::blas_cnnl::bang

#endif // __BLAS_CNNL_BANG_H__
