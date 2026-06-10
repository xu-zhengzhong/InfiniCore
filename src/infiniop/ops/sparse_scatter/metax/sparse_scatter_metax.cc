#include "sparse_scatter_metax.h"
#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"

namespace op::sparse_scatter::metax {

struct Descriptor::Opaque {
    std::shared_ptr<device::metax::Handle::Internal> internal;
    hcsparseSpVecDescr_t vec_x = nullptr;
    hpccDataType data_type = HPCC_R_32F;
    hcsparseIndexType_t index_type = HCSPARSE_INDEX_64I;

    explicit Opaque(std::shared_ptr<device::metax::Handle::Internal> internal)
        : internal(std::move(internal)) {}

    ~Opaque() {
        if (vec_x != nullptr) {
            hcsparseDestroySpVec(vec_x);
        }
    }
};

static hpccDataType dataTypeOf(infiniDtype_t dtype) {
    switch (dtype) {
    case INFINI_DTYPE_F16:
        return HPCC_R_16F;
    case INFINI_DTYPE_BF16:
        return HPCC_R_16BF;
    case INFINI_DTYPE_F32:
        return HPCC_R_32F;
    case INFINI_DTYPE_F64:
        return HPCC_R_64F;
    default:
        return HPCC_R_32F;
    }
}

static hcsparseIndexType_t indexTypeOf(infiniDtype_t dtype) {
    switch (dtype) {
    case INFINI_DTYPE_I32:
        return HCSPARSE_INDEX_32I;
    case INFINI_DTYPE_I64:
        return HCSPARSE_INDEX_64I;
    default:
        return HCSPARSE_INDEX_64I;
    }
}

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopSpVecDescriptor_t input_desc) {
    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    auto result = SparseScatterInfo::create(output_desc, input_desc);
    CHECK_RESULT(result);
    CHECK_OR_RETURN(result->output_vector.stride == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);

    auto opaque = new Opaque(handle->internal());
    opaque->data_type = dataTypeOf(output_desc->dtype());
    opaque->index_type = indexTypeOf(input_desc->indicesDesc()->dtype());

    auto status = hcsparseCreateSpVec(
        &opaque->vec_x,
        static_cast<int64_t>(result->output_vector.size),
        static_cast<int64_t>(result->nnz),
        const_cast<void *>(input_desc->indices()),
        const_cast<void *>(input_desc->values()),
        opaque->index_type,
        HCSPARSE_INDEX_BASE_ZERO,
        opaque->data_type);
    CHECK_API_OR(status, HCSPARSE_STATUS_SUCCESS, {
        delete opaque;
        return INFINI_STATUS_INTERNAL_ERROR;
    });

    *desc_ptr = new Descriptor(
        output_desc->dtype(),
        input_desc->indicesDesc()->dtype(),
        result.take(),
        input_desc,
        0,
        opaque,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    void *stream) const {
    (void)workspace;
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    hcsparseDnVecDescr_t vec_y = nullptr;
    CHECK_MCSPARSE(hcsparseCreateDnVec(
        &vec_y,
        static_cast<int64_t>(_info.output_vector.size),
        output,
        _opaque->data_type));

    auto ret = _opaque->internal->useMcsparse(
        reinterpret_cast<hcStream_t>(stream),
        [&](hcsparseHandle_t sparse_handle) {
            CHECK_MCSPARSE(hcsparseScatter(
                sparse_handle,
                _opaque->vec_x,
                vec_y));
            return INFINI_STATUS_SUCCESS;
        });
    CHECK_MCSPARSE(hcsparseDestroyDnVec(vec_y));
    CHECK_STATUS(ret);
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::sparse_scatter::metax
