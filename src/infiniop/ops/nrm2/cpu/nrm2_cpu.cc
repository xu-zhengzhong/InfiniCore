#include <cmath>
#include <algorithm>
#include <limits>
#include "nrm2_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::nrm2::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto info = Nrm2Info::createNrm2Info(x_desc);
    CHECK_RESULT(info);

    // Create descriptor
    *desc_ptr = new Descriptor(
        info.take(),
        0,
        nullptr,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
infiniStatus_t calculateNrm2(
    const Nrm2Info &info,
    const void *x,
    void *result) {

    const Tdata *x_ptr = reinterpret_cast<const Tdata *>(x);
    Tdata *result_ptr = reinterpret_cast<Tdata *>(result);

    const ptrdiff_t n = info.getSize();
    const size_t incx = info.getIncx();
    
    // // Quick return if possible
    // if (n <= 0) return 0.0f;

    // Blue's scaling constants for IEEE 754 single precision (float)
    // Using hex float literals (C++17) for exact bit-level representation
    constexpr float tsml = 0x1p-63f; // 2^(-63)
    constexpr float tbig = 0x1p52f;  // 2^(52)
    constexpr float ssml = 0x1p75f;  // 2^(75)
    constexpr float sbig = 0x1p-76f; // 2^(-76)

    float scl = 1.0f;
    float sumsq = 0.0f;

    bool notbig = true;
    float asml = 0.0f;
    float amed = 0.0f;
    float abig = 0.0f;

    // Adjust starting index for C++ 0-based arrays and negative strides
    int ix = (incx < 0) ? (1 - n) * incx : 0;

    for (int i = 0; i < n; ++i) {
        float ax = std::abs(x_ptr[ix]);
        if (ax > tbig) {
            abig += (ax * sbig) * (ax * sbig);
            notbig = false;
        } else if (ax < tsml) {
            if (notbig) asml += (ax * ssml) * (ax * ssml);
        } else {
            amed += ax * ax;
        }
        ix += incx;
    }

    // Combine abig and amed or amed and asml if more than one accumulator was used
    if (abig > 0.0f) {
        // Combine abig and amed if abig > 0
        if (amed > 0.0f || std::isinf(amed) || std::isnan(amed)) {
            abig += (amed * sbig) * sbig;
        }
        scl = 1.0f / sbig;
        sumsq = abig;
    } else if (asml > 0.0f) {
        // Combine amed and asml if asml > 0
        if (amed > 0.0f || std::isinf(amed) || std::isnan(amed)) {
            amed = std::sqrt(amed);
            asml = std::sqrt(asml) / ssml;
            
            float ymin = std::min(amed, asml);
            float ymax = std::max(amed, asml);
            
            scl = 1.0f;
            sumsq = (ymax * ymax) * (1.0f + (ymin / ymax) * (ymin / ymax));
        } else {
            scl = 1.0f / ssml;
            sumsq = asml;
        }
    } else {
        // Otherwise all values are mid-range
        scl = 1.0f;
        sumsq = amed;
    }

    // return scl * std::sqrt(sumsq);

    result_ptr[0] = scl * std::sqrt(sumsq);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *x,
    void *result,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;

    switch (_info.getDtype()) {
    case INFINI_DTYPE_F32:
        return calculateNrm2<float>(_info, x, result);
    case INFINI_DTYPE_F64:
        // return calculateNrm2<double>(_info, x, result);
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::nrm2::cpu