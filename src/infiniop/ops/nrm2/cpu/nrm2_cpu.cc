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
    Tdata *result_ptr  = reinterpret_cast<Tdata *>(result);

    const ptrdiff_t n    = info.getSize();
    const ptrdiff_t incx = static_cast<ptrdiff_t>(info.getIncx());

    // Blue's scaling constants (float vs double)
    constexpr Tdata tsml = [] {
        if constexpr (std::is_same_v<Tdata, float>)  return Tdata(0x1p-63f);   // 2^-63
        else                                         return Tdata(0x1p-511);   // 2^-511
    }();
    constexpr Tdata tbig = [] {
        if constexpr (std::is_same_v<Tdata, float>)  return Tdata(0x1p52f);    // 2^52
        else                                         return Tdata(0x1p486);    // 2^486
    }();
    constexpr Tdata ssml = [] {
        if constexpr (std::is_same_v<Tdata, float>)  return Tdata(0x1p75f);    // 2^75
        else                                         return Tdata(0x1p600);    // 2^600
    }();
    constexpr Tdata sbig = [] {
        if constexpr (std::is_same_v<Tdata, float>)  return Tdata(0x1p-76f);   // 2^-76
        else                                         return Tdata(0x1p-601);   // 2^-601
    }();

    Tdata scl = Tdata(1);
    Tdata sumsq = Tdata(0);

    bool notbig = true;
    Tdata asml = Tdata(0);
    Tdata amed = Tdata(0);
    Tdata abig = Tdata(0);

    // 0-based index; handle negative stride
    ptrdiff_t ix = (incx < 0) ? (ptrdiff_t(1) - n) * incx : 0;

    for (ptrdiff_t i = 0; i < n; ++i) {
        Tdata ax = std::abs(x_ptr[ix]);

        if (ax > tbig) {
            const Tdata y = ax * sbig;
            abig += y * y;
            notbig = false;
        } else if (ax < tsml) {
            if (notbig) {
                const Tdata y = ax * ssml;
                asml += y * y;
            }
        } else {
            amed += ax * ax;
        }

        ix += incx;
    }

    if (abig > Tdata(0)) {
        if (amed > Tdata(0) || std::isinf(amed) || std::isnan(amed)) {
            abig += (amed * sbig) * sbig;
        }
        scl = Tdata(1) / sbig;
        sumsq = abig;
    } else if (asml > Tdata(0)) {
        if (amed > Tdata(0) || std::isinf(amed) || std::isnan(amed)) {
            amed = std::sqrt(amed);
            asml = std::sqrt(asml) / ssml;

            const Tdata ymin = std::min(amed, asml);
            const Tdata ymax = std::max(amed, asml);

            scl = Tdata(1);
            sumsq = (ymax * ymax) * (Tdata(1) + (ymin / ymax) * (ymin / ymax));
        } else {
            scl = Tdata(1) / ssml;
            sumsq = asml;
        }
    } else {
        scl = Tdata(1);
        sumsq = amed;
    }

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
        return calculateNrm2<double>(_info, x, result);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::nrm2::cpu