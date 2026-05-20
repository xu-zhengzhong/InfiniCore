#ifndef __BANG_COMPLEX_H__
#define __BANG_COMPLEX_H__

#include <math.h>

namespace device::bang {

struct alignas(8) float2 {
    float x;
    float y;
};

using FloatComplex = float2;
using Complex = FloatComplex;

__mlu_func__ static inline float cReal(FloatComplex x) {
    return x.x;
}

__mlu_func__ static inline float cImag(FloatComplex x) {
    return x.y;
}

__mlu_func__ static inline FloatComplex makeFloatComplex(float r, float i) {
    FloatComplex res;
    res.x = r;
    res.y = i;
    return res;
}

__mlu_func__ static inline FloatComplex cConj(FloatComplex x) {
    return makeFloatComplex(cReal(x), -cImag(x));
}

__mlu_func__ static inline FloatComplex cAdd(
    FloatComplex x,
    FloatComplex y) {
    return makeFloatComplex(cReal(x) + cReal(y),
                            cImag(x) + cImag(y));
}

__mlu_func__ static inline FloatComplex cSub(
    FloatComplex x,
    FloatComplex y) {
    return makeFloatComplex(cReal(x) - cReal(y),
                            cImag(x) - cImag(y));
}

__mlu_func__ static inline FloatComplex cMul(
    FloatComplex x,
    FloatComplex y) {
    return makeFloatComplex((cReal(x) * cReal(y)) - (cImag(x) * cImag(y)),
                            (cReal(x) * cImag(y)) + (cImag(x) * cReal(y)));
}

__mlu_func__ static inline FloatComplex cDiv(
    FloatComplex x,
    FloatComplex y) {
    float scale = fabsf(cReal(y)) + fabsf(cImag(y));
    float inv_scale = 1.0f / scale;
    float ar = cReal(x) * inv_scale;
    float ai = cImag(x) * inv_scale;
    float br = cReal(y) * inv_scale;
    float bi = cImag(y) * inv_scale;
    float denom = (br * br) + (bi * bi);
    float inv_denom = 1.0f / denom;
    return makeFloatComplex(((ar * br) + (ai * bi)) * inv_denom,
                            ((ai * br) - (ar * bi)) * inv_denom);
}

__mlu_func__ static inline float cAbs(FloatComplex x) {
    float a = fabsf(cReal(x));
    float b = fabsf(cImag(x));
    float v = a > b ? a : b;
    float w = a > b ? b : a;
    if (v == 0.0f || v > 3.402823466e38f || w > 3.402823466e38f) {
        return v + w;
    }
    float t = w / v;
    return v * sqrtf(1.0f + t * t);
}

__mlu_func__ static inline bool cIsZero(FloatComplex x) {
    return cReal(x) == 0.0f && cImag(x) == 0.0f;
}

__mlu_func__ static inline bool cIsOne(FloatComplex x) {
    return cReal(x) == 1.0f && cImag(x) == 0.0f;
}

} // namespace device::bang

#endif // __BANG_COMPLEX_H__
