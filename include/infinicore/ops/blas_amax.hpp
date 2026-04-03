#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class BlasAmax {
public:
    using schema = void (*)(int *, const Tensor &);
    static void execute(int *result, const Tensor &x);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor blas_amax(const Tensor &x);
} // namespace infinicore::op
