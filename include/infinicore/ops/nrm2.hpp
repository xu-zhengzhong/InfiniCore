#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Nrm2 {
public:
    using schema = void (*)(void *, const Tensor &);
    static void execute(void *result, const Tensor &x);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor nrm2(const Tensor &x);
} // namespace infinicore::op
