#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Dot {
public:
    using schema = void (*)(void *, const Tensor &, const Tensor &);
    static void execute(void *result, const Tensor &x, const Tensor &y);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor dot(const Tensor &x, const Tensor &y);
} // namespace infinicore::op