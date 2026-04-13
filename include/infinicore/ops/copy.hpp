#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Copy {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor x, Tensor y);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor copy(Tensor x, Tensor y);
void copy_(Tensor x, Tensor y, Tensor out);
} // namespace infinicore::op