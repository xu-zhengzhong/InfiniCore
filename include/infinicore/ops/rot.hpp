#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Rot {
public:
    using schema = void (*)(Tensor, Tensor, void *, void *);
    static void execute(Tensor x, Tensor y, void *c, void *s);
    static common::OpDispatcher<schema> &dispatcher();
};

void rot_(Tensor x, Tensor y, void *c, void *s);
} // namespace infinicore::op