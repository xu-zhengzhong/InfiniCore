#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Scal {
public:
    using schema = void (*)(Tensor, void *);
    static void execute(Tensor y, void *alpha);
    static common::OpDispatcher<schema> &dispatcher();
};

void scal_(Tensor x, void *alpha);
} // namespace infinicore::op
