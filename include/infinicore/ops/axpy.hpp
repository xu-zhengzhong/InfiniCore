#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Axpy {
public:
    using schema = void (*)(Tensor, Tensor, void*);
    static void execute(Tensor y, Tensor x, void *alpha);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor axpy(Tensor y, Tensor x, Tensor alpha);
void axpy_(Tensor y, Tensor x, Tensor alpha, Tensor out);
} // namespace infinicore::op
