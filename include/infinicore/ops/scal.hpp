#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Scal {
public:
    using schema = void (*)(Tensor, Tensor, float);
    static void execute(Tensor y, Tensor x, float alpha);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor scal(Tensor x, float alpha);
void scal_(Tensor y, Tensor x, float alpha);
} // namespace infinicore::op
