#pragma once

#include "../device.hpp"
#include "common/op.hpp"

#include <tuple>

namespace infinicore::op {
class Rot {
public:
    using schema = void (*)(Tensor, Tensor, void *, void *);
    static void execute(Tensor x, Tensor y, void *c, void *s);
    static common::OpDispatcher<schema> &dispatcher();
};

std::tuple<Tensor, Tensor> rot(Tensor x, Tensor y, void *c, void *s);
void rot_(Tensor x, Tensor y, Tensor out_x, Tensor out_y, void *c, void *s);
} // namespace infinicore::op