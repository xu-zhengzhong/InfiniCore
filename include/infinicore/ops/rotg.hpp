#pragma once

#include "../device.hpp"
#include "common/op.hpp"

#include <tuple>

namespace infinicore::op {
class Rotg {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor);
    static void execute(Tensor a, Tensor b, Tensor c, Tensor s);
    static common::OpDispatcher<schema> &dispatcher();
};

std::tuple<Tensor, Tensor, Tensor, Tensor> rotg(Tensor a, Tensor b);
void rotg_(Tensor a, Tensor b, Tensor out_a, Tensor out_b, Tensor out_c, Tensor out_s);
} // namespace infinicore::op