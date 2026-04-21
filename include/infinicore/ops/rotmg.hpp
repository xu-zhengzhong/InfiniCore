#pragma once

#include "../device.hpp"
#include "common/op.hpp"

#include <tuple>

namespace infinicore::op {
class Rotmg {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, Tensor);
    static void execute(Tensor d1, Tensor d2, Tensor x1, Tensor y1, Tensor param);
    static common::OpDispatcher<schema> &dispatcher();
};

std::tuple<Tensor, Tensor, Tensor, Tensor> rotmg(
    Tensor d1, Tensor d2, Tensor x1, Tensor y1);
void rotmg_(
    Tensor d1,
    Tensor d2,
    Tensor x1,
    Tensor y1,
    Tensor out_d1,
    Tensor out_d2,
    Tensor out_x1,
    Tensor out_param);
} // namespace infinicore::op