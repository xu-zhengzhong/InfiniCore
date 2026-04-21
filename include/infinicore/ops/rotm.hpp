#pragma once

#include "../device.hpp"
#include "common/op.hpp"

#include <tuple>

namespace infinicore::op {
class Rotm {
public:
    using schema = void (*)(Tensor, Tensor, Tensor);
    static void execute(Tensor x, Tensor y, Tensor param);
    static common::OpDispatcher<schema> &dispatcher();
};

std::tuple<Tensor, Tensor> rotm(Tensor x, Tensor y, Tensor param);
void rotm_(Tensor x, Tensor y, Tensor param, Tensor out_x, Tensor out_y);
} // namespace infinicore::op