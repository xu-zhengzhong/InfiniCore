#pragma once

#include "../device.hpp"
#include "common/op.hpp"

#include <tuple>

namespace infinicore::op {
class Swap {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor x, Tensor y);
    static common::OpDispatcher<schema> &dispatcher();
};

std::tuple<Tensor, Tensor> swap(Tensor x, Tensor y);
void swap_(Tensor x, Tensor y, Tensor out_x, Tensor out_y);
} // namespace infinicore::op