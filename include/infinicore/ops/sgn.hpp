#pragma once
#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Sgn {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor input, Tensor output);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor sgn(Tensor input);
void sgn_(Tensor input, Tensor output);

} // namespace infinicore::op