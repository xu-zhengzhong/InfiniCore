#pragma once
#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class SignBit {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor input, Tensor output);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor signbit(Tensor input);
void signbit_(Tensor input, Tensor output);

} // namespace infinicore::op