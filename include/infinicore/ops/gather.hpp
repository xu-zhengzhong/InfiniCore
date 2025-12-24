#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Gather {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, int);

    static void execute(Tensor output, Tensor input, Tensor index, int dim);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor gather(Tensor input, Tensor index, int dim);

void gather_(Tensor output, Tensor input, Tensor index, int dim);

} // namespace infinicore::op