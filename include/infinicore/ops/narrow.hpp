#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore:: op {

class Narrow {
public:
    using schema = void (*)(Tensor, Tensor, int, int64_t, int64_t);
    
    static void execute(Tensor output, Tensor input, int dim, int64_t start, int64_t length);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor narrow(Tensor input, int dim, int64_t start, int64_t length);

void narrow_(Tensor output, Tensor input, int dim, int64_t start, int64_t length);

} // namespace infinicore::op