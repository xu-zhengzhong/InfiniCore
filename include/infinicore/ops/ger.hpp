#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Ger, const Tensor &, const Tensor &, const Tensor &, Tensor);

void ger_(const Tensor &alpha, const Tensor &x, const Tensor &y, Tensor a);

} // namespace infinicore::op
