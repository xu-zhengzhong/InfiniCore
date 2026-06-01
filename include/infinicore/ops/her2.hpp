#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Her2, const Tensor &, const Tensor &, const Tensor &, Tensor, int);

void her2_(const Tensor &alpha, const Tensor &x, const Tensor &y, Tensor a, int uplo);

} // namespace infinicore::op
