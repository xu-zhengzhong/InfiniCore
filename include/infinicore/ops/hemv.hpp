#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Hemv, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor, int);

void hemv_(const Tensor &alpha, const Tensor &a, const Tensor &x, const Tensor &beta, Tensor y, int uplo);

} // namespace infinicore::op
