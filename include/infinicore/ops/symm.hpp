#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Symm, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor, int, int);

void symm_(const Tensor &a, const Tensor &b, const Tensor &alpha, const Tensor &beta, Tensor c, int side, int uplo);

} // namespace infinicore::op
