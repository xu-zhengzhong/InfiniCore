#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Syr2k, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor, int, int);

void syr2k_(const Tensor &a, const Tensor &b, const Tensor &alpha, const Tensor &beta, Tensor c, int uplo, int trans);

} // namespace infinicore::op
