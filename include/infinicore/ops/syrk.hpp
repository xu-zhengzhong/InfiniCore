#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Syrk, const Tensor &, const Tensor &, const Tensor &, Tensor, int, int);

void syrk_(const Tensor &a, const Tensor &alpha, const Tensor &beta, Tensor c, int uplo, int trans);

} // namespace infinicore::op
