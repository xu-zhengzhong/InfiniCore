#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Trsm, const Tensor &, const Tensor &, Tensor, int, int, int, int);

void trsm_(const Tensor &a, const Tensor &alpha, Tensor b, int side, int uplo, int trans, int diag);

} // namespace infinicore::op
