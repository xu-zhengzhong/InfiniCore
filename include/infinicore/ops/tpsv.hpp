#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Tpsv, const Tensor &, Tensor, int, int, int);

void tpsv_(const Tensor &ap, Tensor x, int uplo, int trans, int diag);

} // namespace infinicore::op
