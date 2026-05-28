#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Tbmv, const Tensor &, Tensor, int, int, int, size_t);

void tbmv_(const Tensor &a, Tensor x, int uplo, int trans, int diag, size_t k);

} // namespace infinicore::op
