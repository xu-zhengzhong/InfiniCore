#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Hpr, const Tensor &, const Tensor &, Tensor, int);

void hpr_(const Tensor &alpha, const Tensor &x, Tensor ap, int uplo);

} // namespace infinicore::op
