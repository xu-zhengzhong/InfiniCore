#include "infinicore/ops/ger.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Ger);

Ger::Ger(const Tensor &alpha, const Tensor &x, const Tensor &y, Tensor a) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(alpha, x, y, a);
    INFINICORE_GRAPH_OP_DISPATCH(a->device().getType(), alpha, x, y, a);
}

void Ger::execute(const Tensor &alpha, const Tensor &x, const Tensor &y, Tensor a) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Ger, alpha, x, y, a);
}

void ger_(const Tensor &alpha, const Tensor &x, const Tensor &y, Tensor a) {
    Ger::execute(alpha, x, y, a);
}

} // namespace infinicore::op
