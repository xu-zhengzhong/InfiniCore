#include "infinicore/ops/gemv.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Gemv);

Gemv::Gemv(const Tensor &alpha, const Tensor &a, const Tensor &x, const Tensor &beta, Tensor y, int trans) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(alpha, a, x, beta, y);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), alpha, a, x, beta, y, trans);
}

void Gemv::execute(const Tensor &alpha, const Tensor &a, const Tensor &x, const Tensor &beta, Tensor y, int trans) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Gemv, alpha, a, x, beta, y, trans);
}

void gemv_(const Tensor &alpha, const Tensor &a, const Tensor &x, const Tensor &beta, Tensor y, int trans) {
    Gemv::execute(alpha, a, x, beta, y, trans);
}

} // namespace infinicore::op
