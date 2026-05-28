#include "infinicore/ops/herk.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Herk);

Herk::Herk(const Tensor &a, const Tensor &alpha, const Tensor &beta, Tensor c, int uplo, int trans) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(a, alpha, beta, c);
    INFINICORE_GRAPH_OP_DISPATCH(c->device().getType(), a, alpha, beta, c, uplo, trans);
}

void Herk::execute(const Tensor &a, const Tensor &alpha, const Tensor &beta, Tensor c, int uplo, int trans) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Herk, a, alpha, beta, c, uplo, trans);
}

void herk_(const Tensor &a, const Tensor &alpha, const Tensor &beta, Tensor c, int uplo, int trans) {
    Herk::execute(a, alpha, beta, c, uplo, trans);
}

} // namespace infinicore::op
