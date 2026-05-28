#include "infinicore/ops/syrk.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Syrk);

Syrk::Syrk(const Tensor &a, const Tensor &alpha, const Tensor &beta, Tensor c, int uplo, int trans) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(a, alpha, beta, c);
    INFINICORE_GRAPH_OP_DISPATCH(c->device().getType(), a, alpha, beta, c, uplo, trans);
}

void Syrk::execute(const Tensor &a, const Tensor &alpha, const Tensor &beta, Tensor c, int uplo, int trans) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Syrk, a, alpha, beta, c, uplo, trans);
}

void syrk_(const Tensor &a, const Tensor &alpha, const Tensor &beta, Tensor c, int uplo, int trans) {
    Syrk::execute(a, alpha, beta, c, uplo, trans);
}

} // namespace infinicore::op
