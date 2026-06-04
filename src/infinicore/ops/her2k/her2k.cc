#include "infinicore/ops/her2k.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Her2k);

Her2k::Her2k(const Tensor &a, const Tensor &b, const Tensor &alpha, const Tensor &beta, Tensor c, int uplo, int trans) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(a, b, alpha, beta, c);
    INFINICORE_GRAPH_OP_DISPATCH(c->device().getType(), a, b, alpha, beta, c, uplo, trans);
}

void Her2k::execute(const Tensor &a, const Tensor &b, const Tensor &alpha, const Tensor &beta, Tensor c, int uplo, int trans) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Her2k, a, b, alpha, beta, c, uplo, trans);
}

void her2k_(const Tensor &a, const Tensor &b, const Tensor &alpha, const Tensor &beta, Tensor c, int uplo, int trans) {
    Her2k::execute(a, b, alpha, beta, c, uplo, trans);
}

} // namespace infinicore::op
