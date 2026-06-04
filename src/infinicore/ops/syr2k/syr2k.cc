#include "infinicore/ops/syr2k.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Syr2k);

Syr2k::Syr2k(const Tensor &a, const Tensor &b, const Tensor &alpha, const Tensor &beta, Tensor c, int uplo, int trans) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(a, b, alpha, beta, c);
    INFINICORE_GRAPH_OP_DISPATCH(c->device().getType(), a, b, alpha, beta, c, uplo, trans);
}

void Syr2k::execute(const Tensor &a, const Tensor &b, const Tensor &alpha, const Tensor &beta, Tensor c, int uplo, int trans) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Syr2k, a, b, alpha, beta, c, uplo, trans);
}

void syr2k_(const Tensor &a, const Tensor &b, const Tensor &alpha, const Tensor &beta, Tensor c, int uplo, int trans) {
    Syr2k::execute(a, b, alpha, beta, c, uplo, trans);
}

} // namespace infinicore::op
