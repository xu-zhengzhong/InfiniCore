#include "infinicore/ops/sbmv.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Sbmv);

Sbmv::Sbmv(const Tensor &alpha, const Tensor &a, const Tensor &x, const Tensor &beta, Tensor y, int uplo, size_t k) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(alpha, a, x, beta, y);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), alpha, a, x, beta, y, uplo, k);
}

void Sbmv::execute(const Tensor &alpha, const Tensor &a, const Tensor &x, const Tensor &beta, Tensor y, int uplo, size_t k) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Sbmv, alpha, a, x, beta, y, uplo, k);
}

void sbmv_(const Tensor &alpha, const Tensor &a, const Tensor &x, const Tensor &beta, Tensor y, int uplo, size_t k) {
    Sbmv::execute(alpha, a, x, beta, y, uplo, k);
}

} // namespace infinicore::op
